import math
import copy
import os
from datetime import datetime
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import numpy as np
import wandb
from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate import Accelerator
from pynvml import *

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam
from torchvision import transforms as T, utils

from flowdiffusion.model.augmentors import CropRandomizer, RandomAffineAugmentor, RandomAffineAndShiftAugmentor
from flowdiffusion.utils import save_videos

__version__ = "0.0"
# trainer class

import cv2

def exists(x):
    return x is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def flow_to_RGB(flow_tensor):
    batch_size = flow_tensor.shape[0]
    rgb_images = []
    
    for i in range(batch_size):
        flow = flow_tensor[i, 0].numpy()
        
        flow_x, flow_y = flow[0], flow[1]
        magnitude, angle = cv2.cartToPolar(flow_x, flow_y, angleInDegrees=True)
        magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
        hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.float32)
        hsv[..., 0] = angle / 2  # OpenCV uses [0, 180] for hue, divide by 2 to fit [0, 360] into [0, 180]
        hsv[..., 1] = magnitude
        hsv[..., 2] = 1
        rgb_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        rgb_image = (rgb_image * 255).astype(np.uint8)
        rgb_images.append(torch.tensor(rgb_image).permute(2, 0, 1).unsqueeze(0))

    rgb_tensor = torch.cat(rgb_images).unsqueeze(1)
    
    return rgb_tensor

def get_blend_visualization(init_cond, predictions):
    b, n, c, h, w = predictions.shape
    image = 0.5 * init_cond
    blend_weight = 0.5/n
    for i_frame in range(n):
        image += blend_weight*predictions[:,i_frame]
    return image

class Trainer(object):
    def __init__(
        self,
        cfg,
        diffusion_model,
        tokenizer, 
        text_encoder, 
        train_dl,
        valid_ds,
        results_folder,
        accelerator,
        image_size,
        num_input_frame=2,
        num_output_frame=2,
        image_original_channels = 3,
        *,
        train_batch_size = 1,
        valid_batch_size = 1,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_epochs = 1,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every_epoch = 10,
        num_samples = 3,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048, 
        cond_drop_chance=0.1,
        num_eval_frame=80,
        enable_affine_augmentation=False,
        resize_shape=None,
        crop_shape=None,
        evaluation_flag=False
    ):
        super().__init__()

        self.accelerator = accelerator
        self.dl = train_dl
        self.cond_drop_chance = cond_drop_chance
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.model = diffusion_model
        self.channels = image_original_channels
        self.num_input_frame = num_input_frame
        self.num_output_frame = num_output_frame
        self.num_samples = num_samples # assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.save_and_sample_every_epoch = save_and_sample_every_epoch
        self.batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_epochs = train_num_epochs
        self.results_folder = results_folder
        self.valid_dl = valid_ds
        self.num_eval_frame = num_eval_frame

        
        # configure logging
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
        if accelerator.is_main_process and not evaluation_flag:
            self.wandb_run = wandb.init(
                project='video_prediction',
                resume=True,
                mode='online',
                name=f"{formatted_time}-{cfg['task']}-{cfg['modality']}-{cfg['main_camera']}-skip{cfg['frame_skip']}-demo{cfg['num_train_demo']}-diffuse{cfg['eval_diffusion_steps']}-{cfg['num_input_frame']}to{cfg['num_output_frame']}",
                dir=self.results_folder,
                config=cfg,
            )
            wandb.config.update(
                {
                    "output_dir": self.results_folder,
                })

        self.step = 0
        self.epoch = 0
        self.train_loss = []
        self.valid_mse = []

        
        # dataset and dataloader
        # dl = DataLoader(train_set, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 4)
        self.total_step_per_epoch = len(train_dl)
        # dl = self.accelerator.prepare(dl)
        self.dl = cycle(train_dl)
        # self.valid_dl = DataLoader(valid_set, batch_size = valid_batch_size, shuffle = True, pin_memory = True, num_workers = 4)

        # InceptionV3 for fid-score computation
        self.inception_v3 = None
        if calculate_fid:
            assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
            self.inception_v3 = InceptionV3([block_idx])
            self.inception_v3.to(self.device)

        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)


        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt, self.text_encoder = \
            self.accelerator.prepare(self.model, self.opt, self.text_encoder)
        
        this_resizer = nn.Identity()
        if resize_shape is not None:
            this_resizer = T.Resize(
                        size=(resize_shape[0],resize_shape[1]))
            image_size = tuple(resize_shape)
        this_affine_randomizer = nn.Identity()

        if enable_affine_augmentation:
            channels = image_original_channels * num_output_frame
            input_shape = (channels,) + image_size # (C, H, W)
            this_affine_randomizer = RandomAffineAndShiftAugmentor(input_shape, crop_shape[0], crop_shape[1])
        
        self.image_size = image_size
        self.transform_map = nn.Sequential(this_resizer, this_affine_randomizer)
        self.evaluation_flag = evaluation_flag

    @property
    def device(self):
        return self.accelerator.device

    def save(self, epoch, model_only=False):
        if not self.accelerator.is_local_main_process:
            return

        if model_only:
            torch.save(self.accelerator.get_state_dict(self.model), os.path.join(self.results_folder, f'model-epoch{epoch}.pt'))
        else:
            data = {
                'step': self.step,
                'model': self.accelerator.get_state_dict(self.model),
                'opt': self.opt.state_dict(),
                'ema': self.ema.state_dict(),
                'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
                'version': __version__
            }

            torch.save(data, os.path.join(self.results_folder, f'trainer-epoch{epoch}.pt'))

    def load(self, milestone, model_only=False):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(os.path.join(self.results_folder, f'{milestone}.pt'), map_location=device)
        model = self.accelerator.unwrap_model(self.model)
        if model_only:
            model.load_state_dict(data)
        else:
            model.load_state_dict(data['model'])

            self.step = data['step']
            self.opt.load_state_dict(data['opt'])
            if self.accelerator.is_main_process:
                self.ema.load_state_dict(data["ema"])

            if 'version' in data:
                print(f"loading from version {data['version']}")

            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])

    #     return fid_value
    def encode_batch_text(self, batch_text):
        batch_text_ids = self.tokenizer(batch_text, return_tensors = 'pt', padding = True, truncation = True, max_length = 128).to(self.device)
        batch_text_embed = self.text_encoder(**batch_text_ids).last_hidden_state
        return batch_text_embed

    def sample(self, x_conds, batch_text, batch_size=1, guidance_weight=0):
        device = self.device
        task_embeds = self.encode_batch_text(batch_text)
        return self.ema.ema_model.sample(x_conds.to(device), task_embeds.to(device), batch_size=batch_size, guidance_weight=guidance_weight)

    def visualize_depth_image(self, rgb_image, depth_channel):
        import cv2
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(rgb_image[0], cv2.COLOR_BGR2RGB))
        plt.title('RGB Image')
        plt.axis('off')
        print(depth_channel.shape)
        # Display Depth image
        plt.subplot(1, 2, 2)
        plt.imshow(depth_channel, cmap='gray', vmin=0.0, vmax=1.0)
        plt.title('Depth Channel')
        plt.axis('off')

        plt.show()

    def predict_rollout(self, init_obs, text_emb, n_target=80):
        # fill value with Nan to catch bugs
        # the non-loaded region should never be used
        # create a (ENV_NUM,N_OBS,C,H,W)
        batch_size = init_obs.shape[0]
        C, H, W = init_obs.shape[-3:]
        init_obs = rearrange(init_obs, 'b n c h w -> b (n c) h w')
        
        predict_seq = torch.full([batch_size] + [n_target] + [C,H,W], 
                        fill_value=np.nan, dtype=init_obs.dtype)
        # init_obs is (N_NUM,1,C,H,W)
        next_frame = init_obs
        n_rollout = n_target // self.num_output_frame 
        n_rollout = n_rollout + 1 if n_rollout == 0 else n_rollout
        n_rollout_reminder = n_target % self.num_output_frame
        # n_rollout_reminder = n_rollout + 1 if n_rollout == 0 else n_rollout
        for step in tqdm(range(0, n_rollout), desc = f'Video rollout {n_target} frames with {self.num_input_frame}-to-{self.num_output_frame} model', total = n_rollout):
            next_frame = self.ema.ema_model.sample(next_frame[:,-self.num_input_frame*C:,...], text_emb, batch_size, guidance_weight=0)
            chunk_start = step * self.num_output_frame
            chunk_length = n_rollout_reminder if (step==(n_rollout-1) and n_rollout_reminder!=0) else self.num_output_frame 
            predict_seq[:,chunk_start:chunk_start+chunk_length,...] = rearrange(next_frame, 'b (n c) h w -> b n c h w', n=self.num_output_frame)[:,:chunk_length,...]
        return predict_seq
    
    def preprocess_batch(self, x, x_cond):
        num_img_channel, num_img_cond_channel = x.shape[1], x_cond.shape[1]
        x_and_x_cond = torch.cat([x, x_cond], axis=1) # cat here for consistent transform map
        x_and_x_cond = self.transform_map(x_and_x_cond)
        x, x_cond = x_and_x_cond[:,:num_img_channel], x_and_x_cond[:,num_img_channel:] 
        return x, x_cond

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        total_loss = 0.
        while self.epoch < self.train_num_epochs:
            local_step = 0
            pbar = tqdm(initial=local_step, total=self.total_step_per_epoch, disable= not accelerator.is_main_process)
            for local_step in range(self.total_step_per_epoch):

                for _ in range(self.gradient_accumulate_every):
                    x, x_cond, goal_embed = next(self.dl)
                    x, x_cond = x.to(device), x_cond.to(device)
                    x, x_cond = self.preprocess_batch(x, x_cond)
                    # goal_embed = self.encode_batch_text(goal)
                    ### zero whole goal_embed if p < self.cond_drop_chance
                    # goal_embed = goal_embed * (torch.rand(goal_embed.shape[0], 1, 1, device = goal_embed.device) > self.cond_drop_chance).float()

                    with self.accelerator.autocast():
                        loss = self.model(x, x_cond, goal_embed)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                        self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                scale = self.accelerator.scaler.get_scale()
                pbar.set_description(f'Training in epoch {self.epoch} / {self.train_num_epochs} | loss: {total_loss:.4E}, loss scale: {scale:.1E}')
                step_log = {'train_loss': loss.item(),
                            'global_step': self.step,
                            'epoch': self.epoch}
                if self.accelerator.is_main_process and not self.evaluation_flag:
                    self.wandb_run.log(step_log, step=self.step)
                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                pbar.update(1)
                
                self.train_loss.append(total_loss)
                if accelerator.is_main_process:
                    self.ema.update()
                
                    
            self.epoch += 1

            # eval
            if accelerator.is_main_process:
                if self.step != 0 and self.epoch % self.save_and_sample_every_epoch == 0:
                    valid_mse, img_logs, video_logs = self.evaluation(device)
                    step_log['valid_mse'] = valid_mse
                    step_log.update(img_logs)
                    step_log.update(video_logs)
                    if self.accelerator.is_main_process and not self.evaluation_flag:
                        self.wandb_run.log(step_log, step=self.step)

            accelerator.wait_for_everyone()
            pbar.update(1)

        accelerator.print('training complete')
        accelerator.end_training()

    def evaluation(self, device=None):
        self.ema.eval()
        self.ema.ema_model.eval()
        with torch.no_grad():
            batches = num_to_groups(self.num_samples, self.valid_batch_size)
            conds = []
            targets = []
            task_embeds = []
            for i, (x, x_cond, label) in enumerate(self.valid_dl):
                x, x_cond = self.preprocess_batch(x, x_cond)
                targets.append(x)
                conds.append(x_cond.to(device))
                task_embeds.append(label.to(device))
            targets, conds, task_embeds = targets[:len(batches)], conds[:len(batches)], task_embeds[:len(batches)]
                
            with self.accelerator.autocast():
                predicts_list = list(map(lambda n, c, e: self.ema.ema_model.sample(batch_size=n, x_cond=c, task_embed=e), batches, conds, task_embeds))

            img_logs = dict()
            print_gpu_utilization()
            gt_targets = torch.cat(targets, dim = 0)[:self.num_samples] # [batch_size, 3*n, 120, 160]
            # gt_targets = self.transform_map(gt_targets)
            gt_targets = rearrange(gt_targets, 'b (n c) h w -> b n c h w', n=self.num_output_frame)
            conds = torch.cat(conds, dim = 0).detach().cpu()[:self.num_samples]
            conds = rearrange(conds, 'b (n c) h w -> b n c h w', n=self.num_input_frame)
            
            gt_visualization = get_blend_visualization(conds[:,-1], gt_targets)
            # gt_visualization = 0.6*conds[:,-1] + 0.3 * gt_targets[:,0] + 0.1 * gt_targets[:,1]
            gt_rgbs = gt_visualization[:,:3]
            gt_rgb_path = os.path.join(self.results_folder, 'imgs', f'gt-rgb-{self.epoch}.png')
            gt_depth_path = os.path.join(self.results_folder, 'imgs', f'gt-depth-{self.epoch}.png')
            utils.save_image(gt_rgbs, gt_rgb_path)
            img_logs['gt-rgb'] = wandb.Image(gt_rgb_path)
            if self.channels == 4:
                gt_depths = gt_visualization[:, 3:4]
                utils.save_image(gt_depths, gt_depth_path)
                img_logs['gt-depth'] = wandb.Image(gt_depth_path)

            predicts = torch.cat(predicts_list, dim = 0).detach().cpu()
            predicts = rearrange(predicts, 'b (n c) h w -> b n c h w', n=self.num_output_frame)
            # predicts_visualization = 0.6*conds[:,-1] + 0.3 * predicts[:,0] + 0.1 * predicts[:,1]
            predicts_visualization = get_blend_visualization(conds[:,-1], predicts)
            predicts_rgbs = predicts_visualization[:,:3]
            predict_rgb_path = os.path.join(self.results_folder, 'imgs', f'predict-rgb-{self.epoch}.png')
            predict_depth_path = os.path.join(self.results_folder, 'imgs', f'predict-depth-{self.epoch}.png')
            utils.save_image(predicts_rgbs, predict_rgb_path)
            img_logs['predict-rgb'] = wandb.Image(predict_rgb_path)
            save_depth=False
            if self.channels == 4:
                predicts_depths = predicts_visualization[:, 3:4]
                utils.save_image(predicts_depths, predict_depth_path)
                img_logs['predict-depth'] = wandb.Image(predict_depth_path)
                save_depth=True

            task_embeds = torch.cat(task_embeds, dim = 0)
            predict_seqs = self.predict_rollout(conds.to(device), task_embeds, n_target=self.num_eval_frame)
            predict_videos = np.clip(predict_seqs.numpy()*255, 0, 255).astype(np.uint8)
            video_logs = save_videos(os.path.join(self.results_folder, 'eval_videos', f'epoch{self.epoch}'), predict_videos, save_depth=save_depth)

            valid_mse = float(torch.mean((gt_targets - predicts)**2))
            self.valid_mse.append(valid_mse)
            

            self.save(self.epoch, model_only=True)
            return valid_mse, img_logs, video_logs