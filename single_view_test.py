import torch
from world_model.datasetmultiview import DatasetMultiview
from torch.utils.data import DataLoader
from trainer import VWMTrainer
from tqdm import tqdm
import lightning as L
from argparse import ArgumentParser
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import os
from glob import glob

parser = ArgumentParser()
parser.add_argument("--bs", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--mask_ratio", type=float, default=0.5)
parser.add_argument("--use_ar", action="store_true")
parser.add_argument("--use_mv", action="store_true")
parser.add_argument(
    "--mae_type", type=str, default="base", choices=["base", "large", "huge"]
)
args = parser.parse_args()

mae_ckpt_path = f"/users/zli419/data/users/zli419/SSL/SSL_world_model/pretrained/MAE/mae_visualize_vit_{args.mae_type}.pth"
action_dim = 7

test_dataset = DatasetMultiview(
    dataset_root="/users/zli419/data/users/zli419/SSL/SSL_world_model",
    mode="val",
    enable_multi_view=False,
)

if args.use_mv:
    model_ckpt = "/users/zli419/data/users/zli419/SSL/SSL_world_model/SSL_final_project/ihmqxa47/checkpoints/epoch=0499.ckpt"
    save_dir = f"logs/mask_50"
else:
    model_ckpt = "/users/zli419/data/users/zli419/SSL/SSL_world_model/SSL_final_project/hiucv70i/checkpoints/epoch=0499.ckpt"
    save_dir = f"logs/single_view"
    
VWM_trainer = VWMTrainer.load_from_checkpoint(model_ckpt,
                                              mae_ckpt_path=mae_ckpt_path,
                                              action_dim=action_dim,
                                              save_dir=save_dir,
                                              intrinsic=test_dataset.intrinsic,
                                              img_size=224,
                                              mask_ratio=args.mask_ratio,
                                              lr=1e-5,
                                              device="cuda")

for demo_idx in range(test_dataset.num_demos):
    for cam_idx in range(len(test_dataset.cam_keys)):
        print(f"Testing demo_idx: {demo_idx}, cam_idx: {cam_idx}")
        seq, action, cam, cam_name, demo_name = test_dataset.get_seq(demo_idx, cam_idx)
        seq = seq.cuda()
        action = torch.tensor(action).float().cuda()
        cam = torch.stack([cam] * len(seq)).cuda()

        VWM_trainer.test_single_view(seq, cam, action, cam_name, demo_name, is_ar=args.use_ar)