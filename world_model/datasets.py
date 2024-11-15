from typing import Dict, List
import os
from glob import glob
from utils import get_paths, get_paths_from_dir
from tqdm import tqdm
from PIL import Image
import numpy as np
import json
import random
from einops import rearrange
import h5py
from matplotlib import pyplot as plt
import scipy.ndimage
from threadpoolctl import threadpool_limits
import sys
import zarr
import copy
from filelock import FileLock
import concurrent.futures
import multiprocessing
import shutil
# from vidaug import augmentors as va

import torch
from torch.utils.data import Dataset
from torchvideotransforms import video_transforms, volume_transforms
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large

from flowdiffusion.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from flowdiffusion.common.replay_buffer import ReplayBuffer
from flowdiffusion.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from flowdiffusion.training_configs import TASK_TO_INSTRUCTION, MODALITY_TO_CHANNEL_NUM
# register_codecs()

random.seed(0)


def visualize_RGB(image1, image2):
    """
    Visualize two (128, 128, 3) RGB images side by side.

    Parameters:
    image1 (numpy.ndarray): The first RGB image array with shape (128, 128, 3).
    image2 (numpy.ndarray): The second RGB image array with shape (128, 128, 3).
    """
    fig, axes = plt.subplots(1, 2)  # Create a figure with two subplots
    
    axes[0].imshow(image1)
    axes[0].axis('off')  # Turn off axis for the first subplot
    axes[0].set_title('obs')  # Set a title for the first subplot
    
    axes[1].imshow(image2)
    axes[1].axis('off')  # Turn off axis for the second subplot
    axes[1].set_title('next_obs')  # Set a title for the second subplot
    
    plt.show()

class CLIPArgs:
    model_name: str = "ViT-L/14@336px"
    skip_center_crop: bool = True
    batch_size: int = 64

    @classmethod
    def id_dict(cls):
        """Return dict that identifies the CLIP model parameters."""
        return {
            "model_name": cls.model_name,
            "skip_center_crop": cls.skip_center_crop,
        }


class Datasethdf5RGB(Dataset):
    def __init__(self, 
                 tokenizer, 
                 text_encoder,
                 dataset_root='../datasets/', 
                 task='coffee_d2',
                 num_input_frame=2, 
                 num_output_frame=2, 
                 frame_skip=1, # defines a the common difference between two indices, a = frame_skip*n
                 main_camera='spaceview', 
                 modality='rgb',
                 seed=42,
                 val_ratio=0.01,
                 max_train_episodes=None,
                 ):
        assert modality in ['rgb', 'rgbd', 'depth'], "ERROR: Illegal input modality"

        self.frame_skip = frame_skip
        self.pad_before = (num_input_frame - 1)
        self.pad_after = (num_output_frame - 1)

        print('Loading cached ReplayBuffer from Disk.')
        num_channel = MODALITY_TO_CHANNEL_NUM[modality]
        shape_meta = {'obs':{f'{main_camera}_{modality}': {'shape':[num_channel,128,128], 'type':modality}}}
        dataset_path = os.path.join(dataset_root, f'{task}_{main_camera}_rel.hdf5')

        dataset_buffer = None
        data_modality = 'rgbd'
        cache_zarr_path = os.path.join(dataset_root, f'{task}_{main_camera}_{data_modality}_rel.hdf5.zarr.zip')
        cache_lock_path = cache_zarr_path + '.lock'
        print('Acquiring lock on cache.')
        with FileLock(cache_lock_path):
            print(1)
            
        with FileLock(cache_lock_path):
            if not os.path.exists(cache_zarr_path):
                # cache does not exists
                try:
                    print('Cache does not exist. Creating!')
                    # store = zarr.DirectoryStore(cache_zarr_path)
                    dataset_buffer = _convert_robomimic_to_replay(
                        store=zarr.MemoryStore(), 
                        shape_meta=shape_meta, 
                        dataset_path=dataset_path)
                    print('Saving cache to disk.')
                    with zarr.ZipStore(cache_zarr_path) as zip_store:
                        dataset_buffer.save_to_store(
                            store=zip_store
                        )
                except Exception as e:
                    shutil.rmtree(cache_zarr_path)
                    raise e
            else:
                print('Loading cached ReplayBuffer from Disk.')
                with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                    dataset_buffer = ReplayBuffer.copy_from_store(
                        src_store=zip_store, store=zarr.MemoryStore())
                print('Loaded!')
        print('Loaded!')

        val_mask = get_val_mask(
            n_episodes=dataset_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        
        self.sequence_length = num_input_frame + num_output_frame
        sampler = SequenceSampler(
            replay_buffer=dataset_buffer, 
            sequence_length=self.sequence_length,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=train_mask,\
            frame_skip=frame_skip)
        
        instruction = TASK_TO_INSTRUCTION[task]
        batch_text_ids = tokenizer([instruction], return_tensors = 'pt', padding = True, truncation = True, max_length = 128)
        batch_text_embed = text_encoder(**batch_text_ids).last_hidden_state
        self.text_emb = batch_text_embed[0]
        
        self.replay_buffer = dataset_buffer
        self.val_mask = val_mask
        self.sampler = sampler
        self.modality = modality
        self.num_input_frame = num_input_frame
        self.num_output_frame = num_output_frame
        self.obs_dataset_key = f"{main_camera}_{data_modality}"
        self.obs_key = f"{main_camera}_{modality}"

        print('Done')

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.sequence_length,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
            frame_skip=self.frame_skip
            )
        val_set.train_mask = self.val_mask
        return val_set

    def __len__(self):
        return len(self.sampler)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        obs_sequence = np.moveaxis(data[self.obs_dataset_key],-1,1).astype(np.float32) / 255.
        if self.obs_key.endswith('rgb'):
            obs_sequence = obs_sequence[:,:3]
        elif self.obs_key.endswith('depth'):
            obs_sequence = obs_sequence[:,3:4]
        

        cond = rearrange(obs_sequence[:self.num_input_frame], "f c h w -> (f c) h w")
        target = rearrange(obs_sequence[self.num_input_frame:], "f c h w -> (f c) h w")
        text_emb = self.text_emb

        return torch.from_numpy(target), torch.from_numpy(cond), text_emb



def visualize_depth_image(rgb_image, depth_channel):
    import cv2
    from matplotlib import pyplot as plt
    # Display RGB image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    plt.title('RGB Image')
    plt.axis('off')

    # Display Depth image
    plt.subplot(1, 2, 2)
    plt.imshow(depth_channel[:,:,0], cmap='gray', vmin=0.0, vmax=1.0)
    plt.title('Depth Channel')
    plt.axis('off')

    plt.show()



def visualize_flow(rgb_image1, rgb_image2, optical_flow):
    x = np.arange(0, 128, 1)
    y = np.arange(0, 128, 1)
    x, y = np.meshgrid(x, y)

    # Extracting flow components
    u = optical_flow[:,:,0]
    v = optical_flow[:,:,1]

    # Plotting the images
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plotting the RGB image
    axes[0].imshow(rgb_image1)
    axes[0].set_title('Current obs')

    axes[1].imshow(rgb_image2)
    axes[1].set_title('Next obs')

    # Plotting the optical flow vectors
    axes[2].imshow(np.zeros((128, 128)), cmap='gray')  # Displaying a blank image
    axes[2].quiver(x, y, u, v, color='r', angles='xy', scale_units='xy', scale=1)
    axes[2].set_title('Optical Flow Visualization')

    plt.show()


def visualize_voxel(np_voxels):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    indices = np.argwhere(np_voxels[0] != 0)
    colors = np_voxels[1:, indices[:, 0], indices[:, 1], indices[:, 2]].T

    ax.scatter(indices[:, 0], indices[:, 1], indices[:, 2], c=colors, marker='s')

    # Set labels and show the plot
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 32)
    ax.set_zlim(0, 32)  

    plt.savefig('/home/yilong/Desktop/full3d.png')
    plt.show()


class Datasethdf5Voxel(Dataset):
    def __init__(self, path='../datasets/', sample_per_seq=2, frame_skip=3, demo_percentage=1.0, validation=False):
        print("Preparing Voxel data from zarr dataset ...")
        
        self.frame_skip = frame_skip
        self.sample_per_seq = sample_per_seq
        self.sequence_paths = []

        # Find all HDF5 files in the directory
        sequence_dirs = glob(f"{path}/**/*.hdf5", recursive=True)

        for seq_dir in sequence_dirs:
            # Check if the corresponding Zarr file exists
            zarr_path = seq_dir.replace('.hdf5', '.zarr')
            if not os.path.exists(zarr_path):
                # Convert HDF5 to Zarr if Zarr file does not exist
                hdf5_to_zarr(seq_dir)
        
        # Collect all Zarr files
        self.zarr_files = glob(f"{path}/**/*.zarr", recursive=True)
        
        # Open each Zarr file and store the root groups
        self.stores = [zarr.DirectoryStore(zarr_file) for zarr_file in self.zarr_files]
        self.roots = [zarr.open(store, mode='r') for store in self.stores]
        
        self.tasks = []
        
        # Collect all observation data paths
        for zarr_file, root in zip(self.zarr_files, self.roots):
            task = zarr_file.split("/")[-2].replace('_', ' ')
            if validation:
                demos = list(root['data'].keys())[int(len(root['data'].keys())//(1/demo_percentage)):]
            else:
                demos = list(root['data'].keys())[:int(len(root['data'].keys())//(1/demo_percentage))]
            for demo in demos:
                data = root['data'][demo]
                obs_frames = len(data['obs']['voxels'])
                for i in range(obs_frames - self.sample_per_seq * (self.frame_skip + 1)):
                    self.sequence_paths.append((root, demo, i, task))

    def get_samples(self, root, demo, index):
        obs_seq = []
        for i in range(self.sample_per_seq):
            obs = root['data'][demo]['obs']['voxels'][index + i*(self.frame_skip + 1)] / 255.0

            obs_seq.append(obs)

        return obs_seq

    def __len__(self):
        return len(self.sequence_paths)

    def __getitem__(self, idx):
        root, demo, index, task = self.sequence_paths[idx]
        obs_seq = self.get_samples(root, demo, index)
        
        obs_seq = [torch.from_numpy(obs).float() for obs in obs_seq]
        x_cond = obs_seq[0]
        x = torch.cat(obs_seq[1:], dim=0)

        return x, x_cond, task
    


def _convert_robomimic_to_replay(store, shape_meta, dataset_path,
        n_workers=None, max_inflight_tasks=None):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()-1
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    

    # parse shape_meta
    rgb_keys = list()
    voxel_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'rgbd':
            rgb_keys.append(key)
        elif type == 'voxels':
            voxel_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)
    
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file['data']
        len_demos = len(demos)

        episode_ends = list()
        prev_end = 0
        for i in range(len_demos):
            demo = demos[f'demo_{i}']
            episode_length = demo['actions'].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array('episode_ends', episode_ends, 
            dtype=np.int64, compressor=None, overwrite=True)

        
        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False
        
        
        obs_keys = rgb_keys if len(voxel_keys)==0 else voxel_keys
        with tqdm(total=n_steps*(len(obs_keys)), desc="Loading image data", mininterval=1.0) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in obs_keys:
                    data_key = 'obs/' + key
                    shape = tuple(shape_meta['obs'][key]['shape'])
                    if len(voxel_keys)==0:
                        c,h,w = shape
                        obs_shape = (n_steps,h,w,c)
                        obs_chunk = (1,h,w,c)
                        this_compressor = Jpeg2k(level=50)
                    else:
                        c,x,y,z = shape
                        obs_shape = (n_steps,c,x,y,z)
                        obs_chunk = (1,c,x,y,z)
                        this_compressor = None
                    
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=obs_shape,
                        chunks=obs_chunk,
                        compressor=this_compressor,
                        dtype=np.uint8
                    )
                    for episode_idx in range(len_demos):
                        demo = demos[f'demo_{episode_idx}']
                        
                        hdf5_arr = demo['obs'][key]

                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(futures, 
                                    return_when=concurrent.futures.FIRST_COMPLETED)
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError('Failed to encode image!')
                                pbar.update(len(completed))
                                

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(
                            executor.submit(img_copy, 
                                img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer