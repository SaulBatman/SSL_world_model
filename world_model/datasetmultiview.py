import os
import cv2
import h5py
import yaml
import json
import torch
import random
import numpy as np

from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

random.seed(0)

INTRINSICS = {128: np.array([[154.50966799, 0.,  64.],
                             [0., 154.50966799, 64.],
                             [0. ,0., 1.]]),
              224: np.array([[270.39191899, 0., 112.],
                             [0., 270.39191899, 112.],
                             [0., 0.,1.]])}


def get_projs(cam_extrs, intr):
    """
    Calculate camera projection matrices.
    
    Params
    ------
    cam_extrs: dict {cam_name -> [x, y, z, qw, qx, qy, qz]}
    intr: np.array [3, 3]

    Return
    ------
    {camera name: projection matrix}
    """
    projs = {}

    K = np.eye(4)
    K[:3, :3] = intr
    K[3, 3] = 1 

    for cam_name, params in cam_extrs.items():
        x, y, z, qw, qx, qy, qz = params
        rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
        extr = np.eye(4)
        extr[:3, :3] = rotation
        extr[:3, 3] = [x, y, z]
        
        projs[cam_name] = K @ extr

    return projs

# https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/common/sampler.py
def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


class DatasetMultiview(Dataset):
    def __init__(self, 
                 dataset_root='./data/', 
                 cam_file='camera_pose_dict.npy',
                 task='square_d0',
                 mode='train',
                 img_size=224,
                 seed=42,
                 val_ratio=0.01):

        self.mode = mode
        assert self.mode in ['train', 'val'], 'ERROR: mode has to be train or val'
        self.data_root = os.path.join(dataset_root, task)

        # load camera data
        cam_data = np.load(os.path.join(self.data_root, cam_file), allow_pickle=True).item()
        self.cam_keys = list(cam_data.keys())

        intrinsic = INTRINSICS[img_size]
        self.projs = get_projs(cam_data, intrinsic)

        demo_dirs = glob(os.path.join(self.data_root, 'demo_*'))
        # set certain demos to train and certain to val
        val_mask = get_val_mask(
            n_episodes=len(demo_dirs), 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        mask = train_mask if mode=='train' else val_mask

        self.actions = {}
        indices = []
        for i, demo_dir in enumerate(demo_dirs):
            if not mask[i]:
                continue
            
            demo_name = os.path.basename(demo_dir)
            # store the associated action array for the demo
            self.actions[demo_name] = np.load(os.path.join(demo_dir, 'actions.npy'))
            # get all of the frames
            num_frames = len(glob(os.path.join(demo_dir, 'obs', 'camera0_image', '*.jpg'))) # all cams have same num frames
            for j in range(num_frames):
                indices.append([demo_name, j])

        self.indices = indices

    def __len__(self):
        return len(self.indices) - 1

    def __getitem__(self, idx):
        """

        """
        # randomly sample 2 cameras
        cam1, cam2 = random.choices(self.cam_keys, k=2)

        demo1, demo_idx1 = self.indices[idx]
        demo2, demo_idx2 = self.indices[idx+1]

        # img1 is image at given idx for cam1
        # img2 is image at next time step (after action applied) for cam2
        img1_path = os.path.join(self.data_root, demo1, 'obs', f'{cam1}_image', f'{demo_idx1}.jpg')
        img2_path = os.path.join(self.data_root, demo2, 'obs', f'{cam2}_image', f'{demo_idx2}.jpg')
        img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)
        img1 = torch.tensor(img1).permute(2, 0, 1).float() / 255.0
        img2 = torch.tensor(img2).permute(2, 0, 1).float() / 255.0
        imgs = torch.stack([img1, img2])
        
        # corresponding action
        action = torch.tensor((self.actions[demo2])[demo_idx2])

        # camera projection matrices
        proj1 = torch.tensor(self.projs[cam1])
        proj2 = torch.tensor(self.projs[cam2])

        return imgs, action, proj1, proj2


# class Datasethdf5Multiview(Dataset):
#     def __init__(self, 
#                  dataset_root='./data/', 
#                  cam_file='./assets/camera_pose_dict.yaml',
#                  task='multicamera_separate_demo_v141',
#                  img_size=128,
#                  valhold=6):

#         # load camera data
#         with open(cam_file, 'r') as f:
#             cam_data = yaml.load(f, Loader=yaml.FullLoader)

#         intrinsic = INTRINSICS[img_size]
#         self.projs = get_projs(cam_data, intrinsic)

#         self.cam_keys = list(cam_data.keys())

#         # load sequence data
#         dataset_path = os.path.join(dataset_root, f'{task}.hdf5')
#         if not os.path.exists(dataset_path):
#             raise FileNotFoundError(f"ERROR: {dataset_path} does not exist")
        
#         self.hdf5_file = h5py.File(dataset_path, 'r')
#         self.demos = self.hdf5_file['data']

#         indices = []
#         # count total sequence length and associate sequence index with demo
#         sequence_len = 0
#         for i in range(len(self.demos)):
#             demo_name = f'demo_{i}'
#             demo = self.demos[demo_name]
#             demo_len = demo['actions'].shape[0]
#             sequence_len += demo_len
#             for j in range(demo_len):
#                 indices.append([demo_name, j])
#         self.indices = indices

#     def __len__(self):
#         return len(self.indices) - 1

#     def __getitem__(self, idx):
#         # randomly sample 2 cameras
#         cam1, cam2 = random.choices(self.cam_keys, k=2)

#         demo1, demo_idx1 = self.indices[idx]
#         demo2, demo_idx2 = self.indices[idx+1]

#         # img1 is image at given idx for cam1
#         # img2 is image at next time step (after action applied) for cam2
#         img1 = np.array(self.demos[demo1]['obs'][f'{cam1}_image'])[demo_idx1].transpose(2, 0, 1) / 255.
#         img2 = np.array(self.demos[demo2]['obs'][f'{cam2}_image'])[demo_idx2].transpose(2, 0, 1) / 255.
#         imgs = np.array([img1, img2])
        
#         action = np.array(self.demos[demo2]['actions'])[demo_idx2]

#         proj1 = self.projs[cam1]
#         proj2 = self.projs[cam2]

#         return imgs, action, proj1, proj2

#     def __del__(self):
#         # ensure hdf5 file closed when garbage collecting
#         if hasattr(self, 'hdf5_file') and self.hdf5_file is not None:
#             self.hdf5_file.close()