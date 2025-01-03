import os
import torch
import random
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision.io import read_image
from utils.plucker_ray import plucker_embedding
from scipy.spatial.transform import Rotation as R

random.seed(0)

INTRINSICS = {
    128: torch.tensor(
        [[154.50966799, 0.0, 64.0], [0.0, 154.50966799, 64.0], [0.0, 0.0, 1.0]]
    ),
    224: torch.tensor(
        [
            [270.39191899, 0.0, 112.0, 0.0],
            [0.0, 270.39191899, 112.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
}


def get_proj(cam_extrs, intr):
    """
    Calculate camera projection matrices.

    Params
    ------
    cam_extrs: dict {cam_name -> [x, y, z, qw, qx, qy, qz]}
    intr: tensor [3, 3]

    Return
    ------
    projs: dict {camera name: projection matrix}
    """
    c2w = {}

    K = torch.eye(4)
    K[:3, :3] = intr[:3, :3]
    K[3, 3] = 1

    for cam_name, params in cam_extrs.items():
        x, y, z, qw, qx, qy, qz = params
        rot = torch.tensor(R.from_quat([qx, qy, qz, qw]).as_matrix())
        t = torch.tensor([x, y, z])
        w2c = torch.eye(4)
        w2c[:3, :3] = rot
        w2c[:3, 3] = t
        c2w[cam_name] = torch.inverse(w2c)
        # projs[cam_name] = K @ w2c

    return c2w


def get_pixel_coords(img_size):
    """
    Generate pixel coordinates for all pixels in an image.

    Params
    ------
    img_size: int

    Return
    ------
    uv: tensor [N, 2]
    """
    h, w = img_size, img_size
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    uv = torch.stack([x.flatten(), y.flatten(), torch.ones_like(x.flatten())], dim=-1)
    return uv


def get_plucker(cam_extrs, intr, img_size=224):
    """
    Calculate Plucker embedding.

    Params
    ------
    cam_extrs: dict {cam_name -> [x, y, z, qw, qx, qy, qz]}
    intr: tensor [3, 3]
    img_size: int

    Return
    ------
    plucker: dict {camera name: Plucker embedding}
    """
    plucker = {}

    K = torch.eye(4)
    K[:3, :3] = intr
    K[3, 3] = 1
    uv = get_pixel_coords(img_size)

    for cam_name, params in cam_extrs.items():
        x, y, z, qw, qx, qy, qz = params
        rot = torch.tensor(R.from_quat([qx, qy, qz, qw]).as_matrix())
        t = torch.tensor([x, y, z])
        w2c = torch.eye(4)
        w2c[:3, :3] = rot
        w2c[:3, 3] = t
        c2w = torch.inverse(w2c)

        plucker[cam_name] = plucker_embedding(c2w, uv, K)

    return plucker


# https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/common/sampler.py
def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


class DatasetMultiview(Dataset):
    def __init__(
        self,
        dataset_root="./data/",
        cam_file="camera_pose_dict.npy",
        task="square_d0",
        mode="train",
        cam_mode="proj",
        img_size=224,
        seed=42,
        val_ratio=0.01,
        device="cuda",
        enable_multi_view=True,
    ):

        self.mode = mode
        assert self.mode in ["train", "val"], "ERROR: mode has to be train or val"
        assert cam_mode in [
            "proj",
            "plucker",
        ], "ERROR: camera mode has to be proj or plucker"
        self.data_root = os.path.join(dataset_root, task)
        self.device = device

        # load camera data
        cam_data = np.load(
            os.path.join(self.data_root, cam_file), allow_pickle=True
        ).item()
        self.cam_keys = list(cam_data.keys())
        self.intrinsic = INTRINSICS[img_size]
        self.cams = get_proj(cam_data, self.intrinsic)

        demo_dirs = glob(os.path.join(self.data_root, "demo_*"))
        # set certain demos to train and certain to val
        val_mask = get_val_mask(
            n_episodes=len(demo_dirs), val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        mask = train_mask if mode == "train" else val_mask
        
        self.demo_dirs = [demo_dir for i, demo_dir in enumerate(demo_dirs) if mask[i]]
        self.num_demos = len(self.demo_dirs)
        
        self.actions = {}
        indices = []
        for i, demo_dir in enumerate(demo_dirs):
            if not mask[i]:
                continue

            demo_name = os.path.basename(demo_dir)
            # store the associated action array for the demo
            self.actions[demo_name] = np.load(os.path.join(demo_dir, "actions.npy"))
            # get all of the frames
            num_frames = len(
                glob(os.path.join(demo_dir, "obs", "camera0_image", "*.jpg"))
            )  # all cams have same num frames
            for j in range(num_frames):
                indices.append([demo_name, j])

        self.indices = indices
        self.enable_multi_view = enable_multi_view

    def __len__(self):
        return len(self.indices) - 1

    def __getitem__(self, idx):
        """
        Return
        ------
        imgs: [2, c, h, w]
        action: [x, y, z, roll, pitch, yaw, gripper]
        cam1: proj [4, 4] or plucker [N, 6]
        cam2: proj [4, 4] or plucker [N, 6]
        """
        # randomly sample 2 cameras
        cam1, cam2 = random.choices(self.cam_keys, k=2)
        if not self.enable_multi_view:
            cam2 = cam1

        demo1, demo_idx1 = self.indices[idx]
        demo2, demo_idx2 = self.indices[idx + 1]

        # img1 is image at given idx for cam1
        # img2 is image at next time step (after action applied) for cam2
        img1_path = os.path.join(
            self.data_root, demo1, "obs", f"{cam1}_image", f"{demo_idx1}.jpg"
        )
        img2_path = os.path.join(
            self.data_root, demo2, "obs", f"{cam2}_image", f"{demo_idx2}.jpg"
        )
        img1 = read_image(img1_path).float() / 255.0
        img2 = read_image(img2_path).float() / 255.0

        # corresponding action
        action = torch.tensor((self.actions[demo1])[demo_idx1]).float()

        # cameras
        cam1 = self.cams[cam1].float()
        cam2 = self.cams[cam2].float()

        return img1, img2, action, cam1, cam2

    def get_seq(self, demo_idx, cam_idx):
        cam = self.cam_keys[cam_idx]
        imgs_path = os.path.join(
            self.demo_dirs[demo_idx], "obs", f"{cam}_image"
        )

        imgs = glob(os.path.join(imgs_path, "*.jpg"))
        imgs = sorted(imgs, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        
        # corresponding action
        action = np.load(os.path.join(self.demo_dirs[demo_idx], "actions.npy"))
        
        return torch.stack([read_image(img).float()/255. for img in imgs]), action, self.cams[cam].float(), cam, os.path.basename(self.demo_dirs[demo_idx])
    
    def get_frames(self, demo_idx, frame_idx):

        imgs = [os.path.join(self.demo_dirs[demo_idx], "obs", f"{cam}_image", f"{frame_idx}.jpg") for cam in self.cam_keys]

        action = np.load(os.path.join(self.demo_dirs[demo_idx], "actions.npy"))[frame_idx]
        
        cam = np.stack([self.cams[cam].float() for cam in self.cam_keys])
        
        return torch.stack([read_image(img).float()/255. for img in imgs]), action, cam, self.cam_keys, os.path.basename(self.demo_dirs[demo_idx])

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
