import os
import numpy as np

import torch
from scipy.spatial.transform import Rotation as R

def invert_trajectory_actions(actions):
    L, action_dim = actions.shape
    assert action_dim == 7, "action should be (x, y, z, roll, pitch, yaw, gripper)"

    inverted_actions = torch.zeros_like(actions)

    for i in range(L):
        # Extract translation and orientation
        x, y, z, roll, pitch, yaw, _ = actions[i]
        gripper_inv = actions[i-1][6] if i >1 else -1
        
        # Convert to homogeneous transformation matrix
        R = euler_to_rotation_matrix(roll, pitch, yaw)
        T = torch.eye(4)
        T[:3, :3] = R
        T[:3, 3] = torch.tensor([x, y, z])
        
        # Invert the transformation
        T_inv = torch.eye(4)
        T_inv[:3, :3] = T[:3, :3].T  # Transpose rotation
        T_inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]  # Inverse translation
        
        # Convert back to (x, y, z, roll, pitch, yaw)
        R_inv = T_inv[:3, :3]
        x_inv, y_inv, z_inv = T_inv[:3, 3]
        roll_inv, pitch_inv, yaw_inv = rotation_matrix_to_euler(R_inv)
        
        # Store inverted action
        inverted_actions[i] = torch.tensor([x_inv, y_inv, z_inv, roll_inv, pitch_inv, yaw_inv, gripper_inv])
    
    # Reverse the order of actions
    inverted_actions = inverted_actions.flip(0)
    
    return inverted_actions

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Converts Euler angles (roll, pitch, yaw) to a 3x3 rotation matrix.
    """
    R_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(roll), -torch.sin(roll)],
        [0, torch.sin(roll), torch.cos(roll)]
    ])
    R_y = torch.tensor([
        [torch.cos(pitch), 0, torch.sin(pitch)],
        [0, 1, 0],
        [-torch.sin(pitch), 0, torch.cos(pitch)]
    ])
    R_z = torch.tensor([
        [torch.cos(yaw), -torch.sin(yaw), 0],
        [torch.sin(yaw), torch.cos(yaw), 0],
        [0, 0, 1]
    ])
    return R_z @ R_y @ R_x

def rotation_matrix_to_euler(R):
    """
    Converts a 3x3 rotation matrix to Euler angles (roll, pitch, yaw).
    """
    sy = torch.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        roll = torch.atan2(R[2, 1], R[2, 2])
        pitch = torch.atan2(-R[2, 0], sy)
        yaw = torch.atan2(R[1, 0], R[0, 0])
    else:
        roll = torch.atan2(-R[1, 2], R[1, 1])
        pitch = torch.atan2(-R[2, 0], sy)
        yaw = 0
    return roll, pitch, yaw

def adding_invert_actions_to_dataset(dataset_folder_path):
    demo_list: list[str] = os.listdir(dataset_folder_path)
    for folder in demo_list:
        if folder.startswith("demo"):
            print(f"processing {folder}")
            demo_path = os.path.join(dataset_folder_path, folder)
            actions = torch.from_numpy(np.load(os.path.join(demo_path, "actions.npy")))
            invert_actions = invert_trajectory_actions(actions)
            np.save(os.path.join(demo_path, "invert_actions.npy"), invert_actions.numpy())

if __name__ == '__main__':
    adding_invert_actions_to_dataset("/home/mingxi/mingxi_ws/ssl/sqaure_0")