"""For general notes on Plucker coordinates:
https://faculty.sites.iastate.edu/jia/files/inline-files/plucker-coordinates.pdf"""

import torch
from torch.nn import functional as F


def get_ray_origin(cam2world):
    return cam2world[..., :3, 3]


def parse_intrinsics(intrinsics):
    fx = intrinsics[..., 0, :1]
    fy = intrinsics[..., 1, 1:2]
    cx = intrinsics[..., 0, 2:3]
    cy = intrinsics[..., 1, 2:3]
    return fx, fy, cx, cy


def expand_as(x, y):
    if len(x.shape) == len(y.shape):
        return x
    for i in range(len(y.shape) - len(x.shape)):
        x = x.unsqueeze(-1)
    return x


def lift(x, y, z, intrinsics, homogeneous=False):
    """
    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    """
    fx, fy, cx, cy = parse_intrinsics(intrinsics)
    x_lift = (x - expand_as(cx, x)) / expand_as(fx, x) * z
    y_lift = (y - expand_as(cy, y)) / expand_as(fy, y) * z
    if homogeneous:
        return torch.stack((x_lift, y_lift, z, torch.ones_like(z).to(x.device)), dim=-1)
    else:
        return torch.stack((x_lift, y_lift, z), dim=-1)


def world_from_xy_depth(xy, depth, cam2world, intrinsics):
    batch_size, *_ = cam2world.shape
    x_cam = xy[..., 0]
    y_cam = xy[..., 1]
    z_cam = depth
    pixel_points_cam = lift(
        x_cam, y_cam, z_cam, intrinsics=intrinsics, homogeneous=True
    )
    world_coords = torch.einsum("b...ij,b...kj->b...ki", cam2world, pixel_points_cam)[
        ..., :3
    ]
    return world_coords


def get_ray_directions(xy, cam2world, intrinsics):
    z_cam = torch.ones(xy.shape[:-1]).to(xy.device)
    pixel_points = world_from_xy_depth(
        xy, z_cam, intrinsics=intrinsics, cam2world=cam2world
    )  # (batch, num_samples, 3)
    cam_pos = cam2world[..., :3, 3]
    ray_dirs = pixel_points - cam_pos[..., None, :]  # (batch, num_samples, 3)
    ray_dirs = F.normalize(ray_dirs, dim=-1)
    return ray_dirs


def plucker_embedding(cam2world, uv, intrinsics):
    """Computes the plucker coordinates from batched cam2world & intrinsics matrices, as well as pixel coordinates
    cam2world: (b, 4, 4)
    intrinsics: (b, 4, 4)
    uv: (b, n, 2)"""
    ray_dirs = get_ray_directions(uv, cam2world=cam2world, intrinsics=intrinsics)
    cam_pos = get_ray_origin(cam2world)
    cam_pos = cam_pos[..., None, :].expand(list(uv.shape[:-1]) + [3])
    # https://www.euclideanspace.com/maths/geometry/elements/line/plucker/index.htm
    # https://web.cs.iastate.edu/~cs577/handouts/plucker-coordinates.pdf
    cross = torch.cross(cam_pos, ray_dirs, dim=-1)
    plucker = torch.cat((ray_dirs, cross), dim=-1)
    return plucker


if __name__ == "__main__":
    # Dummy input parameters
    b = 2  # Batch size
    # Camera-to-world transformation matrices (identity for simplicity)
    cam2world = torch.eye(4).unsqueeze(0).repeat(b, 1, 1)
    # Intrinsic camera matrices (simple focal lengths and principal points)
    focal_length = 800.0
    principal_point = 512.0
    intrinsics = (
        torch.tensor(
            [
                [focal_length, 0, principal_point, 0],
                [0, focal_length, principal_point, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .unsqueeze(0)
        .repeat(b, 1, 1)
    )
    # Pixel coordinates (uv) in image space
    uv = torch.tensor(
        [[[100, 200]], [[150, 250]]], dtype=torch.float32  # Batch 1  # Batch 2
    ).repeat(1, 224 * 224, 1)
    breakpoint()
    # Test the plucker_embedding function
    plucker = plucker_embedding(cam2world, uv, intrinsics)
    print("Plucker coordinates shape:", plucker.shape)
    print("Plucker coordinates:", plucker)
