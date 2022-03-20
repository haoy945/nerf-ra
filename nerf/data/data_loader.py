import math
import torch
import numpy as np

from ..utils import get_rays
from .data_sampler import TrainingSampler, InferenceSampler

__all__ = ["DataLoaderTrain", "DataLoaderTest", ]


def get_rays_rgb(images, poses, hwf, use_pixel_batching):
    H, W, focal = hwf
    # [N, ro+rb, H, W, 3]
    rays = torch.stack(
        [get_rays(H, W, focal, p) for p in poses[:, :3, :4]], dim=0
    )
    # [N, ro+rb+rgb, H, W, 3]
    rays_rgb = torch.cat([rays, images[:, None]], dim=1)
    # [N, H, W, ro+rb+rgb, 3]
    rays_rgb = rays_rgb.permute(0, 2, 3, 1, 4)

    if use_pixel_batching:
        # [N*H*W, ro+rb+rgb, 3]
        rays_rgb = rays_rgb.reshape(-1, 3, 3)
    else:
        # [N, H*W, ro+rb+rgb, 3]
        rays_rgb = rays_rgb.reshape(-1, H*W, 3, 3)
    return rays_rgb


def creat_ray_batch(rays_o, rays_d, near, far, ues_viewdirs):
    near = near * torch.ones_like(rays_d[:, :1])
    far = far * torch.ones_like(rays_d[:, :1])

    rays = torch.cat([rays_o, rays_d, near, far], dim=-1)

    if ues_viewdirs:
        viewdirs = rays_d
        viewdirs = viewdirs / torch.linalg.norm(viewdirs, axis=-1, keepdims=True)
        rays = torch.cat([rays, viewdirs], dim=-1)
    
    return rays


class DataLoaderTrain:
    def __init__(self, images, poses, hwf, near, far, use_pixel_batching, ues_viewdirs, 
                 batch_size, precrop_iters, precrop_frac, sampler=None) -> None:
        """
        Args:
            images (tensor): Images sized [N, H, W, 3].
            poses (tensor): Camera poses sized [N, 4, 4].
            hwf (list): Height, width and focal respectively.
            near (float): 
            far (float):
            ues_viewdirs (bool): If True, use viewing direction of a point in space in model.
            use_pixel_batching (bool): Whether to sample pixels over all images.
            batch_size (int): How many samples per batch to load.
            precrop_iters (int): number of steps to train on central crops.
            precrop_frac (float): fraction of img taken for central crops.
            sampler (Sampler or Iterable, optional): Defines the strategy to draw
                samples from the dataset.
        """

        self.rays_rgb = get_rays_rgb(images, poses, hwf, use_pixel_batching)
        self.bs = batch_size
        self.near = near
        self.far = far
        self.use_pixel_batching = use_pixel_batching
        self.ues_viewdirs = ues_viewdirs
        self.precrop_iters = precrop_iters
        self.precrop_frac = precrop_frac

        self.len_ = self.rays_rgb.shape[0]
        self.iter_ = 0

        if sampler == None:
            sampler = TrainingSampler(self.len_)
        self.sampler = iter(sampler)

    def __len__(self):
        return self.len_
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.use_pixel_batching:
            inds = [next(self.sampler) for _ in range(self.bs)]
            # [bs, ro+rb+rgb, 3]
            batch = self.rays_rgb[inds]
        else:
            # [H*W, ro+rb+rgb, 3]
            batch = self.rays_rgb[next(self.sampler)]

            if self.iter_ < self.precrop_iters:
                '''Center Cropping'''
                H = W = int(math.sqrt(batch.shape[0]))
                dH = int(H // 2 * self.precrop_frac)
                dW = int(W // 2 * self.precrop_frac)
                coords_x, coords_y = torch.meshgrid(
                        torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                        torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW), 
                indexing='xy')
                # 2d coords -> 1d coords
                inds = torch.flatten(coords_y * W + coords_x, start_dim=0)
                select_inds = inds[torch.tensor(np.random.choice(
                    inds.shape[0], size=self.bs, replace=False))].long()
                self.iter_ += 1
            else:
                select_inds = torch.tensor(np.random.choice(
                    batch.shape[0], size=self.bs, replace=False))
            # [bs, ro+rb+rgb, 3]
            batch = batch[select_inds]

        rays_o, rays_d, targets = batch[:, 0], batch[:, 1], batch[:, 2]
        rays = creat_ray_batch(rays_o, rays_d, self.near, self.far, self.ues_viewdirs)
        return {
            "batched_rays": rays,  # [bs, 8 or 11]
            "batched_targets": targets,  # [bs, 3]
        }


class DataLoaderTest:
    def __init__(self, images, poses, hwf, near, far, ues_viewdirs, 
                 sampler=None) -> None:
        """
        Args:
            images (tensor): Images sized [N, H, W, 3].
            poses (tensor): Camera poses sized [N, 4, 4].
            hwf (list): Height, width and focal respectively.
            near (float): 
            far (float):
            ues_viewdirs (bool): If True, use viewing direction of a point in space in model.
            sampler (Sampler or Iterable, optional): Defines the strategy to draw
                samples from the dataset.
        """

        self.rays_rgb = get_rays_rgb(images, poses, hwf, use_pixel_batching=False)
        self.near = near
        self.far = far
        self.ues_viewdirs = ues_viewdirs

        self.len_ = self.rays_rgb.shape[0]

        if sampler == None:
            sampler = InferenceSampler(self.len_)
        self.sampler = iter(sampler)

    def __len__(self):
        return self.len_

    def __iter__(self):
        return self

    def __next__(self):
        ind = next(self.sampler)
        batch = self.rays_rgb[ind]

        rays_o, rays_d, targets = batch[:, 0], batch[:, 1], batch[:, 2]
        rays = creat_ray_batch(rays_o, rays_d, self.near, self.far, self.ues_viewdirs)
        return {
            "batched_rays": rays,  # [bs, 8 or 11]
            "batched_targets": targets,  # [bs, 3]
        }
