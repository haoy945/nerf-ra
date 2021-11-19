import torch
import numpy as np

from nerf.utils import get_rays
from .data_sampler import TrainingSampler


class DataLoaderTrain:
    def __init__(self, images, poses, hwf, near, far, use_pixel_batching, ues_viewdirs, batch_size, sampler=None) -> None:
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
            sampler (Sampler or Iterable, optional): Defines the strategy to draw
                samples from the dataset.
        """

        self.rays_rgb = self.get_rays_rgb(images, poses, hwf, use_pixel_batching)
        self.bs = batch_size
        self.near = near
        self.far = far
        self.use_pixel_batching = use_pixel_batching
        self.ues_viewdirs = ues_viewdirs

        self.len_ = self.rays_rgb.shape[0]

        if sampler == None:
            sampler = TrainingSampler(self.len_)
        self.sampler = iter(sampler)

    def get_rays_rgb(self, images, poses, hwf, use_pixel_batching):
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
            batch = self.rays_rgb[next(self.sampler)]
            select_inds = torch.tensor(np.random.choice(
                batch.shape[0], size=self.bs, replace=False))
            # [bs, ro+rb+rgb, 3]
            batch = batch[select_inds]

        rays_o, rays_d, targets = batch[:, 0], batch[:, 1], batch[:, 2]
        rays = self.creat_ray_batch(rays_o, rays_d)
        return {
            "batch_rays": rays,
            "batch_targets": targets,
        }

    def creat_ray_batch(self, rays_o, rays_d):
        near = self.near * torch.ones_like(rays_d[:, :1])
        far = self.far * torch.ones_like(rays_d[:, :1])

        rays = torch.cat([rays_o, rays_d, near, far], dim=-1)

        if self.ues_viewdirs:
            viewdirs = rays_d
            viewdirs = viewdirs / torch.linalg.norm(viewdirs, axis=-1, keepdims=True)
            rays = torch.cat([rays, viewdirs], dim=-1)
        
        return rays
