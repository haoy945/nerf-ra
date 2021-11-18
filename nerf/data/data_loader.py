import torch
import numpy as np

from nerf.utils import get_rays


class DataLoaderTrain:
    def __init__(self, images, poses, hwf, near, far, use_batching, batch_size) -> None:
        self.rays_rgb = self.get_rays_rgb(images, poses, hwf, use_batching)
        self.bs = batch_size
        self.near = near
        self.far = far
        self.use_batching = use_batching

        self.len_ = self.rays_rgb.shape[0]
        # index used for sampling
        self.inds = torch.randperm(self.rays_rgb.shape[0])

    def get_rays_rgb(self, images, poses, hwf, use_batching):
        H, W, focal = hwf
        # [N, ro+rb, H, W, 3]
        rays = torch.stack(
            [get_rays(H, W, focal, p) for p in poses[:, :3, :4]], dim=0
        )
        # [N, ro+rb+rgb, H, W, 3]
        rays_rgb = torch.cat([rays, images[:, None]], dim=1)
        rays_rgb = rays_rgb.permute(0, 2, 3, 1, 4)

        if use_batching:
            # [N*H*W, ro+rb+rgb, 3]
            rays_rgb = rays_rgb.view(-1, 3, 3)
        else:
            # [N, H*W, ro+rb+rgb, 3]
            rays_rgb = rays_rgb.view(-1, H*W, 3, 3)
        return rays_rgb

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        if self.use_batching:
            # TODO: use TrainingSampler
            start, end = idx * self.bs, (idx+1) * self.bs
            batch = self.rays_rgb[self.inds[start:end]]
            if end >= len(self):
                self.inds = torch.randperm(len(self))
        else:
            # TODO: use TrainingSampler
            batch = self.rays_rgb[idx]
            select_inds = torch.tensor(np.random.choice(
                len(self), size=self.bs, replace=False))
            batch = batch[select_inds]
        
        batch_rays, batch_targets = batch[:2], batch[2]
        return {
            "batch_rays": batch_rays,
            "batch_targets": batch_targets,
            "near": self.near,
            "far": self.fat,
        }
