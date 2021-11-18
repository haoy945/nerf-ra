import torch
import numpy as np

__all__ = ["get_rays", ]


def get_rays(H, W, focal, c2w):
    """
    Get ray origins, directions from a pinhole camera.
    """

    i, j = torch.meshgrid(torch.range(W, dtype=torch.float32),
                          torch.range(H, dtype=torch.float32), indexing='xy')
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)

    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)

    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d
