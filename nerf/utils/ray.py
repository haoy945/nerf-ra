import torch

__all__ = ["get_rays", ]


def get_rays(H, W, focal, c2w):
    """
    Get ray origins, directions from a pinhole camera.
    """

    i, j = torch.meshgrid(torch.linspace(0, W-1, W),
                          torch.linspace(0, H-1, H), indexing='xy')
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)

    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)

    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    
    return torch.stack([rays_o, rays_d], dim=0)
