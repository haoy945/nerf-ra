import torch

__all__ = ["build_render", ]


def _renderRays(sigma, dists, rgb, output_weight=False):
    """
    Args:
        sigma (tensor): volume density with the shape (num_rays, num_points).
        dists (tensor): distance of each segment with the shape (num_rays, num_points).
        rgb (tensor): predicted RGB value of each sampling points with the shape 
            (num_rays, num_points, 3).
        output_weight (bool): in 'fine' stage, we need to caculate the weights of each 
            samping points so that we can resample the points. This variable determines
            weather we need to return the weights.
    """
    assert sigma.shape == dists.shape
    alpha = 1. - torch.exp(-sigma * dists)
    weight = alpha * torch.cumprod(
        1. - torch.cat([torch.zeros_like(alpha[:, :1]), alpha[:, :-1]], dim=-1) + 1e-10, 
        dim=-1
    )

    rgb = torch.sum(weight[..., None] * rgb, dim=-2)
    
    if output_weight:
        return rgb, weight
    else:
        return rgb, None


def renderRays(raw, pts, output_weight=False, raw_noise_std=0.):
    """
    Args:
        raw (tensor): Batch of raw outputs of MLP sized (num_rays, num_points, 4),
            including: rgb, sigma.
        pts (tensor): 3D coordinate of points along each ray, 
            sized (num_rays, num_points, 3)
        output_weight (bool): in 'fine' stage, we need to caculate the weights of each 
            samping points so that we can resample the points. This variable determines
            weather we need to return the weights.
    """
    # Compute 'distance' between each integration time along a ray.
    dists = pts[:, 1:] - pts[:, :-1]
    dists = dists.norm(dim=-1)
    # The 'distance' from the last integration time is infinity.
    dists = torch.cat([dists, torch.tensor([1e10]).expand(dists[:, :1].shape)], dim=-1)

    # Extract RGB and Sigma of each sample position along each ray.
    rgb, sigma = raw[..., :3], raw[..., 3]
    rgb = torch.sigmoid(rgb)
    noise = torch.rand_like(sigma) * raw_noise_std
    sigma = torch.nn.functional.relu(sigma + noise)

    return _renderRays(sigma, dists, rgb, output_weight)


def build_render(cfg):
    return renderRays
