import torch

__all__ = ["renderRays", ]


def renderRays(sigma, delta, color, output_weight=False):
    """
    Args:
        sigma (tensor): volume density with the shape (num_rays, num_points).
        delta (tensor): distance of each segment with the shape (num_rays, num_points).
        color (tensor): predicted RGB value of each sampling points with the shape 
            (num_rays, num_points, 3).
        output_weight (bool): in 'fine' stage, we need to caculate the weights of each 
            samping points so that we can resample the points. This variable determines
            weather we need to return the weights.
    """
    assert sigma.shape == delta.shape
    alpha = 1 - torch.exp(-sigma * delta)
    weight = alpha * torch.cumprod(1 - alpha * 1e-10, dim=-1)

    rgb = torch.sum(weight[..., None] * color, dim=-2)
    
    if output_weight:
        return rgb, weight
    else:
        return rgb, None


def render():
    pass
