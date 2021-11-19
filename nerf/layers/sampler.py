import torch

__all__ = ["build_sampler", ]


def sample_pdf(upper, lower, weights, num_samples_fine):
    """
    Sample points based on pdf generated by the rendering weights.
    
    Args:
        upper (tensor): The right side of each interval.
        lower (tensor): The left side of each interval.
        weights (tensor): Rendering weights of the interval sized 
            [batch, num_samples].
        num_samples_fine (int): Number of sampling points.
    """
    # Get pdf and cdf
    weights += 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)

    # Take uniform samples
    u = torch.rand([*cdf.shape[:-1], num_samples_fine])

    # Inverse transform sampling
    inds = torch.searchsorted(cdf, u)
    
    # Gather
    pdf_g = torch.gather(pdf, -1, inds)
    cdf_g = torch.gather(cdf, -1, inds)
    lower_g = torch.gather(lower, -1, inds)
    upper_g = torch.gather(upper, -1, inds)

    # Get sample points
    t = 1. - (cdf_g - u) / pdf_g
    samples = lower_g + (upper_g - lower_g) * t
    
    return samples
    

def point_sampling(rays, num_samples):
    # TODO: a little different from origin implementation, check it
    """
    Sample points along the rays.
    
    We first split the ray into N intervals and then sample a point in each 
    interval uniformly in coarse stage. While in fine stage, we sample points
    based on pdf generated by the rendering weights.

    Args:
        rays (tensor): Batch of rays sized [batch, 8 or 11]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
        num_samples (int): Number of sampling points along each ray.
    """
    # Extract ray origin, direction.
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
    # Extract lower, upper bound for ray distance.
    # bounds = rays[..., 6:8].view([-1, 1, 2])
    # near, far = bounds[..., 0], bounds[..., 1]
    near, far = rays[..., 6], rays[..., 7]
    
    # sample points
    # split the ray into N intervals
    breakpoints = torch.linspace(0., 1., num_samples + 1)
    breakpoints = near[:, None] * (1. - breakpoints[None, :]) + \
                  far[:, None] * breakpoints[None, :]
    upper, lower = breakpoints[..., 1:], breakpoints[..., :-1]
    # sampling in each interval
    rand = torch.rand(upper.shape)
    t_ray = lower + (upper - lower) * rand

    points = rays_o[..., None, :] + rays_d[..., None, :] * \
        t_ray[..., :, None]

    def point_sampling_fine(weights, num_samples_fine):
        """
        Args:
            weights (tensor): Weights assigned to each sampled point sized
                [batch, num_samples].
            num_samples_fine (int): Number of sampling points along each ray 
                in fine stage.
        """
        # sample points based on weights
        t_ray_fine = sample_pdf(upper, lower, weights, num_samples_fine)
        # add the newly samples to the original sample collection
        t_ray_fine = torch.sort(torch.cat([t_ray, t_ray_fine], dim=-1), dim=-1)
        points_fine = rays_o[..., None, :] + rays_d[..., None, :] * \
            t_ray_fine[..., :, None]
        return points_fine

    return points, point_sampling_fine
