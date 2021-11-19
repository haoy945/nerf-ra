import torch.nn as nn
import torch.nn.functional as F

from ..network import build_nerf_mlp
from ...layers import (
    build_embedder,
    build_render,
    build_sampler,
)

__all__ = ["NeRF", "build_meta_arch", ]


class NeRF(nn.Module):
    """
    There are about 4 steps to render an image.
    First, we should sample the points and each point is represented by (x,y,z,theta,phi).
    Then we map the vector to a higher dimensional space using high frequency functions.
    Thirdly, we have neural network to operate on the coodinates to get the color and 
    density for every point.
    Last, we use volume rendering to composite these values into an image.

    At each optimization iteration, we randomly samplea batch of camera rays from the 
    set of all pixels in the dataset. Our loss is simply the total squared error 
    betweenthe rendered and true pixel colors.
    """

    def __init__(self, nerf_mlp, embedder, points_sampler, render, 
                 num_samples, num_samples_fine, nerf_mlp_fine=None, raw_noise_std=0.):
        super().__init__()
        self.nerf_mlp = nerf_mlp
        self.nerf_mlp_fine = nerf_mlp_fine
        self.embedder = embedder
        self.points_sampler = points_sampler
        self.render = render

        self.num_samples = num_samples
        self.num_samples_fine = num_samples_fine
        self.raw_noise_std = raw_noise_std

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        batched_rays = batched_inputs["batched_rays"]
        batched_targets = batched_inputs["batched_targets"]
        use_viewdirs = batched_rays.shape[-1] > 8
        losses = {}

        # coarse stage
        # sampling
        points, point_sampling_fine = self.points_sampler(batched_rays, self.num_samples)

        losses_coarse, weights = self._forward(
            points, batched_rays, batched_targets, use_viewdirs, coarse_stage=True)
        losses.update(losses_coarse)

        # fine stage 
        if self.nerf_mlp_fine:
            points_fine = point_sampling_fine(weights, self.num_samples_fine)

            losses_fine, _ = self._forward(
                points_fine, batched_rays, batched_targets, use_viewdirs, coarse_stage=False)
            losses.update(losses_fine)

        return losses

    def _forward(self, pts, batched_rays, batched_targets, use_viewdirs, coarse_stage=True):
        # 
        if use_viewdirs:
            viewdirs = batched_rays[:, 8:]
            viewdirs = viewdirs[:, None].expand(pts.shape)
        else:
            viewdirs = None

        if coarse_stage:
            mlp = self.nerf_mlp
            loss_name = "loss_coarse"
        else:
            mlp = self.nerf_mlp_fine
            loss_name = "loss_fine"

        # embedding
        pos_embed, dir_embed = self.embedder(pts, viewdirs)
        # running network
        outputs = mlp(pos_embed, dir_embed)
        # rendering
        rgb, weights = self.render(outputs, pts, output_weight=coarse_stage, 
            raw_noise_std=self.raw_noise_std)
        # caculate loss
        losses = {loss_name: self.loss(rgb, batched_targets)}

        return losses, weights

    def inference(self):
        assert not self.training

    def loss(self, preds, targets):
        return F.mse_loss(preds, targets, reduction='mean')


def build_meta_arch(cfg):
    nerf_mlp = build_nerf_mlp(cfg)
    embedder = build_embedder(cfg)
    points_sampler = build_sampler(cfg)
    render = build_render(cfg)

    num_samples = cfg.MODEL.NUM_SAMPLES
    num_samples_fine = cfg.MODEL.NUM_SAMPLES_FINE
    raw_noise_std = cfg.MODEL.RENDER.RAW_NOISE_STD

    nerf_mlp_fine = build_nerf_mlp(cfg) if num_samples_fine > 0 else None

    model = NeRF(nerf_mlp, embedder, points_sampler, render, num_samples, 
                 num_samples_fine, nerf_mlp_fine, raw_noise_std)
    return model
