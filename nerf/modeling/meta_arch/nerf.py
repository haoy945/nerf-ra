import torch.nn as nn
import torch.nn.functional as F

from nerf.modeling.network import nerf_mlp

from ..network import NerfMlp
from ...layers import point_sampling


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
                 num_samples, num_samples_fine, *args, nerf_mlp_fine=None):
        super().__init__()
        self.nerf_mlp = nerf_mlp
        self.nerf_mlp_fine = nerf_mlp_fine
        self.embedder = embedder
        self.points_sampler = points_sampler
        self.render = render

        self.num_samples = num_samples
        self.num_samples_fine = num_samples_fine

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        batched_rays = batched_inputs["batched_rays"]
        batched_targets = batched_inputs["batched_targets"]
        losses = {}

        # coarse stage
        # sampling
        points, point_sampling_fine = self.points_sampler(batched_rays, self.num_samples)
        use_viewdirs = batched_rays.shape[-1] > 8

        losses_coarse, weights = self._forward(
            points, batched_rays, batched_targets, use_viewdirs, coarse_stage=True)
        losses.update(losses_coarse)

        # fine stage 
        if not self.nerf_mlp_fine:
            points_fine = point_sampling_fine(weights, self.num_samples_fine)

            losses_fine = self._forward(
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

        # embedding
        pos_embed, dir_embed = self.embedder(pts, viewdirs)
        # running network
        outputs = self.nerf_mlp(pos_embed, dir_embed)
        # rendering
        rgb, weights = self.render(outputs, pts, output_weight=coarse_stage)

        # caculate loss
        if coarse_stage:
            losses = {"loss_coarse": self.loss(rgb, batched_targets)}
            return losses, weights
        else:
            losses = {"loss_fine": self.loss(rgb, batched_targets)}
            return losses

    def inference(self):
        assert not self.training

    def loss(self, preds, targets):
        return F.mse_loss(preds, targets, reduction='mean')
