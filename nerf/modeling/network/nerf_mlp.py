import torch
import torch.nn as nn

from ...layers import Linear

__all__ = ["build_nerf_mlp", ]


class NerfMlp(nn.Module):
    """
    This module is a 9 layers MLP.
    Given position and direction, output color and sigma.
    """

    def __init__(self, num_layers, skips, position_dim, direction_dim, 
                 middle_dim, use_viewdirs) -> None:
        super().__init__()
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        in_dim = position_dim
        out_dim = middle_dim
        self.fcs = []
        for i in range(num_layers):
            if i in skips:
                in_dim += position_dim
            
            fc = Linear(in_dim, out_dim, activation=nn.ReLU())
            self.add_module("fc{}".format(i), fc)
            self.fcs.append(fc)

            in_dim = out_dim

        # TODO: why no activation
        if use_viewdirs:
            self.sigma_branch = Linear(in_dim, 1)
            self.bottleneck1 = Linear(in_dim, out_dim)
            self.bottleneck2 = Linear(
                out_dim + direction_dim, out_dim // 2, activation=nn.ReLU()
            )
            self.rgb_branch = Linear(out_dim // 2, 3)
        else:
            self.output_branch = Linear(in_dim, 4)

    def forward(self, position, direction):
        """
        Args:
            position (tensor): position embedding sized [bs, num_samples, position_dim].
            direction (tensor): direction embedding sized [bs, num_samples, direction_dim].
        """
        assert position.dim() == 3
        
        bs, num_samples, pos_dim = position.shape
        position = position.reshape([-1, pos_dim])
        if self.use_viewdirs:
            direction = direction.reshape([-1, direction.shape[-1]])

        output = self._forward(position, direction)
        output = output.reshape([bs, num_samples, 4])

        return output

    def _forward(self, position, direction):
        """
        Args:
            position (tensor): position embedding sized [bs*num_samples, position_dim].
            direction (tensor): direction embedding sized [bs*num_samples, direction_dim].
        """
        output = position
        for i, fc in enumerate(self.fcs):
            if i in self.skips:
                output = torch.cat((output, position), dim=-1)
            output = fc(output)
        
        if self.use_viewdirs:
            sigma = self.sigma_branch(output)
            output = self.bottleneck1(output)
            output = self.bottleneck2(torch.cat([output, direction], dim=-1))
            rgb = self.rgb_branch(output)
            output = torch.cat([rgb, sigma], dim=-1)
        else:
            output = self.output_branch(output)

        return output


def build_nerf_mlp(cfg):
    num_layers = cfg.MODEL.MLP.NUM_LAYERS
    skips = cfg.MODEL.MLP.SKIPS
    position_dim = cfg.MODEL.MLP.POSITION_DIM
    direction_dim = cfg.MODEL.MLP.DIRECTION_DIM
    middle_dim = cfg.MODEL.MLP.MIDDEL_DIM
    use_viewdirs = cfg.DATASET.USE_VIEWDIRS

    nerfmlp = NerfMlp(num_layers, skips, position_dim, direction_dim, 
                      middle_dim, use_viewdirs)
    return nerfmlp
