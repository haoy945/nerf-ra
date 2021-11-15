import torch
import torch.nn as nn

from ...layers import Linear

__all__ = ["NerfMlp", ]


class NerfMlp(nn.Module):
    """
    This module is a 9 layers MLP.
    Given position and direction, output color and sigma.
    """

    def __init__(self, num_layers, skips, position_dim, direction_dim, middle_dim) -> None:
        super().__init__()
        self.skips = skips

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
        self.sigma_branch = Linear(in_dim, 1)
        self.bottleneck1 = Linear(in_dim, out_dim)
        self.bottleneck2 = Linear(
            out_dim + direction_dim, out_dim // 2, activation=nn.ReLU()
        )
        self.color_branch = Linear(out_dim // 2, 3)

    def forward(self, position, direction):
        """
        Args:
            position (tensor): position embedding sized [batch_size, position_dim].
            direction (tensor): direction embedding sized [batch_size, direction_dim].
        """
        output = position
        for i, fc in enumerate(self.fcs):
            if i in self.skips:
                output = torch.cat((output, position), dim=-1)
            output = fc(output)
        
        sigma = self.sigma_branch(output)
        output = self.bottleneck1(output)
        output = self.bottleneck2(torch.cat([output, direction], dim=-1))
        color = self.color_branch(output)

        return color, sigma
