import torch
import torch.nn as nn

from ...layers import Linear


class NerfMlp(nn.Module):
    def __init__(self, num_layers, skips, position_dim, direction_dim, middle_dim) -> None:
        super().__init__()
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
        self.bottleneck1 = Linear(in_dim, out_dim)
        self.sigma_branch = Linear(in_dim, 1)
        self.bottleneck2 = Linear(
            out_dim + direction_dim, out_dim / 2, activation=nn.ReLU()
        )
        self.color_branch = Linear(out_dim / 2, 3)

    def forward(self, position, direction):
        output = position
        for fc in self.fcs:
            output = fc(output)
        
        sigma = self.sigma_branch(output)
        output = self.bottleneck1(output)
        output = self.bottleneck2(torch.cat([output, direction], dim=-1))
        color = self.color_branch(output)

        return color, sigma
