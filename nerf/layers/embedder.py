import math

import torch
import torch.nn as nn

__all__ = ["build_embedder", ]


def embedding(x, L, include_input=True):
    if x is None:
        return None

    freq = 2 ** torch.arange(L, device=x.device) * math.pi
    freq = x[..., None] * freq

    embed = torch.cat((freq.sin(), freq.cos()), dim=-1).flatten(x.dim() - 1)
    if include_input:
        embed = torch.cat([x, embed], dim=-1)
    return embed


class Embedder(nn.Module):
    def __init__(self, dim_position, dim_direction, include_input=True) -> None:
        super().__init__()
        self.dim_pos = dim_position
        self.dim_dir = dim_direction
        self.include_input = include_input

    def forward(self, position, direction):
        pos_embed = embedding(position, self.dim_pos, self.include_input)
        dir_embed = embedding(direction, self.dim_dir, self.include_input)

        return pos_embed, dir_embed


def build_embedder(cfg):
    dim_position = cfg.MODEL.EMBEDDER.POSITION_DIM
    dim_direction = cfg.MODEL.EMBEDDER.DIRECTION_DIM
    include_input = cfg.MODEL.MLP.INCLUDE_INPUT
    embedder = Embedder(dim_position, dim_direction, include_input)
    return embedder
