import math

import torch
import torch.nn as nn

__all__ = ["build_embedder", ]


def embedding(x, L):
    if x is None:
        return None

    freq = 2 ** torch.arange(L) * math.pi
    freq = x[..., None] * freq
    return torch.cat((freq.sin(), freq.cos()), dim=-1).flatten(x.dim() - 1)


class Embedder(nn.Module):
    def __init__(self, dim_position, dim_direction) -> None:
        super().__init__()
        self.dim_pos = dim_position
        self.dim_dir = dim_direction

    def forward(self, position, direction):
        pos_embed = embedding(position, self.dim_pos)
        dir_embed = embedding(direction, self.dim_dir)

        return pos_embed, dir_embed


def build_embedder(cfg):
    dim_position = cfg.MODEL.EMBEDDER.POSITION_DIM
    dim_direction = cfg.MODEL.EMBEDDER.DIRECTION_DIM
    embedder = Embedder(dim_position, dim_direction)
    return embedder

