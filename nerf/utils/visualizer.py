import os
import math
import torch
import imageio
import numpy as np

__all__ = ["visualize", ]


def visualize(img, filename, savedir):
    save_path = os.path.join(savedir, filename)

    if isinstance(img, torch.Tensor):
        img = img.numpy()
    if img.dtype == np.float32:
        img = (255 * np.clip(img, 0, 1)).astype(np.uint8)
    if img.ndim == 2:
        H = W = int(math.sqrt(img.shape[0]))
        img = img.reshape(H, W, -1)
        
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    imageio.imwrite(save_path, img)
