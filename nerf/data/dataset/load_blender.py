import os
import json

import cv2
import torch
import numpy as np
import imageio

__all__ = ["load_blender_data", "build_blender_data", ]

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, split='train', half_res=False, skip=1, white_bkgd=False):
    with open(os.path.join(basedir, 'transforms_{}.json'.format(split)), 'r') as fp:
        meta = json.load(fp)
    
    imgs = []
    poses = []
    for frame in meta['frames'][::skip]:
        fname = os.path.join(basedir, frame['file_path'] + '.png')
        imgs.append(imageio.imread(fname))
        poses.append(np.array(frame['transform_matrix']))
    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    if white_bkgd:
        imgs = imgs[..., :3] * imgs[..., -1:] + (1. - imgs[..., -1:])
    else:
        imgs = imgs[..., :3]

    near = 2.
    far = 6.

    return torch.tensor(imgs, dtype=torch.float32), torch.tensor(poses), [H, W, focal], near, far    


def build_blender_data(cfg, split):
    data_root = cfg.DATASET.ROOT_PATH
    datadir = cfg.DATASET.DATADIR
    datadir = os.path.join(data_root, datadir)
    half_res = cfg.DATASET.HALF_RES
    skip = cfg.DATASET.SKIP
    white_bkgd = cfg.DATASET.WHITE_BACKGROUND

    return load_blender_data(datadir, split, half_res, skip, white_bkgd)
