from .dataset import build_blender_data
from .data_loader import DataLoaderTrain, DataLoaderTest

__all__ = ["build_train_loader", "build_test_loader",]


def build_train_loader(cfg):
    dataset = cfg.DATASET.TYPE
    use_pixel_batching = cfg.DATASET.USE_PIXEL_BATCHING
    ues_viewdirs = cfg.DATASET.USE_VIEWDIRS
    batch_size = cfg.SOLVER.BATCH_SIZE
    split = cfg.DATASET.TRAIN
    skip = 1
    half_res = cfg.DATASET.HALF_RES
    precrop_iters = cfg.DATASET.PRECROP_ITERS
    precrop_frac = cfg.DATASET.PRECROP_FRAC

    if dataset == "blender":
        imgs, poses, hwf, near, far = build_blender_data(
            cfg, split, skip, half_res)
    else:
        raise NotImplementedError

    return DataLoaderTrain(
        imgs, poses, hwf, near, far, use_pixel_batching, 
        ues_viewdirs, batch_size, precrop_iters, precrop_frac,
    )


def build_test_loader(cfg):
    dataset = cfg.DATASET.TYPE
    ues_viewdirs = cfg.DATASET.USE_VIEWDIRS
    split = cfg.DATASET.TEST
    skip = cfg.DATASET.SKIP
    half_res = cfg.DATASET.HALF_RES
    
    if dataset == "blender":
        imgs, poses, hwf, near, far = build_blender_data(
            cfg, split, skip, half_res)
    else:
        raise NotImplementedError
    
    return  DataLoaderTest(imgs, poses, hwf, near, far, ues_viewdirs)
