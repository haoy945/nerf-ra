from .dataset import build_blender_data
from .data_loader import DataLoaderTrain, DataLoaderTest

__all__ = ["build_train_loader", "build_test_loader",]


def build_train_loader(cfg):
    dataset = cfg.DATASET.TYPE
    use_pixel_batching = cfg.DATASET.USE_PIXEL_BATCHING
    ues_viewdirs = cfg.DATASET.USE_VIEWDIRS
    batch_size = cfg.SOLVER.BATCH_SIZE
    split = cfg.DATASET.TRAIN

    if dataset == "blender":
        imgs, poses, hwf, near, far = build_blender_data(cfg, split)
    else:
        raise NotImplementedError

    return DataLoaderTrain(
        imgs, poses, hwf, near, far, 
        use_pixel_batching, ues_viewdirs, batch_size,
    )


def build_test_loader(cfg):
    dataset = cfg.DATASET.TYPE
    ues_viewdirs = cfg.DATASET.USE_VIEWDIRS
    split = cfg.DATASET.TEST
    
    if dataset == "blender":
        imgs, poses, hwf, near, far = build_blender_data(cfg, split)
    else:
        raise NotImplementedError
    
    return  DataLoaderTest(imgs, poses, hwf, near, far, ues_viewdirs)
