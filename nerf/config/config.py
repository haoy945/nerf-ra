from yacs.config import CfgNode


def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.

    Returns:
        a CfgNode instance.
    """
    from .defaults import _C

    return _C.clone()
