import argparse

from nerf.config import get_cfg
from nerf.engine import default_setup, DefaultTrainer


def get_parser():
    parser = argparse.ArgumentParser(description="NeRF demo for builtin configs")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    return parser.parse_args()


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup_cfg(args)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()
    return None


if __name__ == "__main__":
    args = get_parser()
    main(args)
    