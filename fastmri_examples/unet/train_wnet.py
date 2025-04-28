"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pathlib
from argparse import ArgumentParser

import pytorch_lightning as pl

from fastmri.data.mri_data import fetch_dir
from fastmri.pl_modules import FastMriDataModule, WnetModule
import yaml
import torch

# -----------------------
# SIMPLE IMAGE-ONLY TRANSFORM
# -----------------------
from fastmri.data.transforms import to_tensor

class ImageOnlyDataTransform:
    def __init__(self):
        pass

    def __call__(self, kspace, target, attrs, filename, slice_num):
        image = to_tensor(target)  # target is already in image space
        return image, target, attrs, filename, slice_num

# -----------------------
# MAIN FUNCTION
# -----------------------
def cli_main(args):
    pl.seed_everything(args.seed)

    # -----------------------
    # Data module
    # -----------------------
    train_transform = ImageOnlyDataTransform()
    val_transform = ImageOnlyDataTransform()
    test_transform = ImageOnlyDataTransform()

    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    )

    # -----------------------
    # Model
    # -----------------------
    model = WnetModule(
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        chans=args.chans,
        num_pool_layers=args.num_pool_layers,
        drop_prob=args.drop_prob,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        lambda_tv=args.lambda_tv,
    )

    # -----------------------
    # Trainer
    # -----------------------
    trainer = pl.Trainer.from_argparse_args(args)

    # -----------------------
    # Run
    # -----------------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")

# -----------------------
# BUILD ARGS
# -----------------------
def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path("../../fastmri_dirs.yaml")
    num_gpus = 2
    backend = "ddp"
    batch_size = 1 if backend == "ddp" else num_gpus

    # set defaults based on optional directory config
    data_path = fetch_dir("knee_path", path_config)
    default_root_dir = fetch_dir("log_path", path_config) / "wnet" / "wnet_demo"

    # operation mode
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )

    # data config
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(data_path=data_path, batch_size=batch_size, test_path=None)

    # WnetModule specific args
    parser = WnetModule.add_model_specific_args(parser)
    parser.set_defaults(
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        lambda_tv=0.1,
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,
        replace_sampler_ddp=False,
        strategy=backend,
        seed=42,
        deterministic=True,
        default_root_dir=default_root_dir,
        max_epochs=50,
    )

    args = parser.parse_args()
    args.default_root_dir = pathlib.Path(args.default_root_dir)

    # configure checkpointing
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_best = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
    )

    checkpoint_every_5 = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch{epoch:03d}",
        save_top_k=-1,
        every_n_epochs=5,
        save_on_train_epoch_end=True,
        verbose=True,
    )

    args.callbacks = [checkpoint_best, checkpoint_every_5]

    return args

# -----------------------
# RUN
# -----------------------
def run_cli():
    args = build_args()
    cli_main(args)

if __name__ == "__main__":
    run_cli()
