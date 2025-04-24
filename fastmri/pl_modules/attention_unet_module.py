"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser

import torch
from fastmri.models import AttentionUnet
from fastmri.pl_modules.mri_module import MriModule
from piq import ssim, psnr
from fastmri.losses import ROILoss, SSIMLoss, SSIMLossMaskGauss, L2LossMaskGauss
import csv
import os
import torch.nn.functional as F

import pdb

def validation_epoch_end(self, outputs):
    global_ssims, roi_ssims = [], []
    global_psnrs, roi_mses = [], []

    for out in outputs:
        recon = out["output"].unsqueeze(0)
        target = out["target"].unsqueeze(0)

        # Normalize to [0, 1]
        recon = torch.clamp(recon / recon.max(), 0, 1)
        target = torch.clamp(target / target.max(), 0, 1)

        # Metrics
        global_ssims.append(ssim(recon, target, data_range=1.0).item())
        global_psnrs.append(psnr(recon, target, data_range=1.0).item())

        # ROI mask for metric
        # build a 4D mask from our 3D recon tensor
        mask = self.loss_fn.make_mask(recon)  # implements the unsqueeze‑and‑route logic

        # Mask image logging
        self.logger.experiment.add_image(
            "val/mask",
            mask.squeeze(0),  # [1,H,W] or [H,W]
            self.current_epoch
        )

        # ROI-SSIM (invert SSIM to get "difference" then weight)
        diff = 1 - ssim(recon, target, reduction="none")
        weighted_diff = diff * mask
        roi_ssims.append((1 - weighted_diff.sum() / mask.sum()).item())

        # ROI‑MSE
        roi_mses.append(((recon - target) ** 2 * mask).mean().item())

    # average
    avg_global_ssim = sum(global_ssims) / len(global_ssims)
    avg_global_psnr = sum(global_psnrs) / len(global_psnrs)
    avg_roi_ssim    = sum(roi_ssims)    / len(roi_ssims)
    avg_roi_mse     = sum(roi_mses)     / len(roi_mses)


    # log
    self.log("val/global_ssim", avg_global_ssim, prog_bar=True)
    self.log("val/global_psnr", avg_global_psnr, prog_bar=True)
    self.log("val/roi_ssim",    avg_roi_ssim,    prog_bar=True)
    self.log("val/roi_mse",     avg_roi_mse,     prog_bar=True)

    # Print to console
    print(f"\n[Validation Metrics @ Epoch {self.current_epoch}]")
    print(f"Global SSIM     : {avg_global_ssim:.4f}")
    print(f"ROI SSIM        : {avg_roi_ssim:.4f}")
    print(f"Global PSNR     : {avg_global_psnr:.2f} dB")
    print(f"ROI MSE         : {avg_roi_mse:.6f}")

    # Save to CSV
    csv_path = os.path.join(self.logger.log_dir, "val_metrics.csv")
    write_header = not os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "global_ssim", "roi_ssim", "global_psnr", "roi_mse"])
        writer.writerow([self.current_epoch, avg_global_ssim, avg_roi_ssim, avg_global_psnr, avg_roi_mse])


class AttentionUnetModule(MriModule):
    """
    Unet training module.

    This can be used to train baseline U-Nets from the paper:

    J. Zbontar et al. fastMRI: An Open Dataset and Benchmarks for Accelerated
    MRI. arXiv:1811.08839. 2018.
    """
    def __init__(
        self,
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,


        # –– ROI loss args
        loss_type:      str   = "l1",  # "l1" or "l2"
        roi_weighting:  bool  = False,  # turn ROI on/off
        roi_mask:       str   = "binary",  # "binary" or "gaussian"
        roi_margin:     float = 0.2,  # for binary mask
        roi_strength:   float = 5.0,  # for gaussian mask

        **kwargs,
    ):
        """
        Args:
            in_chans (int, optional): Number of channels in the input to the
                U-Net model. Defaults to 1.
            out_chans (int, optional): Number of channels in the output to the
                U-Net model. Defaults to 1.
            chans (int, optional): Number of output channels of the first
                convolution layer. Defaults to 32.
            num_pool_layers (int, optional): Number of down-sampling and
                up-sampling layers. Defaults to 4.
            drop_prob (float, optional): Dropout probability. Defaults to 0.0.
            lr (float, optional): Learning rate. Defaults to 0.001.
            lr_step_size (int, optional): Learning rate step size. Defaults to
                40.
            lr_gamma (float, optional): Learning rate gamma decay. Defaults to
                0.1.
            weight_decay (float, optional): Parameter for penalizing weights
                norm. Defaults to 0.0.
        """

        # pop our custom args so MriModule.__init__ isn't confused
        # (but keep them in locals for save_hyperparameters)
        for key in ("loss_type", "roi_weighting", "roi_mask", "roi_margin", "roi_strength"):
            kwargs.pop(key, None)

        super().__init__(**kwargs)

        self.save_hyperparameters()

        # restore all the standard args as attributes
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        # –– instantiate the ROI loss
        self.loss_fn = None
        if loss_type == "ssim" and roi_mask == "gaussian":
            self.loss_fn = SSIMLossMaskGauss(margin_ratio=roi_margin, strength=roi_strength)
        elif loss_type == "l2" and roi_mask == "gaussian":
            self.loss_fn = L2LossMaskGauss(margin_ratio=roi_margin, strength=roi_strength)
        else:
            self.loss_fn = ROILoss(
                loss_type = loss_type,
                use_roi = roi_weighting,
                roi_mask = roi_mask,
                margin_ratio = roi_margin,
                strength = roi_strength,
            )

        # –– model
        self.unet = AttentionUnet(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
            chans=self.chans,
            num_pool_layers=self.num_pool_layers,
            drop_prob=self.drop_prob,
        )

    def forward(self, image):
        return self.unet(image.unsqueeze(1)).squeeze(1)

    def training_step(self, batch, batch_idx):
        output = self(batch.image)

        # compute ROI‐weighted pixel loss directly
        loss = self.loss_fn(output, batch.target, batch.max_value)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)
        val_loss = self.loss_fn(output, batch.target, batch.max_value)
        self.log("validation_loss", val_loss, prog_bar=True)

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output * std + mean,
            "target": batch.target * std + mean,
            "val_loss": val_loss,
        }

    def test_step(self, batch, batch_idx):
        output = self.forward(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)

        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output": (output * std + mean).cpu().numpy(),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, self.lr_step_size, self.lr_gamma
        )
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # network params
        parser.add_argument(
            "--in_chans", default=1, type=int, help="Number of U-Net input channels"
        )
        parser.add_argument(
            "--out_chans", default=1, type=int, help="Number of U-Net output chanenls"
        )
        parser.add_argument(
            "--chans", default=1, type=int, help="Number of top-level U-Net filters."
        )
        parser.add_argument(
            "--num_pool_layers",
            default=4,
            type=int,
            help="Number of U-Net pooling layers.",
        )
        parser.add_argument(
            "--drop_prob", default=0.0, type=float, help="U-Net dropout probability"
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.001, type=float, help="RMSProp learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma", default=0.1, type=float, help="Amount to decrease step size"
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )
        parser.add_argument("--loss-type", default="l1", choices=["l1", "l2", "ssim"])
        parser.add_argument("--roi-weighting", action="store_true", help = "Enable ROI loss instead of uniform")
        parser.add_argument("--roi-mask", default="binary", choices = ["binary", "gaussian"], help = "Type of ROI mask")
        parser.add_argument("--roi-margin", default=0.2, type=float, help = "Fractional border to zero out (binary mask)")
        parser.add_argument("--roi-strength", default=5.0, type=float, help = "Sharpness of Gaussian mask")

        return parser
