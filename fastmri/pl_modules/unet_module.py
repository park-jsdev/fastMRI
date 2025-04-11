"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser

import torch
from torch.nn import functional as F

from fastmri.models import Unet

from .mri_module import MriModule

from piq import ssim, psnr
from fastmri.losses.roi_loss import roi_weighted_loss

import csv
import os
from piq import ssim, psnr
import torch.nn.functional as F
from fastmri.losses.roi_loss import make_soft_center_mask

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

        mask = make_soft_center_mask(recon.shape, margin_ratio=self.roi_margin, strength=self.roi_strength)
        weighted_diff = (1 - ssim(recon, target, reduction='none')) * mask
        roi_ssims.append((1 - weighted_diff.sum() / mask.sum()).item())

        roi_mses.append(((recon - target) ** 2 * mask).mean().item())

    # Compute averages
    avg_global_ssim = sum(global_ssims) / len(global_ssims)
    avg_roi_ssim = sum(roi_ssims) / len(roi_ssims)
    avg_global_psnr = sum(global_psnrs) / len(global_psnrs)
    avg_roi_mse = sum(roi_mses) / len(roi_mses)

    # Log to TensorBoard
    self.log("val/global_ssim", avg_global_ssim, prog_bar=True)
    self.log("val/roi_ssim", avg_roi_ssim, prog_bar=True)
    self.log("val/global_psnr", avg_global_psnr, prog_bar=True)
    self.log("val/roi_mse", avg_roi_mse, prog_bar=True)

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


class UnetModule(MriModule):
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

        # New params
        loss_type="l1",
        roi_weighting=False,
        roi_margin=0.2,  
        roi_strength=5.0,

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
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        # new loss configs
        self.loss_type = loss_type
        self.roi_weighting = roi_weighting
        self.roi_margin = roi_margin
        self.roi_strength = roi_strength

        self.unet = Unet(
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

        # loss = F.l1_loss(output, batch.target)
        loss = roi_weighted_loss(
            output,
            batch.target,
            loss_type=self.loss_type,
            use_roi=self.roi_weighting,
            margin_ratio=self.roi_margin,
            strength=self.roi_strength,
        )


        self.log("loss", loss.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)

        val_loss = roi_weighted_loss(
            output,
            batch.target,
            loss_type=self.loss_type,
            use_roi=self.roi_weighting,
            margin_ratio=self.roi_margin,
            strength=self.roi_strength,
        )

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
        optim = torch.optim.RMSprop(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

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
        parser.add_argument("--loss-type", default="l1", choices=["l1", "l2"], type=str)
        parser.add_argument("--roi-weighting", action="store_true", help="Use center-weighted ROI loss")
        parser.add_argument("--roi-margin", default=0.2, type=float, help="Margin ratio for center ROI mask")
        parser.add_argument("--roi-strength", default=5.0, type=float, help="Strength of soft ROI Gaussian weighting")


        return parser
