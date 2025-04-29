"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================
# ROI Loss
# =============================

class ROILoss(nn.Module):
    def __init__(
        self,
        loss_type: str = "l2",             # "l2" or "l1"
        use_roi: bool = False,             # whether to apply any ROI mask
        roi_mask: str = "binary",          # "binary" or "gaussian"
        margin_ratio: float = 0.2,         # for binary mask
        strength: float = 5.0,             # for gaussian mask
    ):
        super().__init__()
        self.loss_type    = loss_type
        self.use_roi      = use_roi
        self.roi_mask     = roi_mask
        self.margin_ratio = margin_ratio
        self.strength     = strength

    def make_mask(self, tensor: torch.Tensor):
        # tensor is [B,H,W] or [B,1,H,W]; want a [B,1,H,W] mask on same device
        output = tensor.unsqueeze(1) if tensor.ndim == 3 else tensor
        device = output.device
        if self.roi_mask == "binary":
            return self._binary_mask(output.shape, device)
        return self._gaussian_mask(output.shape, device)

    def _binary_mask(self, shape, device):
        # keep center, zero margin
        B, C, H, W = shape

        h_m = int(H * self.margin_ratio)
        w_m = int(W * self.margin_ratio)
        m = torch.zeros((B, C, H, W), device=device)
        m[:, :, h_m:H-h_m, w_m:W-w_m] = 1.0
        return m

    def _gaussian_mask(self, shape, device):
        # soft center via Gaussian
        # shape is (B, C, H, W); soft center via Gaussian
        B, C, H, W = shape
        y = torch.linspace(-1,1,H,device=device).view(-1,1).expand(H,W)
        x = torch.linspace(-1,1,W,device=device).view(1,-1).expand(H,W)
        # scale coordinates so margin_ratio controls spread:
        y = y / self.margin_ratio
        x = x / self.margin_ratio
        g = torch.exp(-self.strength * (x**2 + y**2))
        m = g.unsqueeze(0).unsqueeze(0).expand(B,C,H,W)
        return m

    def forward(self, output, target):
        # ensure [B,1,H,W]
        if output.ndim == 3:
            output = output.unsqueeze(1)
            target = target.unsqueeze(1)

        # pick mask
        if not self.use_roi:
            mask = torch.ones_like(output)
        else:
            mask = self.make_mask(output)

        # elementwise difference
        if self.loss_type == "l2":
            diff = (output - target) ** 2
        else:
            diff = torch.abs(output - target)
        weighted = diff * mask

        if self.use_roi:
            # average only over the ROI (binary or gaussian)
            return weighted.sum() / mask.sum()
        else:
            # uniform average over the whole image
            return weighted.mean()

# =============================
# SSIM Loss
# =============================

class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        data_range: torch.Tensor,
        reduced: bool = True,
    ):
        assert isinstance(self.w, torch.Tensor)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        if reduced:
            return 1 - S.mean()
        else:
            return 1 - S

# =============================
# PSNR Loss (as a metric-style loss)
# =============================

class PSNRLoss(nn.Module):
    def __init__(
        self,
        use_roi: bool = False,
        roi_mask: str = "binary",      # "binary" or "gaussian"
        margin_ratio: float = 0.2,      # for binary mask
        strength: float = 5.0,          # for gaussian mask
    ):
        super().__init__()
        self.use_roi = use_roi
        self.roi_mask = roi_mask
        self.margin_ratio = margin_ratio
        self.strength = strength

    def make_mask(self, tensor: torch.Tensor):
        # tensor is [B,H,W] or [B,1,H,W]; return a [B,1,H,W] mask on same device
        output = tensor.unsqueeze(1) if tensor.ndim == 3 else tensor
        device = output.device
        if self.roi_mask == "binary":
            return self._binary_mask(output.shape, device)
        return self._gaussian_mask(output.shape, device)

    def _binary_mask(self, shape, device):
        B, C, H, W = shape
        h_m = int(H * self.margin_ratio)
        w_m = int(W * self.margin_ratio)
        m = torch.zeros((B, C, H, W), device=device)
        m[:, :, h_m:H-h_m, w_m:W-w_m] = 1.0
        return m

    def _gaussian_mask(self, shape, device):
        B, C, H, W = shape
        y = torch.linspace(-1,1,H,device=device).view(-1,1).expand(H,W)
        x = torch.linspace(-1,1,W,device=device).view(1,-1).expand(H,W)
        y = y / self.margin_ratio
        x = x / self.margin_ratio
        g = torch.exp(-self.strength * (x**2 + y**2))
        m = g.unsqueeze(0).unsqueeze(0).expand(B,C,H,W)
        return m

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        if output.ndim == 3:
            output = output.unsqueeze(1)
            target = target.unsqueeze(1)

        if not self.use_roi:
            mask = torch.ones_like(output)
        else:
            mask = self.make_mask(output)

        # Compute MSE with or without mask
        mse = ((output - target) ** 2) * mask
        if self.use_roi:
            mse = mse.sum(dim=[1,2,3]) / mask.sum(dim=[1,2,3])  # per image
        else:
            mse = mse.mean(dim=[1,2,3])  # per image

        # max value per image in batch
        max_vals = target.view(target.size(0), -1).max(dim=1)[0]
        psnr_vals = 10 * torch.log10((max_vals ** 2) / mse)

        return -psnr_vals.mean()  # Negative because we minimize loss