"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.filters import threshold_otsu

# =============================
# ROI Loss
# =============================

class ROILoss(nn.Module):
    def __init__(
        self,
        loss_type: str = "l2",             # "l2" or "l1"
        use_roi: bool = False,             # whether to apply any ROI mask
        roi_mask: str = "binary",          # "binary", "gaussian", or "otsu"
        margin_ratio: float = 0.2,         # for binary mask
        strength: float = 5.0,             # for gaussian mask
        percentile: float  = 0.3,        # for "percentile" mask: fraction of top‐brightness to keep
    ):
        super().__init__()
        self.loss_type    = loss_type
        self.use_roi      = use_roi
        self.roi_mask     = roi_mask
        self.margin_ratio = margin_ratio
        self.strength     = strength
        self.percentile   = percentile

    def make_mask(self, tensor: torch.Tensor):
        # tensor is [B,H,W] or [B,1,H,W]; want a [B,1,H,W] mask on same device
        output = tensor.unsqueeze(1) if tensor.ndim == 3 else tensor
        device = output.device

        if self.roi_mask == "binary":
            return self._binary_mask(output.shape, device)
        elif self.roi_mask == "gaussian":
            return self._gaussian_mask(output.shape, device)
        elif self.roi_mask == "otsu":
            return self._otsu_mask(output)
        else:
            return self._percentile_mask(output)


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
    
    def _otsu_mask(self, x: torch.Tensor):
        B,_,H,W = x.shape
        device = x.device
        out = torch.zeros_like(x)
        for b in range(B):
            arr = x[b,0].cpu().numpy()
            th  = threshold_otsu(arr)
            out[b,0] = (x[b,0] >= float(th)).to(device).float()
        return out

    def _percentile_mask(self, x: torch.Tensor):
        # keep top `percentile` fraction of brightest pixels in each batch-item
        B, _, H, W = x.shape
        flat = x.view(B, -1)
        q = 1.0 - self.percentile
        thresh = torch.quantile(flat, q, dim=1)
        
        # build mask
        return (x >= thresh.view(B,1,1,1)).float()


    def forward(self, output, target):
        # bring to 4D
        if output.ndim == 3:
            output, target = output.unsqueeze(1), target.unsqueeze(1)

        if not self.use_roi:
            mask = torch.ones_like(output)
        elif self.roi_mask in ("binary", "gaussian"):
            mask = self.make_mask(output) # masks still built on output‐shape
        else:  # otsu or percentile
            mask = self.make_mask(target) # intensity masks built on ground truth

        # compute diff
        diff = (output-target)**2 if self.loss_type=="l2" else torch.abs(output-target)
        weighted = diff * mask

        return weighted.sum() / mask.sum() if self.use_roi else weighted.mean()


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
