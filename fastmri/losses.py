"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

# =============================
# ROI Loss
# =============================

def gaussian_mask(shape, device, margin_ratio=0.2, strength=5.0):
    # soft center via Gaussian
    # shape is (B, C, H, W); soft center via Gaussian
    B, C, H, W = shape
    y = torch.linspace(-1,1,H,device=device).view(-1,1).expand(H,W)
    x = torch.linspace(-1,1,W,device=device).view(1,-1).expand(H,W)
    # scale coordinates so margin_ratio controls spread:
    y = y / margin_ratio
    x = x / margin_ratio
    g = torch.exp(-strength * (x**2 + y**2))
    m = g.unsqueeze(0).unsqueeze(0).expand(B,C,H,W)
    return m

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
    
    def calculate_ssim_loss(self, batch_size, mask, output, target, batch_max_value):
        loss_fn = SSIMLoss()
        data_range = batch_max_value

        if self.use_roi:
            loss = loss_fn(output, target, data_range, mask)
        else:
            loss = loss_fn(output, target, data_range)
        return loss

    def forward(self, output, target, batch_max_value):
        # ensure [B,1,H,W]
        if output.ndim == 3:
            output = output.unsqueeze(1)
            target = target.unsqueeze(1)
        B, _, H, W = output.shape
        # pick mask
        if not self.use_roi:
            mask = torch.ones_like(output)
        else:
            mask = self.make_mask(output)

        # elementwise difference
        if self.loss_type == "l2":
            diff = (output - target) ** 2
        elif self.loss_type == "ssim":
            loss = self.calculate_ssim_loss(B, mask, output, target, batch_max_value)
            return loss
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

    def crop_gaussian_mask(self, mask, target_size):
        """Crop the center of the Gaussian mask to the target size."""
        original_size = mask.shape[2:]  # Assuming the mask shape is [1, 1, H, W]
        start_x = (original_size[0] - target_size[0]) // 2
        start_y = (original_size[1] - target_size[1]) // 2
        cropped_mask = mask[:, :, start_x:start_x + target_size[0], start_y:start_y + target_size[1]]
        return cropped_mask

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        data_range: torch.Tensor,
        mask: torch.Tensor = None,
        reduced: bool = True,
    ):
        assert isinstance(self.w, torch.Tensor)
        self.w = self.w.to(X.device)
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
        
        if mask is not None:
            # Ensure the mask has the same spatial dimensions as S
            target_size = S.shape[2:]
            mask = self.crop_gaussian_mask(mask, target_size)
            assert mask.shape[2:] == S.shape[2:], "Mask dimensions do not match SSIM output dimensions"
            # Weight the SSIM loss with the mask (element-wise multiplication)
            S = S * mask  # Apply the mask element-wise

        if reduced:
            # Return the weighted mean SSIM loss (mean over the weighted values)
            return 1 - S.sum() / mask.sum() if mask is not None else 1 - S.mean()
        else:
            # Return the un-reduced (pixel-wise) weighted SSIM
            return 1 - S

class SSIMLossMaskGauss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03, margin_ratio: float=0.2, strength: float=5.0):
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
        self.margin_ratio = margin_ratio
        self.strength = strength

    def crop_gaussian_mask(self, mask, target_size):
        """Crop the center of the Gaussian mask to the target size."""
        original_size = mask.shape[2:]  # Assuming the mask shape is [1, 1, H, W]
        start_x = (original_size[0] - target_size[0]) // 2
        start_y = (original_size[1] - target_size[1]) // 2
        cropped_mask = mask[:, :, start_x:start_x + target_size[0], start_y:start_y + target_size[1]]
        return cropped_mask

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        data_range: torch.Tensor,
        reduced: bool = True,
    ):
        X = X.unsqueeze(1)
        Y = Y.unsqueeze(1)
        assert isinstance(self.w, torch.Tensor)
        self.w = self.w.to(X.device)
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
        mask = gaussian_mask(X.shape, X.device, self.margin_ratio, self.strength)
        
        # Ensure the mask has the same spatial dimensions as S
        target_size = S.shape[2:]
        mask = self.crop_gaussian_mask(mask, target_size)
        assert mask.shape[2:] == S.shape[2:], "Mask dimensions do not match SSIM output dimensions"
        # Weight the SSIM loss with the mask (element-wise multiplication)
        S = S * mask  # Apply the mask element-wise
        # Return the weighted mean SSIM loss (mean over the weighted values)
        return 1 - S.sum() / mask.sum()
    
class L2LossMaskGauss(nn.Module):
    """
    L2 loss module with Gaussian mask.
    """
    def __init__(self, margin_ratio: float=0.2, strength: float=5.0):
        super().__init__()
        self.margin_ratio = margin_ratio
        self.strength = strength

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        # Ensure the input tensors have the same shape
        X = X.unsqueeze(1)
        Y = Y.unsqueeze(1)
        assert X.shape == Y.shape, "Input tensors must have the same shape"
        
        # Compute the L2 loss
        l2_loss = F.mse_loss(X, Y, reduction='none')
        
        # Create a Gaussian mask
        mask = gaussian_mask(X.shape, X.device, self.margin_ratio, self.strength)
        
        # Apply the mask to the L2 loss
        masked_l2_loss = l2_loss * mask
        # Return the mean of the masked L2 loss
        return masked_l2_loss.sum() / mask.sum()