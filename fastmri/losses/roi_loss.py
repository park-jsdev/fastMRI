import torch
import torch.nn.functional as F

def make_soft_center_mask(shape, margin_ratio=0.2, strength=5.0):
    """
    Returns a soft Gaussian mask where center pixels are weighted more,
    and edge pixels still contribute to loss but are weighted less.

    strength: higher = more emphasis on center.
    """
    _, _, H, W = shape

    # Create coordinate grid
    y = torch.linspace(-1, 1, H).view(-1, 1).repeat(1, W)
    x = torch.linspace(-1, 1, W).view(1, -1).repeat(H, 1)

    # 2D Gaussian centered at (0, 0)
    gaussian = torch.exp(-strength * (x**2 + y**2))  # shape [H, W]
    mask = gaussian.unsqueeze(0).unsqueeze(0)  # shape [1, 1, H, W]

    return mask.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def roi_weighted_loss(output, target, loss_type="l2", use_roi=False, margin_ratio=0.2, strength=5.0):
    if use_roi:
        mask = make_soft_center_mask(output.shape, margin_ratio, strength)
    else:
        mask = torch.ones_like(output)

    if loss_type == "l2":
        return ((output - target) ** 2 * mask).mean()
    elif loss_type == "l1":
        return (torch.abs(output - target) * mask).mean()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
