import torch
import pytest
from fastmri.losses import ROILoss

@pytest.mark.parametrize("mask_type,margin,strength,expected_mean", [
    ("binary", 0.2,   5.0, 1.0),  # all differences in ROI are 1, so mean should be 1.0
    ("gaussian", 0.5, 10.0, 1.0), # we won’t assert exact value, just that it’s <= 1.0
])
def test_roi_loss_simple(mask_type, margin, strength, expected_mean):
    # toy batch: output=0, target=1 → diff=1 everywhere
    output = torch.zeros(1, 100, 100)
    target = torch.ones((1, 100, 100))

    loss_fn = ROILoss(
        loss_type="l2",
        use_roi=True,
        roi_mask=mask_type,
        margin_ratio=margin,
        strength=strength
    )

    l = loss_fn(output, target).item()

    if expected_mean is not None:
        # for binary mask, we zero out border but diff=1 inside so mean is still 1.0
        assert pytest.approx(expected_mean, rel=1e-6) == l
    else:
        # for Gaussian mask, we should get something strictly between 0 and 1
        assert 0.0 < l <= 1.0

def test_roi_loss_no_roi():
    # when use_roi=False, mask=1 everywhere → MSE=1.0
    output = torch.zeros(2, 100, 100)
    target = torch.ones((2, 100, 100))
    loss_fn = ROILoss(use_roi=False, loss_type="l2")
    assert pytest.approx(1.0, rel=1e-6) == loss_fn(output, target).item()

def test_gaussian_mask_shape_and_peak():
    loss_fn = ROILoss(
        use_roi=True,
        roi_mask="gaussian",
        margin_ratio=0.5,
        strength=10.0,
    )
    # make a dummy “image” of the right shape
    dummy = torch.zeros(1, 100, 100)
    mask = loss_fn.make_mask(dummy)     # -> [1,1,100,100] Gaussian weights

    # shape is correct
    assert mask.shape == (1,1,100,100)

    # center pixel is strictly larger than a corner
    center_val = mask[0,0,50,50].item()
    corner_val = mask[0,0,0,0].item()
    assert center_val > corner_val > 0.0

    # mask values lie in (0,1], peaking at 1.0 in the very center
    assert pytest.approx(1.0, rel=1e-1) == mask.max().item()
