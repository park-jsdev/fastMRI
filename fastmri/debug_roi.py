import torch
from fastmri.losses import ROILoss

# configure Gaussian ROI
loss_fn = ROILoss(
    loss_type="l2",
    use_roi=True,
    roi_mask="gaussian",
    margin_ratio=0.5,
    strength=10.0,
)

# toy data
output = torch.zeros(1, 100, 100)
target = torch.ones( (1, 100, 100) )

# trigger forward()
l = loss_fn(output, target)
print("loss =", l.item())
