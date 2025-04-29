




# import torch
# from torch import nn
# import pytorch_lightning as pl
#
# from fastmri.losses import CTLoss, ReconstructionLoss  # your loss classes
# from fastmri.models import Wnet
# from fastmri.pl_modules.mri_module import MriModule
# import csv
# import os
# import torch.nn.functional as F
#
#
# class WnetModule(pl.LightningModule):
#     def __init__(
#         self,
#         in_chans,
#         out_chans,
#         chans,
#         num_pool_layers,
#         drop_prob,
#         lr,
#         lr_step_size,
#         lr_gamma,
#         weight_decay,
#         lambda_tv=0.1,
#     ):
#         super().__init__()
#
#         # Save hyperparameters for logging
#         self.save_hyperparameters()
#
#         # Model
#         self.wnet = Wnet(in_channels=in_chans, out_channels=out_chans, num_classes=5)
#
#         # Loss functions
#         self.ct_loss_fn = CTLoss(reduction="mean")   # encoder loss (segmentation)
#         self.recon_loss_fn = ReconstructionLoss(method="l2", reduction="mean")  # decoder loss (reconstruction)
#
#         # Optimizer params
#         self.lr = lr
#         self.lr_step_size = lr_step_size
#         self.lr_gamma = lr_gamma
#         self.weight_decay = weight_decay
#         self.lambda_tv = lambda_tv  # scaling TV loss
#
#     def forward(self, x):
#         # W-Net returns (encoded, reconstructed)
#         return self.wnet(x)
#
#     def training_step(self, batch, batch_idx):
#         input, target, _, _, _ = batch  # FastMRI returns 5-tuple
#
#         encoded, reconstructed = self.forward(input)
#
#         # Compute losses
#         loss_seg = self.ct_loss_fn(encoded)  # segmentation CT-loss
#         loss_recon = self.recon_loss_fn(reconstructed, input)  # reconstruction L2 loss
#
#         loss = loss_seg + loss_recon
#
#         # Log losses
#         self.log("train_loss", loss)
#         self.log("train_ct_loss", loss_seg)
#         self.log("train_recon_loss", loss_recon)
#
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         input, target, _, _, _ = batch
#
#         encoded, reconstructed = self.forward(input)
#
#         loss_seg = self.ct_loss_fn(encoded)
#         loss_recon = self.recon_loss_fn(reconstructed, input)
#
#         loss = loss_seg + loss_recon
#
#         # Log validation losses
#         self.log("val_loss", loss, prog_bar=True)
#         self.log("val_ct_loss", loss_seg)
#         self.log("val_recon_loss", loss_recon)
#
#         return loss
#
#     def add_model_specific_args(parent_parser):
#         parser = parent_parser.add_argument_group("WnetModule")
#
#         parser.add_argument(
#             "--in_chans", default=1, type=int, help="Number of input channels to the model"
#         )
#         parser.add_argument(
#             "--out_chans", default=1, type=int, help="Number of output channels to the model"
#         )
#         parser.add_argument(
#             "--chans", default=32, type=int, help="Number of channels in W-Net feature maps"
#         )
#         parser.add_argument(
#             "--num_pool_layers", default=4, type=int, help="Number of pooling layers in encoder/decoder"
#         )
#         parser.add_argument(
#             "--drop_prob", default=0.0, type=float, help="Dropout probability"
#         )
#         parser.add_argument(
#             "--lr", default=0.001, type=float, help="Learning rate"
#         )
#         parser.add_argument(
#             "--lr_step_size", default=40, type=int, help="Epoch step to decay LR"
#         )
#         parser.add_argument(
#             "--lr_gamma", default=0.1, type=float, help="LR decay factor"
#         )
#         parser.add_argument(
#             "--weight_decay", default=0.0, type=float, help="Weight decay strength"
#         )
#         parser.add_argument(
#             "--lambda_tv", default=0.1, type=float, help="TV loss scaling factor"
#         )
#
#         return parent_parser
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(
#             self.parameters(),
#             lr=self.lr,
#             weight_decay=self.weight_decay,
#         )
#
#         scheduler = torch.optim.lr_scheduler.StepLR(
#             optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
#         )
#
#         return [optimizer], [scheduler]
