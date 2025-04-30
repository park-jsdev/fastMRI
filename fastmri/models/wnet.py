from torch import nn
from .encoder import Encoder
from .decoder import Decoder
import torch.nn.functional as F


class Wnet(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes = 5, kernel_size=3, stride=1, padding=0):
        super(Wnet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_classes = num_classes

        self.encoder = Encoder(squeeze=self.num_classes, ch_mul=64, in_chans=self.in_channels)
        self.decoder = Decoder(squeeze=self.num_classes, ch_mul=64, in_chans=self.out_channels)

    def forward(self, x):
        encoded = self.encoder(x)
        seg_map = F.softmax(encoded, 1)
        decoded = self.decoder(seg_map)  # recon

        return decoded, seg_map


