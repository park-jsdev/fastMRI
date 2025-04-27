import torch
from torch import nn
from torch.nn import functional as F


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(ConvModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=self.padding),
        self.conv1 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=self.padding),

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.bnorm1 = nn.BatchNorm2d(self.out_channels)
        self.bnorm2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bnorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bnorm2(x)
        x = self.relu2(x)

        return x
