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
                               padding=self.padding)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=self.padding)

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


class SeparableConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(SeparableConvModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.spatial1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels, padding=1)
        self.depth1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv1 = lambda x: self.depth1(self.spatial1(x))

        self.spatial2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.depth2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        self.conv2 = lambda x: self.depth2(self.spatial2(x))

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


class Encoder(nn.Module):
    def __init__(self, squeeze, ch_mul=64, in_chans=3):
        super(Encoder, self).__init__()

        self.enc1 = ConvModule(in_chans, ch_mul)
        self.enc2 = SeparableConvModule(ch_mul, 2 * ch_mul)
        self.enc3 = SeparableConvModule(2 * ch_mul, 4 * ch_mul)
        self.enc4 = SeparableConvModule(4 * ch_mul, 8 * ch_mul)

        self.middle = SeparableConvModule(8 * ch_mul, 16 * ch_mul)

        self.up1 = nn.ConvTranspose2d(16 * ch_mul, 8 * ch_mul, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec1 = SeparableConvModule(16 * ch_mul, 8 * ch_mul)
        self.up2 = nn.ConvTranspose2d(8 * ch_mul, 4 * ch_mul, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2 = SeparableConvModule(8 * ch_mul, 4 * ch_mul)
        self.up3 = nn.ConvTranspose2d(4 * ch_mul, 2 * ch_mul, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec3 = SeparableConvModule(4 * ch_mul, 2 * ch_mul)
        self.up4 = nn.ConvTranspose2d(2 * ch_mul, ch_mul, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec4 = ConvModule(2 * ch_mul, ch_mul)

        self.final = nn.Conv2d(ch_mul, squeeze, kernel_size=(1, 1))
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        enc1 = self.enc1(x)

        enc2 = self.enc2(F.max_pool2d(enc1, (2, 2)))

        enc3 = self.enc3(F.max_pool2d(enc2, (2, 2)))

        enc4 = self.enc4(F.max_pool2d(enc3, (2, 2)))

        middle = self.middle(F.max_pool2d(enc4, (2, 2)))

        up1 = torch.cat([enc4, self.up1(middle)], 1)
        dec1 = self.dec1(up1)

        up2 = torch.cat([enc3, self.up2(dec1)], 1)
        dec2 = self.dec2(up2)

        up3 = torch.cat([enc2, self.up3(dec2)], 1)
        dec3 = self.dec3(up3)

        up4 = torch.cat([enc1, self.up4(dec3)], 1)
        dec4 = self.dec4(up4)

        final = self.final(dec4)

        return final