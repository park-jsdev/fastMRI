import torch
from torch import nn


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # First conv 3 channel into 64
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=self.padding)

        # Second conv 64 to 64
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=self.padding)

        # ReLU non-linarity for each conv
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        # Batch normalization after conv --> 64 feat channels
        self.bnorm1 = nn.BatchNorm2d(self.out_channels)
        self.bnorm2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        # x should be 3 channels (but doesn't matter)
        # into first conv
        # print(f'x.shape: {x.shape}')
        x = self.conv1(x)
        x = self.bnorm1(x)
        x = self.relu1(x)

        # second conv layer
        x = self.conv2(x)
        x = self.bnorm2(x)
        x = self.relu2(x)

        return x

# Code refactored from https://aswali.github.io/WNet/
class SeparableConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SeparableConvModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.spatial1 = nn.Conv2d(in_channels, in_channels, kernel_size=self.kernel_size, groups=self.in_channels, padding=self.padding)
        self.depth1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv1 = lambda x: self.depth1(self.spatial1(x))

        self.spatial2 = nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding, groups=self.out_channels)
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


class Decoder(nn.Module):
    def __init__(self, feat_channel=64, in_channels=3, num_classes=5):
        super(Decoder, self).__init__()

        # Downwards in the U 1 to 4
        # 1st module has simple 3x3 conv
        self.dec10 = ConvModule(num_classes, feat_channel)

        # Subsequent modules have separable conv 3x3
        self.dec11 = SeparableConvModule(feat_channel, 2 * feat_channel)
        self.dec12 = SeparableConvModule(2 * feat_channel, 4 * feat_channel)
        self.dec13 = SeparableConvModule(4 * feat_channel, 8 * feat_channel)

        # Bottom module 5
        self.dec14 = SeparableConvModule(8 * feat_channel, 16 * feat_channel)

        # Upwards in the U 6 to 9
        self.dec15 = SeparableConvModule(16 * feat_channel, 8 * feat_channel)
        self.dec16 = SeparableConvModule(8 * feat_channel, 4 * feat_channel)
        self.dec17 = SeparableConvModule(4 * feat_channel, 2 * feat_channel)

        # Final module has simple 3x3 conv
        self.dec18 = ConvModule(2 * feat_channel, 1 * feat_channel)

        # Final layer conv 1x1
        self.final = nn.Conv2d(feat_channel, in_channels, kernel_size=1)

        # Maxpool 2x2 downsample
        self.down1011 = nn.MaxPool2d(2)
        self.down1112 = nn.MaxPool2d(2)
        self.down1213 = nn.MaxPool2d(2)
        self.down1314 = nn.MaxPool2d(2)

        # Up-conv 2x2
        # # Code refactored from https://aswali.github.io/WNet/
        self.up1415 = nn.ConvTranspose2d(16 * feat_channel, 8 * feat_channel, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up1516 = nn.ConvTranspose2d(8 * feat_channel, 4 * feat_channel, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up1617 = nn.ConvTranspose2d(4 * feat_channel, 2 * feat_channel, kernel_size=3, stride=2, padding=1,output_padding=1)
        self.up1718 = nn.ConvTranspose2d(2 * feat_channel, feat_channel, kernel_size=3, stride=2, padding=1, output_padding=1)


    def forward(self, x):
        # Going down the U
        dec10 = self.dec10(x)
        dec11 = self.dec11(self.down1011(dec10))
        dec12 = self.dec12(self.down1112(dec11))
        dec13 = self.dec13(self.down1213(dec12))
        dec14 = self.dec14(self.down1314(dec13))

        # Going up the U
        dec15 = self.dec15(torch.cat([dec13, self.up1415(dec14)], dim=1))
        dec16 = self.dec16(torch.cat([dec12, self.up1516(dec15)], dim=1))
        dec17 = self.dec17(torch.cat([dec11, self.up1617(dec16)], dim=1))
        dec18 = self.dec18(torch.cat([dec10, self.up1718(dec17)], dim=1))

        output = self.final(dec18)

        return output

