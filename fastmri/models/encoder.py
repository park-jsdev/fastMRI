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
        # into first conv layer
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


class Encoder(nn.Module):
    def __init__(self, feat_channel=64, in_channels=3, num_classes=5):
        super(Encoder, self).__init__()

        # Downwards in the U 1 to 4
        # 1st module has simple 3x3 conv
        self.enc1 = ConvModule(in_channels, feat_channel)

        # Subsequent modules have separable conv 3x3
        self.enc2 = SeparableConvModule(feat_channel, 2 * feat_channel)
        self.enc3 = SeparableConvModule(2 * feat_channel, 4 * feat_channel)
        self.enc4 = SeparableConvModule(4 * feat_channel, 8 * feat_channel)
        self.enc5 = SeparableConvModule(8 * feat_channel, 16 * feat_channel)

        # Upwards in the U 6 to 9
        self.enc6 = SeparableConvModule(16 * feat_channel, 8 * feat_channel)
        self.enc7 = SeparableConvModule(8 * feat_channel, 4 * feat_channel)
        self.enc8 = SeparableConvModule(4 * feat_channel, 2 * feat_channel)

        # Final module has simple 3x3 conv
        self.enc9 = ConvModule(2 * feat_channel, 1 * feat_channel)

        # Final layer conv 1x1
        self.final = nn.Conv2d(feat_channel, num_classes, kernel_size=(1, 1))

        # Maxpool 2x2 downsample
        self.down12 = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        self.down23 = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        self.down34 = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        self.down45 = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

        # Up-conv 2x2
        self.up56 = nn.ConvTranspose2d(16 * feat_channel, 8 * feat_channel, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up67 = nn.ConvTranspose2d(8 * feat_channel, 4 * feat_channel, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up78 = nn.ConvTranspose2d(4 * feat_channel, 2 * feat_channel, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up89 = nn.ConvTranspose2d(2 * feat_channel, feat_channel, kernel_size=3, stride=2, padding=1, output_padding=1)


    def forward(self, x):
        # Going down the U
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.down12(enc1))
        enc3 = self.enc3(self.down23(enc2))
        enc4 = self.enc4(self.down34(enc3))
        enc5 = self.enc5(self.down45(enc4))

        # Going up the U
        enc6 = self.enc6(torch.cat([enc4, self.up56(enc5)], dim=1))
        enc7 = self.enc7(torch.cat([enc3, self.up67(enc6)], dim=1))
        enc8 = self.enc8(torch.cat([enc2, self.up78(enc7)], dim=1))
        enc9 = self.enc9(torch.cat([enc1, self.up89(enc8)], dim=1))

        out = self.final(enc9)

        # print(f'enc4: {enc4.shape}, up56(enc5): {self.up56(enc5).shape}')
        # print(f'enc3: {enc3.shape}, up67(enc6): {self.up67(enc6).shape}')
        # print(f'enc2: {enc2.shape}, up78(enc7): {self.up78(enc7).shape}')
        # print(f'enc1: {enc1.shape}, up89(enc8): {self.up89(enc8).shape}')
        #
        # print(f'final: {self.final(enc9).shape}')

        return out
