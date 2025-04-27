from torch import nn
from encoder import Encoder
from decoder import Decoder


class Wnet(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes = 5, kernel_size=3, stride=1, padding=0):
        super(Wnet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_classes = num_classes

        self.encoder = Encoder(in_channels=self.in_channels, num_classes=self.num_classes)
        self.decoder = Decoder(in_channels=self.in_channels, num_classes=self.num_classes)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded
