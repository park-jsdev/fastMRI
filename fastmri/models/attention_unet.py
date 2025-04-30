import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.models import Unet
import pdb

class AttentionGate(nn.Module):
    def __init__(self, in_chans: int, gating_chans: int):
        super().__init__()
        # inspired by https://github.com/ozan-oktay/Attention-Gated-Networks
        # project encoder features into gating space
        self.W_x = nn.Conv2d(in_chans, gating_chans, kernel_size=1, bias=False)
        # project decoder features into gating space
        self.W_g = nn.Conv2d(gating_chans, gating_chans, kernel_size=1, bias=False)

        self.psi = nn.Sequential(
            nn.Conv2d(gating_chans, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # attn = psi(relu(W_x*x + W_g*g))
        x_proj = self.W_x(x)
        g_proj = self.W_g(g)
        f       = self.relu(x_proj + g_proj)
        attn    = self.psi(f)
        return x * attn


# Integrating Attention Gate into U-Net
class AttentionUnet(Unet):
    def __init__(self, in_chans: int, out_chans: int, chans: int = 32, num_pool_layers: int = 4, drop_prob: float = 0.0):
        super().__init__(in_chans, out_chans, chans, num_pool_layers, drop_prob)
        self.attn_gates = nn.ModuleList()
        ch = chans * (2 ** (num_pool_layers - 1))
        for _ in range(num_pool_layers):
            self.attn_gates.append(AttentionGate(in_chans=ch, gating_chans=ch))
            ch //= 2


    def forward(self, image: torch.Tensor) -> torch.Tensor:
        stack = []
        output = image

        # Down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # Up-sampling layers with Attention Gates
        for i, (transpose_conv, conv) in enumerate(zip(self.up_transpose_conv, self.up_conv)):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # Handle padding
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            # Apply Attention Gate before concatenation
            gated_downsample_layer = self.attn_gates[i](downsample_layer, output)
            # Concatenate the modulated output with the skip connection
            output = torch.cat([output, gated_downsample_layer], dim=1)
            output = conv(output)

        return output
