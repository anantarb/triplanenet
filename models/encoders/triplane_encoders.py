import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Flatten, Sequential, Module

from models.encoders.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.PReLU(out_channels),
            nn.PReLU(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super().__init__()

        self.up = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TriPlane_Encoder(Module):

    def __init__(self):
        super(TriPlane_Encoder, self).__init__()

        blocks = get_blocks(num_layers=50)
        unit_module = bottleneck_IR_SE

        self.input_layer = Sequential(Conv2d(6, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.up1 = (Up(1024, 512, upscale_factor=1))
        self.up2 = (Up(384, 384))
        self.up3 = (Up(224, 256))
        self.up4 = (Up(128, 96))
        self.head = nn.PixelShuffle(upscale_factor=2)
        self.final_head = nn.Sequential(
            nn.Conv2d(24, 96, kernel_size=3, padding=1),
            nn.PReLU(96),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.PReLU(96),
            nn.Conv2d(96, 96, kernel_size=1)
        )

    def forward(self, x):
        x = self.input_layer(x)
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 2:
                c0 = x
            if i == 6:
                c1 = x
            if i == 20:
                c2 = x
            elif i == 21:
                c3 = x

        tri_plane = self.up1(x, c3)
        tri_plane = self.up2(tri_plane, c2)
        tri_plane = self.up3(tri_plane, c1)
        tri_plane = self.up4(tri_plane, c0)
        tri_plane = self.head(tri_plane)
        tri_plane = self.final_head(tri_plane)
        return tri_plane