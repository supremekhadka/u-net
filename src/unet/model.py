import torch
import torch.nn as nn
from .utils import crop_feature_map
from .blocks import DoubleConv, DownSample, UpSample

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64, depth=4):
        super().__init__()
        self.depth = depth
        self.encoder_channels = [in_channels] + [base_channels*(2**i) for i in range(depth)]
        self.decoder_channels = [base_channels*(2**i) for i in range(depth+1)][::-1]

        self.encoder = nn.ModuleList(
            [nn.ModuleList(
                [DoubleConv(in_channels=self.encoder_channels[i], out_channels=self.encoder_channels[i+1], kernel_size=3, stride=1, padding=0),
                DownSample(kernel_size=2, stride=2, padding=0)]
            ) for i in range(depth)]
        )

        self.bottleneck = DoubleConv(in_channels=self.encoder_channels[-1], out_channels=self.decoder_channels[0], kernel_size=3, stride=1, padding=0)

        self.decoder = nn.ModuleList(
            [nn.ModuleList(
                [UpSample(in_channels=self.decoder_channels[i], out_channels=self.decoder_channels[i+1], kernel_size=2, stride=2, padding=0),
                DoubleConv(in_channels=self.decoder_channels[i], out_channels=self.decoder_channels[i+1], kernel_size=3, stride=1, padding=0)]
            ) for i in range(depth)]
        )
        
        self.conv = nn.Conv2d(in_channels=self.decoder_channels[-1], out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        feature_maps = dict([(index, None) for index in range(self.depth) ])

        for level in range(len(self.encoder)):
            for index in range(len(self.encoder[level])):
                x = self.encoder[level][index](x)
                if index == 0:
                    feature_maps[level] = x.clone()
                
        x = self.bottleneck(x)

        for level in range(len(self.decoder)):
            for index in range(len(self.decoder[level])):
                x = self.decoder[level][index](x)
                if index == 0:
                    feature_map_cropped = crop_feature_map(feature_maps[self.depth - level - 1], x)
                    x = torch.cat((feature_map_cropped, x), dim=-3)

        x = self.conv(x)

        return x