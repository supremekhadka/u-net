import torch
import torch.nn as nn
from .blocks import DoubleConv, DownSample, UpSample

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, depth=4):
        super().__init__()
        self.depth = depth
        self.encoder_channels = [in_channels] + [64*(2**i) for i in range(depth)]
        self.decoder_channels = [64*(2**i) for i in range(depth+1)][::-1]

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

    def _crop_feature_map(self, feature_map, x):
        '''
        Center crops feature map to the shape of x in (h, w) dimensions.
        '''
        diff_h = (feature_map.shape[-2] - x.shape[-2])/2 
        diff_w = (feature_map.shape[-1] - x.shape[-1])/2 

        assert diff_h.is_integer(), "Height difference is not a whole number."
        assert diff_w.is_integer(), "Width difference is not a whole number."

        cropped = feature_map[:, :, int(diff_h) : int(x.shape[-2] + diff_h), int(diff_w) : int(x.shape[-1] + diff_w)]

        return cropped

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
                    feature_map_cropped = self._crop_feature_map(feature_maps[self.depth - level - 1], x)
                    x = torch.cat((feature_map_cropped, x), dim=-3)

        x = self.conv(x)

        return x