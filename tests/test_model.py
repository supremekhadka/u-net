import torch
from unet import UNet

class TestModel:
    def setup_method(self):
        self.depth = 4
        self.in_channels = 3
        self.out_channels = 10
        self.model = UNet(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            depth=self.depth
        )

    def test_model_construction(self):
        assert self.model is not None

    def test_stage_counts(self):
        assert len(self.model.encoder) == self.depth
        assert len(self.model.decoder) == self.depth

    def test_forward_shape(self):
        x = torch.rand((1, 3, 572, 572))

        with torch.no_grad():
            output = self.model(x)

        assert output.shape[-2:] == (388, 388)