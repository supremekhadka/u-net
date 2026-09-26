import torch
import pytest
from unet import UNet

@pytest.mark.parametrize(
    "depth, in_channels, out_channels, input_size, expected_output_size",
    [
        (4, 3, 10, 572, 388),
        (5, 3, 10, 1148, 772)
    ]
)

def test_forward_shape(depth, in_channels, out_channels, input_size, expected_output_size):
    model = UNet(in_channels=in_channels, out_channels=out_channels, depth=depth)
    x = torch.rand((1, in_channels, input_size, input_size))
    with torch.no_grad():
        output = model(x)
    assert output.shape[-2:] == (expected_output_size, expected_output_size)

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