import torch

from unet.metrics import dice


def test_dice_returns_one_for_perfect_multiclass_predictions():
    logits = torch.tensor(
        [
            [
                [[10.0, -10.0], [-10.0, 10.0]],
                [[-10.0, 10.0], [10.0, -10.0]],
            ]
        ]
    )
    labels = torch.tensor([[[0, 1], [1, 0]]])

    assert dice(logits, labels) == 1.0


def test_dice_returns_zero_for_completely_wrong_predictions():
    logits = torch.tensor(
        [
            [
                [[-10.0, 10.0], [10.0, -10.0]],
                [[10.0, -10.0], [-10.0, 10.0]],
            ]
        ]
    )
    labels = torch.tensor([[[0, 1], [1, 0]]])

    assert dice(logits, labels) == 0.0
