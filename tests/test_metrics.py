import torch

from unet.metrics import segmentation_metrics


def test_segmentation_metrics_for_perfect_predictions():
    logits = torch.tensor(
        [
            [
                [[10.0, -10.0], [-10.0, 10.0]],
                [[-10.0, 10.0], [10.0, -10.0]],
            ]
        ]
    )
    labels = torch.tensor([[[0, 1], [1, 0]]])

    assert segmentation_metrics(logits, labels) == {
        "dice": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "iou": 1.0,
        "specificity": 1.0,
        "accuracy": 1.0,
    }


def test_segmentation_metrics_for_completely_wrong_predictions():
    logits = torch.tensor(
        [
            [
                [[-10.0, 10.0], [10.0, -10.0]],
                [[10.0, -10.0], [-10.0, 10.0]],
            ]
        ]
    )
    labels = torch.tensor([[[0, 1], [1, 0]]])

    metrics = segmentation_metrics(logits, labels)

    assert metrics["dice"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["iou"] == 0.0
    assert metrics["specificity"] == 0.0
    assert metrics["accuracy"] == 0.0
