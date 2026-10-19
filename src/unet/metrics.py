from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_f1_score,
    multiclass_jaccard_index,
    multiclass_precision,
    multiclass_recall,
    multiclass_specificity,
)
from torchmetrics.functional.segmentation import dice_score


def segmentation_metrics(outputs, labels):
    num_classes = outputs.shape[1]
    predictions = outputs.argmax(dim=1)

    metric_kwargs = {
        "num_classes": num_classes,
        "average": "macro",
        "zero_division": 0,
    }
    specificity_kwargs = {
        "num_classes": num_classes,
        "average": "macro",
    }
    return {
        "dice": dice_score(
            predictions,
            labels,
            num_classes=num_classes,
            input_format="index",
            average="macro",
            aggregation_level="global",
        ).item(),
        "precision": multiclass_precision(predictions, labels, **metric_kwargs).item(),
        "recall": multiclass_recall(predictions, labels, **metric_kwargs).item(),
        "f1": multiclass_f1_score(predictions, labels, **metric_kwargs).item(),
        "iou": multiclass_jaccard_index(
            predictions, labels, **metric_kwargs
        ).item(),
        "accuracy": multiclass_accuracy(
            predictions,
            labels,
            num_classes=num_classes,
            average="micro",
            multidim_average="global",
        ).item(),
        "specificity": multiclass_specificity(
            predictions, labels, **specificity_kwargs
        ).item(),
    }