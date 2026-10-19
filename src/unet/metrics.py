from torchmetrics.functional.segmentation import dice_score


def dice(outputs, labels):
    num_classes = outputs.shape[1]
    predictions = outputs.argmax(dim=1)

    return dice_score(
        predictions,
        labels,
        num_classes=num_classes,
        input_format="index",
        average="macro",
        aggregation_level="global",
    ).item()