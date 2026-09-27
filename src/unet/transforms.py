from torchvision.transforms import v2

train_transforms = v2.Compose(
    v2.Resize(572),
    v2.CenterCrop(572),
    v2.RandomRotation(15),
    v2.RandomVerticalFlip(),
    v2.RandomHorizontalFlip()
)

test_transforms = v2.Compose(
    v2.Resize(572),
    v2.CenterCrop(572)
)