import torch
from torchvision.transforms import v2
from torchvision.tv_tensors import Mask

brisc_mean, brisc_std = 0.195, 0.168

class BriscTransform:
    def __init__(self, size=572, mean=0.195, std=0.168, train=True):
        self.size = size
        self.mean = mean
        self.std = std
        self.train = train

        self.image_only = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),        
            v2.Grayscale(),
            v2.Normalize([self.mean], [self.std]),
        ])

        self.spatial = [
            v2.Resize(self.size),
            v2.CenterCrop(self.size),
        ]

        if train:
            self.spatial += [
                v2.RandomRotation(15),
                v2.RandomVerticalFlip(),
                v2.RandomHorizontalFlip()
            ]

        self.spatial = v2.Compose(self.spatial)

    def __call__(self, image, mask):
        image = self.image_only(image)
        mask = Mask(mask)

        image, mask = self.spatial(image, mask)

        return image, mask.long()