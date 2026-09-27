import pandas as pd
import torch
from .utils import crop_feature_map
from pathlib import Path
from torchvision.transforms import v2
from torchvision.io import decode_image
from torch.utils.data import Dataset

class BriscDataset(Dataset):
    def __init__(self, images_root, csv, split, transforms=None):
        super().__init__()
        self.images_root = images_root
        self.csv = csv
        self.df = pd.read_csv(csv)
        self.df = self.df[self.df["split"] == split]
        self.transforms = transforms

    @staticmethod
    def _combine_mask_with_label(mask, label):
        binary = mask >= 128

        target = torch.zeros_like(binary, dtype=torch.int)
        target[binary] = label
        target.squeeze_()

        return target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = Path(self.images_root) / Path(self.df.iloc[index]["image_path"].replace('\\', '/'))
        mask_path = Path(self.images_root) / Path(self.df.iloc[index]["mask_path"].replace('\\', '/'))
        label = int(self.df.iloc[index]["label"]) 

        image = decode_image(image_path)
        mask = decode_image(mask_path)
        mask = self._combine_mask_with_label(mask, label)

        if self.transforms:
            image, mask = self.transforms(image, mask)

        mask = crop_feature_map(mask, torch.zeros((388,388)))
        return image, mask
