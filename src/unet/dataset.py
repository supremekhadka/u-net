import pandas as pd
from torch.utils.data import Dataset

class BricsDataset(Dataset):
    def __init__(self, images_path, csv, transforms=None, target_transforms=None):
        super().__init__()
        self.images_path = images_path
        self.csv = csv
        self.df = pd.read_csv(csv)
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = self.df.iloc[index]["image_path"]
        label = self.df.iloc[index]["label"]

        if self.transforms:
            image = self.transforms(image)
        if self.target_transforms:
            label = self.transforms(label)

        return image, label
