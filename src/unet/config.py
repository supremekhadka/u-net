from .train import load_model
from .dataset import BriscDataset
from .transforms import BriscTransform
from torch.utils.data import DataLoader

def build_dataloaders(config):
    train_transforms = BriscTransform(
            size=config["dataset"]["size"],
            mean=config["dataset"]["mean"],
            std=config["dataset"]["std"],
            train=True
        )

    val_transforms = BriscTransform(
            size=config["dataset"]["size"],
            mean=config["dataset"]["mean"],
            std=config["dataset"]["std"],
            train=False
        )

    train_dataset = BriscDataset(
            images_root=config["dataset"]["images_root"],
            csv=config["dataset"]["csv"],
            split="train",
            transforms=train_transforms
        )

    val_dataset = BriscDataset(
            images_root=config['dataset']["images_root"],
            csv=config['dataset']["csv"],
            split="val",
            transforms=val_transforms
        )

    train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=config["training"]["batch_size"]
        )

    val_dataloader = DataLoader(
            val_dataset, 
            batch_size=config["training"]["batch_size"]
        )

    return train_dataloader, val_dataloader


def build_model(config):
    model = load_model(
        device=config["training"]["device"],
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        base_channels=config["model"]["base_channels"],
        depth=config["model"]["depth"]
    )

    return model