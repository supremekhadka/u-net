import argparse
import yaml
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from unet import run_training, load_model, BriscDataset, BriscTransform
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_transforms = BriscTransform(
            size=572,
            mean=0.195,
            std=0.168,
            train=True
        )

    val_transforms = BriscTransform(
            size=572,
            mean=0.195,
            std=0.168,
            train=False
        )

    train_dataset = BriscDataset(
            images_root=config.images_root,
            csv=config.csv,
            split="train",
            transforms=train_transforms
        )

    val_dataset = BriscDataset(
            images_root=config.images_root,
            csv=config.csv,
            split="val",
            transforms=val_transforms
        )

    train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=config.batch_size
        )

    val_dataloader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size
        )

    model = load_model(
        device=config.device,
        in_channels=1,
        out_channels=4,
        base_channels=64,
        depth=4
    )

    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.lr)

    run_training(
        model=model,
        device=config.device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=config.epochs,
        output_dir=config.output_dir
    )

if __name__ == "__main__":
    main()