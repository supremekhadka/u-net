import argparse
import yaml
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from unet import run_training, build_dataloaders, build_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_dataloader, val_dataloader = build_dataloaders(config)
    model = build_model(config)

    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config["lr"])

    run_training(
        model=model,
        device=config["training"]["device"],
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=config["training"]["epochs"],
        output_dir=config["training"]["output_dir"]
    )

if __name__ == "__main__":
    main()