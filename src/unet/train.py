import os

import torch
from unet.model import UNet
from tqdm import tqdm
from unet.metrics import dice
from pathlib import Path

def load_model(device, in_channels, out_channels, base_channels=64, depth=4, checkpoint=None):
    model = UNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels, depth=depth)

    if checkpoint:
        model.load_state_dict(checkpoint)

    model = model.to(device)

    return model

def train_one_batch(model, images, masks, criterion, optimizer):
    output = model(images)
    loss = criterion(output, masks)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def train_one_epoch(epoch, model, device, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0

    print(f"Epoch {epoch+1}: ")
    for images, masks in tqdm(dataloader):
        images, masks = images.to(device), masks.to(device)
        running_loss += train_one_batch(model, images, masks, criterion, optimizer)

    average_loss = running_loss / len(dataloader)

    return average_loss

def validate(model, device, dataloader, criterion):
    model.eval()
    running_loss = 0

    outputs = []
    labels = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader):
            images, masks = images.to(device), masks.to(device)
            labels.append(masks)
            output = model(images)
            outputs.append(output)
            loss = criterion(output, masks)
            running_loss += loss.item()

        average_loss = running_loss / len(dataloader)
        dice_score = dice(torch.cat(outputs), torch.cat(labels))

    return average_loss, dice_score

def save_checkpoint(model, dir, name):
    os.makedirs(dir, exist_ok=True)
    checkpoint_path = Path(dir) / name
    torch.save(model.state_dict(), checkpoint_path)

def run_training(model, device, train_dataloader, val_dataloader, criterion, optimizer, epochs, output_dir):
    criterion = criterion.to(device)

    best_dice_score = -float('inf')

    for epoch in range(epochs):
        train_loss = train_one_epoch(epoch, model, device, train_dataloader, criterion, optimizer)
        _, train_dice_score = validate(model, device, train_dataloader, criterion)
        val_loss, val_dice_score = validate(model, device, val_dataloader, criterion)

        print(f"Train Loss: {train_loss} | Train Dice Score: {train_dice_score} | Val Loss: {val_loss} | Val Dice Score: {val_dice_score}")
        if val_dice_score > best_dice_score:
            best_dice_score = val_dice_score
            save_checkpoint(model, output_dir, "best.pt")
            
        save_checkpoint(model, output_dir, "last.pt")

    print(f"Training completed! Model checkpoints saved to {output_dir}")