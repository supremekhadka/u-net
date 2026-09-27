import argparse
import os
import torch
import pandas as pd
from pathlib import Path
from torchvision.io import decode_image
from torchvision.transforms import v2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--images-root")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir")
    parser.add_argument("--output-filename", default="mean_std")

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["split"] == args.split]
    paths = df["image_path"].to_list()

    sum_ = 0.0
    sum_sq = 0.0
    count = 0

    for path in paths:
        image_path = Path(args.images_root) / Path(path.replace("\\", "/"))
        image = decode_image(image_path)
        image = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Grayscale()])(image)

        tissue_pixels = image[image > 0]

        sum_ += tissue_pixels.sum().item()
        sum_sq += (tissue_pixels ** 2).sum().item()
        count += tissue_pixels.numel()

    mean = sum_ / count
    std = (sum_sq / count - mean ** 2) ** 0.5

    os.makedirs(args.output_dir, exist_ok=True)
    with open(Path(args.output_dir) / (args.output_filename + ".csv"), "w") as csv:
        csv.write("mean,std\n")
        csv.write(f"{round(mean, 3)},{round(std, 3)}")

    print(f"Average mean and std written to {args.output_dir}/{args.output_filename}.csv")
    
if __name__ == "__main__":
    main()