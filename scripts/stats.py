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

    mean_sum = 0
    std_sum = 0

    for path in paths:
        image_path = Path(args.images_root) / Path(path.replace("\\", "/"))

        image = decode_image(image_path)
        image = v2.Compose(       
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Grayscale()
            ]
        )(image)

        tissue_pixels = image[image>0]

        image_mean = tissue_pixels.mean().item()
        image_std = tissue_pixels.std().item()

        mean_sum += image_mean
        std_sum += image_std

    mean = mean_sum / len(paths)
    std = std_sum / len(paths)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(Path(args.output_dir) / (args.output_filename + ".csv"), "w") as csv:
        csv.write("mean,std\n")
        csv.write(f"{round(mean, 3)},{round(std, 3)}")

    print(f"Average mean and std written to {args.output_dir}/{args.output_filename}.csv")
    
if __name__ == "__main__":
    main()