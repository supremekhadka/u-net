from unet import split

from pathlib import Path
import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--train", required=True, type=float, help="Train ratio, [0,1]")
    parser.add_argument("--val", required=True, type=float, help="Val ratio, [0,1]")
    parser.add_argument("--test", required=True, type=float, help="Test ratio, [0,1]")
    parser.add_argument("--separate_csvs", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-filename", default="metadata.csv")
    parser.add_argument("--seed", default=42)

    args = parser.parse_args()

    train_df, val_df, test_df = split(args.csv, args.train, args.val, args.test, args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.separate_csvs:
        train_df.to_csv(Path(args.output_dir) / "train.csv", index=False)
        val_df.to_csv(Path(args.output_dir) / "val.csv", index=False)
        test_df.to_csv(Path(args.output_dir) / "test.csv", index=False)

        print(f"Split csvs saved to {args.output_dir}")
    else:
        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"

        combined_df = pd.concat([train_df, val_df, test_df], axis=0)
        combined_df.to_csv(Path(args.output_dir) / args.output_filename, index=False)
        print(f"Split csv saved as {args.output_dir}/{args.output_filename}")

if __name__ == "__main__":
    main()