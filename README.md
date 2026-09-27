# u-net
PyTorch implementation of the U-Net architecture for BRISC 2025 image segmentation.

## Requirements

- Python 3.12+
- The [BRISC 2025](https://www.kaggle.com/datasets/briscdataset/brisc2025) dataset
- A CPU or CUDA-enabled PyTorch installation

## Installation

Create and activate a Python environment, then install the package:

```bash
pip install -e .
```

## Usage

### 1. Split the dataset

Create train, validation, and test splits from the dataset metadata CSV:

```bash
python -m scripts.split \
    --csv <csv_path> \
    --train <train_ratio> \
    --val <val_ratio> \
    --test <test_ratio> \
    --output-dir <output_dir> \
    [--output-filename <output_csv_filename>] \
    [--seed <seed_int>] \
    [--separate_csvs]
```

The ratios must add up to `1`. By default, one combined CSV is written. Use
`--separate_csvs` to write `train.csv`, `val.csv`, and `test.csv` instead.

### 2. Configure the dataset

Update `configs/config.yaml` with the paths to the dataset and split CSV, then
choose the model and training settings.

### 3. Train

```bash
python train.py --config configs/config.yaml
```

Model checkpoints are written to the configured `training.output_dir`.

### Tests

```bash
pytest
```

Prediction and inference scripts are not included yet.
