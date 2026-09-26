# u-net
PyTorch implementation of the U-Net architecture 

## Introduction
This project re-implements the U-Net segmentation model. The [BRISC 2025](https://www.kaggle.com/datasets/briscdataset/brisc2025) dataset is used with custom splits to train it. 

## Installation

Inside a python environment, install unet as a package:

```
pip install -e .
```

## Instructions

### Split

Split the dataset using the split module in /scripts. The command is as follows:

```
python -m scripts.split \
    --csv <csv_path> \
    --train <train_ratio> \
    --val <val_ratio> \
    --test <test_ratio> \
    --output-dir \
    (--output-filename <output_csv_filename>) \
    (--seed <seed_int>) \
    (--separate-csvs) 
```

The arguments inside parantheses are optional.

### Train

### Predict
