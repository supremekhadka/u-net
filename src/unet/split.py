import pandas as pd
from sklearn.model_selection import train_test_split

def split(csv, train=0.6, val=0.2, test=0.2, seed=42):
    assert abs(train + val + test - 1) < 1e-6, "Splits must sum up to 1.0."

    if not isinstance(csv, pd.DataFrame):
        df = pd.read_csv(csv)
    else:
        df = csv

    train_df, val_plus_test_df = train_test_split(
        df, 
        test_size=val+test,
        train_size=train,
        stratify=df["label"],
        random_state=seed
    )

    val_df, test_df = train_test_split(
        val_plus_test_df,
        test_size=test / (test + val),
        train_size=val / (test + val),
        stratify=val_plus_test_df["label"],
        random_state=seed
    )

    return train_df, val_df, test_df