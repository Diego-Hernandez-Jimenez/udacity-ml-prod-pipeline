"""
Split stage. Reads the cleaned dataset and produces stratified train/test splits,
preserving all columns (label included) so downstream stages work from a single file.

Author: Diego Hernández Jiménez
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split
from utils import load_config


def split_dataset(save: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    df = pd.read_csv(cfg["paths"]["cleaned_data"])
    train, test = train_test_split(
        df,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
    )

    if save:
        splits_dir = cfg["paths"]["splits_dir"]
        os.makedirs(splits_dir, exist_ok=True)
        train.to_csv(os.path.join(splits_dir, cfg["filenames"]["train"]), index=False)
        test.to_csv(os.path.join(splits_dir, cfg["filenames"]["test"]), index=False)

    return train, test


if __name__ == "__main__":
    split_dataset(save=True)
