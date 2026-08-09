"""
Preprocessing stage. Strips whitespace, replaces '?' with NaN, and drops
rows with missing values to produce a clean dataset for the training pipeline.

Author: Diego Hernández Jiménez
"""

import pandas as pd
from utils import load_config


def preprocess(save: bool = False) -> pd.DataFrame:
    cfg = load_config()
    df = pd.read_csv(cfg["paths"]["raw_data"], na_values="?", skipinitialspace=True)
    df = df.dropna()

    if save:
        df.to_csv(cfg["paths"]["cleaned_data"], index=False)

    return df


if __name__ == "__main__":
    preprocess(save=True)
