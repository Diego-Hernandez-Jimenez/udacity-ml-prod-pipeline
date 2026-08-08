"""
Training stage: fit the active model from ml_config.yaml on encoded training data.

Author: Diego Hernández Jiménez
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import load_config


def _build_model(cfg: dict) -> RandomForestClassifier:
    active = cfg["models"]["active"]
    hyperparams = cfg["models"][active]
    if active == "random_forest":
        return RandomForestClassifier(**hyperparams)
    raise ValueError(f"Unknown model: {active}")


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    save: bool = False,
) -> RandomForestClassifier:
    cfg = load_config()
    model = _build_model(cfg)
    model.fit(X_train, y_train)

    if save:
        model_dir = cfg["paths"]["model_dir"]
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, cfg["filenames"]["model"]))

    return model


if __name__ == "__main__":
    from encode_data import encode

    cfg = load_config()
    splits_dir = cfg["paths"]["splits_dir"]
    model_dir = cfg["paths"]["model_dir"]
    fn = cfg["filenames"]

    train_df = pd.read_csv(os.path.join(splits_dir, fn["train"]))
    test_df = pd.read_csv(os.path.join(splits_dir, fn["test"]))
    encoder = joblib.load(os.path.join(model_dir, fn["encoder"]))
    lb = joblib.load(os.path.join(model_dir, fn["lb"]))

    X_train, y_train, *_ = encode(train_df, test_df, encoder=encoder, lb=lb)

    train(X_train=X_train, y_train=y_train, save=True)
