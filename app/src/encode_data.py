"""
Encoding stage: fit OneHotEncoder and LabelBinarizer on the train split,
transform both splits, and optionally persist all artifacts to disk.

Pass pre-fitted encoder/lb to reuse existing artifacts (e.g. inference pipeline).

Author: Diego Hernández Jiménez
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from utils import load_config


def encode(
    train: pd.DataFrame,
    test: pd.DataFrame,
    encoder: OneHotEncoder | None = None,
    lb: LabelBinarizer | None = None,
    save: bool = False,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, OneHotEncoder, LabelBinarizer
]:
    cfg = load_config()
    cat_features = cfg["data"]["categorical_features"]
    label = cfg["data"]["label"]

    cat_train = train[cat_features]
    num_train = train.drop(columns=cat_features + [label]).values
    cat_test = test[cat_features]
    num_test = test.drop(columns=cat_features + [label]).values

    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit(cat_train)

    if lb is None:
        lb = LabelBinarizer()
        lb.fit(train[label].values)

    X_train = np.concatenate([num_train, encoder.transform(cat_train)], axis=1)
    X_test = np.concatenate([num_test, encoder.transform(cat_test)], axis=1)
    y_train = lb.transform(train[label].values).ravel()
    y_test = lb.transform(test[label].values).ravel()

    if save:
        model_dir = cfg["paths"]["model_dir"]
        fn = cfg["filenames"]
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(encoder, os.path.join(model_dir, fn["encoder"]))
        joblib.dump(lb, os.path.join(model_dir, fn["lb"]))

    return X_train, y_train, X_test, y_test, encoder, lb


if __name__ == "__main__":
    cfg = load_config()
    splits_dir = cfg["paths"]["splits_dir"]
    fn = cfg["filenames"]

    train = pd.read_csv(os.path.join(splits_dir, fn["train"]))
    test = pd.read_csv(os.path.join(splits_dir, fn["test"]))

    encode(train=train, test=test, save=True)
