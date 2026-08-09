"""
Compute and report model performance on slices of each categorical feature.

Run from the app/ directory:
    python src/evaluate_performance.py

Output is printed to stdout and written to slice_output.txt.

Author: Diego Hernández Jiménez
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from utils import load_config


def compute_slice_metrics(
    df: pd.DataFrame,
    feature: str,
    model,
    encoder,
    lb,
    cfg: dict,
) -> pd.DataFrame:
    """Return a DataFrame with precision/recall/f1
    for each unique value of *feature*."""
    cat_features = cfg["data"]["categorical_features"]
    label = cfg["data"]["label"]
    rows = []

    for value in sorted(df[feature].unique()):
        slice_df = df[df[feature] == value]
        cat = slice_df[cat_features]
        num = slice_df.drop(columns=cat_features + [label]).values
        X = np.concatenate([num, encoder.transform(cat)], axis=1)
        y_true = lb.transform(slice_df[label].values).ravel()
        y_pred = model.predict(X)

        rows.append(
            {
                "feature": feature,
                "value": value,
                "n": len(slice_df),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": fbeta_score(y_true, y_pred, beta=1, zero_division=0),
            }
        )

    return pd.DataFrame(rows)


def run_slice_metrics(output_path: str = "slice_output.txt") -> pd.DataFrame:
    cfg = load_config()
    fn = cfg["filenames"]
    splits_dir = cfg["paths"]["splits_dir"]
    model_dir = cfg["paths"]["model_dir"]
    cat_features = cfg["data"]["categorical_features"]

    test_df = pd.read_csv(os.path.join(splits_dir, fn["test"]))
    model = joblib.load(os.path.join(model_dir, fn["model"]))
    encoder = joblib.load(os.path.join(model_dir, fn["encoder"]))
    lb = joblib.load(os.path.join(model_dir, fn["lb"]))

    all_metrics = pd.concat(
        [
            compute_slice_metrics(test_df, feat, model, encoder, lb, cfg)
            for feat in cat_features
        ],
        ignore_index=True,
    )

    lines = []
    for feat in cat_features:
        lines.append(f"\n{'=' * 60}")
        lines.append(f"Feature: {feat}")
        lines.append(f"{'=' * 60}")
        for _, row in all_metrics[all_metrics["feature"] == feat].iterrows():
            lines.append(
                f"  {row['value']!s:<35} n={int(row['n']):<6} "
                f"precision={row['precision']:.3f}  "
                f"recall={row['recall']:.3f}  "
                f"f1={row['f1']:.3f}"
            )

    output = "\n".join(lines)
    print(output)

    with open(output_path, "w") as f:
        f.write(output + "\n")

    return all_metrics


if __name__ == "__main__":
    run_slice_metrics()
