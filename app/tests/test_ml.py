"""
Unit tests for the ML pipeline functions: preprocess, encode, train.

Author: Diego Hernández Jiménez
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import encode_data
import preprocess_data
import train_model

TEST_CONFIG = {
    "paths": {
        "raw_data": "data/census.csv",
        "cleaned_data": "data/cleaned_census.csv",
        "model_dir": "model",
    },
    "filenames": {"model": "model.pkl", "encoder": "encoder.pkl", "lb": "lb.pkl"},
    "data": {
        "label": "salary",
        "categorical_features": [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ],
    },
    "models": {
        "active": "random_forest",
        "random_forest": {
            "n_estimators": 10,
            "max_depth": 3,
            "random_state": 42,
            "n_jobs": 1,
        },
    },
}

_CAT_VALUES = {
    "workclass": ["Private", "Self-emp"],
    "education": ["Bachelors", "HS-grad"],
    "marital-status": ["Married-civ-spouse", "Never-married"],
    "occupation": ["Tech-support", "Sales"],
    "relationship": ["Husband", "Not-in-family"],
    "race": ["White", "Black"],
    "sex": ["Male", "Female"],
    "native-country": ["United-States", "Mexico"],
}


def _make_df(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "age": rng.integers(20, 70, n),
            "workclass": rng.choice(_CAT_VALUES["workclass"], n),
            "fnlgt": rng.integers(10_000, 999_999, n),
            "education": rng.choice(_CAT_VALUES["education"], n),
            "education-num": rng.integers(1, 16, n),
            "marital-status": rng.choice(_CAT_VALUES["marital-status"], n),
            "occupation": rng.choice(_CAT_VALUES["occupation"], n),
            "relationship": rng.choice(_CAT_VALUES["relationship"], n),
            "race": rng.choice(_CAT_VALUES["race"], n),
            "sex": rng.choice(_CAT_VALUES["sex"], n),
            "capital-gain": rng.integers(0, 5_000, n),
            "capital-loss": rng.integers(0, 1_000, n),
            "hours-per-week": rng.integers(20, 60, n),
            "native-country": rng.choice(_CAT_VALUES["native-country"], n),
            "salary": rng.choice(["<=50K", ">50K"], n),
        }
    )


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_csv_with_missing(tmp_path) -> str:
    """Mini CSV with one clean row and one row containing '?' values."""
    content = (
        "age,workclass,fnlgt,salary\n"
        "25,Private,77516,<=50K\n"
        "40,?,209642,>50K\n"
        "52,Local-gov,338409,>50K\n"
    )
    p = tmp_path / "census_mini.csv"
    p.write_text(content)
    return str(p)


def test_preprocess_drops_missing_values(raw_csv_with_missing):
    cfg = {
        **TEST_CONFIG,
        "paths": {**TEST_CONFIG["paths"], "raw_data": raw_csv_with_missing},
    }
    with patch("preprocess_data.load_config", return_value=cfg):
        result = preprocess_data.preprocess(save=False)
    assert result.isna().sum().sum() == 0, "Result must have no NaN values"
    assert len(result) == 2, "Row with '?' must be removed"


def test_preprocess_returns_dataframe(raw_csv_with_missing):
    cfg = {
        **TEST_CONFIG,
        "paths": {**TEST_CONFIG["paths"], "raw_data": raw_csv_with_missing},
    }
    with patch("preprocess_data.load_config", return_value=cfg):
        result = preprocess_data.preprocess(save=False)
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------


@patch("encode_data.load_config", return_value=TEST_CONFIG)
def test_encode_output_shapes(mock_cfg):
    train_df = _make_df(40)
    test_df = _make_df(10)
    X_train, y_train, X_test, y_test, *_ = encode_data.encode(train_df, test_df)

    assert X_train.shape[0] == 40
    assert X_test.shape[0] == 10
    assert X_train.shape[1] == X_test.shape[1], (
        "Train and test must have the same number of features"
    )
    assert y_train.shape == (40,)
    assert y_test.shape == (10,)


@patch("encode_data.load_config", return_value=TEST_CONFIG)
def test_encode_binary_labels(mock_cfg):
    train_df = _make_df(30)
    test_df = _make_df(10)
    _, y_train, _, y_test, *_ = encode_data.encode(train_df, test_df)

    assert set(np.unique(y_train)).issubset({0, 1}), "Labels must be binary (0/1)"
    assert set(np.unique(y_test)).issubset({0, 1})


@patch("encode_data.load_config", return_value=TEST_CONFIG)
def test_encode_reuses_fitted_encoder(mock_cfg):
    """Passing pre-fitted encoder/lb should skip re-fitting and return the same objects."""
    train_df = _make_df(30)
    test_df = _make_df(10)
    cat_features = TEST_CONFIG["data"]["categorical_features"]

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(train_df[cat_features])
    lb = LabelBinarizer()
    lb.fit(train_df["salary"])

    _, _, _, _, returned_enc, returned_lb = encode_data.encode(
        train_df, test_df, encoder=encoder, lb=lb
    )

    assert returned_enc is encoder
    assert returned_lb is lb


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@patch("train_model.load_config", return_value=TEST_CONFIG)
def test_train_returns_fitted_classifier(mock_cfg):
    rng = np.random.default_rng(1)
    X = rng.random((50, 15))
    y = rng.integers(0, 2, 50)
    model = train_model.train(X, y, save=False)

    assert isinstance(model, RandomForestClassifier)
    preds = model.predict(X)
    assert preds.shape == (50,)
    assert set(preds).issubset({0, 1})


@patch("train_model.load_config", return_value=TEST_CONFIG)
def test_train_model_respects_hyperparams(mock_cfg):
    rng = np.random.default_rng(2)
    X = rng.random((30, 10))
    y = rng.integers(0, 2, 30)
    model = train_model.train(X, y, save=False)

    assert model.n_estimators == TEST_CONFIG["models"]["random_forest"]["n_estimators"]
    assert model.max_depth == TEST_CONFIG["models"]["random_forest"]["max_depth"]
    assert model.random_state == TEST_CONFIG["models"]["random_forest"]["random_state"]
