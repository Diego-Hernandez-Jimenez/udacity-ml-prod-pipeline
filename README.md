# ML Production Pipeline — Census Income Classifier

A production-ready ML pipeline that predicts whether an individual's annual income exceeds $50K based on U.S. Census data. The project covers data versioning with DVC, model training, a FastAPI inference service, and CI/CD via GitHub Actions for deployment on Render.

---

## Project structure

```
udacity-ml-prod-pipeline/
├── dvc.yaml                    # 4-stage DVC pipeline
├── .github/workflows/          # CI: flake8 + pytest on push to master
└── app/
    ├── main.py                 # FastAPI app (GET / and POST /predict)
    ├── ml_config.yaml          # Central config: paths, features, hyperparams
    ├── pyproject.toml          # Dependencies (managed with uv)
    ├── model_card_template.md  # Filled-in model card
    ├── data/
    │   ├── census.csv          # Raw UCI Adult dataset
    │   ├── cleaned_census.csv  # After preprocessing
    │   └── splits/             # train.csv / test.csv
    ├── model/                  # Artifacts: model.pkl, encoder.pkl, lb.pkl
    ├── src/
    │   ├── preprocess_data.py  # Stage 1: drop missing values
    │   ├── split_data.py       # Stage 2: stratified 80/20 split
    │   ├── encode_data.py      # Stage 3: OHE + LabelBinarizer
    │   ├── train_model.py      # Stage 4: fit RandomForestClassifier
    │   ├── slice_metrics.py    # Per-slice performance report
    │   └── utils.py            # load_config() helper
    └── tests/
        ├── test_api.py         # 3 API tests (GET + 2x POST)
        └── test_ml.py          # 7 ML unit tests
```

---

## Environment setup

Requires Python 3.13. [uv](https://github.com/astral-sh/uv) is recommended.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (from app/)
cd app
uv sync
```

All commands below that run Python should be executed from the `app/` directory.

---

## Reproducing the training pipeline

The pipeline is defined in `dvc.yaml` and has four sequential stages:

| Stage | Script | Input | Output |
|---|---|---|---|
| `preprocess` | `src/preprocess_data.py` | `data/census.csv` | `data/cleaned_census.csv` |
| `split` | `src/split_data.py` | `data/cleaned_census.csv` | `data/splits/` |
| `encode` | `src/encode_data.py` | `data/splits/` | `model/encoder.pkl`, `model/lb.pkl` |
| `train` | `src/train_model.py` | `data/splits/`, encoder artifacts | `model/model.pkl` |

To run the full pipeline (only re-runs stages with changed dependencies):

```bash
# From the repo root
dvc repro
```

To force a full re-run from scratch:

```bash
dvc repro --force
```

To run a single stage in isolation (from `app/`):

```bash
uv run python src/preprocess_data.py
uv run python src/split_data.py
uv run python src/encode_data.py
uv run python src/train_model.py
```

Hyperparameters and file paths are centralized in `app/ml_config.yaml`. Edit that file to change the model, split ratio, or directory layout.

---

## Running tests

All tests run from `app/` with pytest:

```bash
# Run all tests
uv run pytest tests -v

# Run only ML unit tests
uv run pytest tests/test_ml.py -v

# Run only API tests
uv run pytest tests/test_api.py -v
```

**ML unit tests** (`tests/test_ml.py`) — 7 tests covering `preprocess`, `encode`, and `train`. All functions are tested with mocked config so no artifacts need to be present on disk.

**API tests** (`tests/test_api.py`) — 3 tests using FastAPI's `TestClient`: one for `GET /` and one for each prediction class on `POST /predict`. These tests require the trained model artifacts (`model/*.pkl`) to be present.

### Linting

```bash
uv run flake8 src
```

---

## Slice performance report

To evaluate model performance broken down by each categorical feature:

```bash
# From app/
uv run python src/slice_metrics.py
```

This prints a report to stdout and writes it to `app/slice_output.txt`. Requires trained artifacts in `model/`.

---

## Running the API locally

```bash
# From app/
uv run uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

**Endpoints:**

- `GET /` — returns a welcome message
- `POST /predict` — accepts a Census record and returns `{"salary": "<=50K"}` or `{"salary": ">50K"}`

Example request body for `POST /predict`:

```json
{
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}
```

---

## CI/CD

GitHub Actions runs on every push to `master` (`.github/workflows/code-quaility-action.yml`):

1. Sets up Python 3.13 and installs dependencies with `uv sync`
2. Runs `flake8 src` for linting
3. Runs `pytest tests` for all unit and API tests

The app is deployed on [Render](https://render.com) with automatic deploys triggered when the CI pipeline passes on `master`.
