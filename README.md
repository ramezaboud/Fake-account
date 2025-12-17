# 🤖 Fake Account Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Fake account classifier using a Random Forest pipeline with custom feature engineering for text, user info, and activity-based features.

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Features](#-features)
- [Model Performance](#-model-performance)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

This project detects fake/bot accounts on social media platforms using machine learning. It analyzes user profiles based on:
- **Activity patterns** (tweets per day, account age)
- **Network metrics** (followers, friends, listed count)
- **Profile characteristics** (description length, default profile, verified status)
- **Demographic inference** (gender from name)
- **Ratio-based features** (followers/friends ratio, engagement ratio)
- **Suspicious patterns** (zero followers, high friend rate)

## 📁 Project Structure

```
fake-account/
├── 📂 app/                    # Streamlit/Flask application
│   └── app.py
├── 📂 config/                 # Configuration files
│   └── config.yaml
├── 📂 data/                   # Data files (gitignored)
│   ├── labeled_dataset.csv
│   └── .gitkeep
├── 📂 docs/                   # Documentation
│   └── figures/
├── 📂 models/                 # Trained models (gitignored)
│   ├── randomforest_pipeline.joblib
│   └── .gitkeep
├── 📂 notebooks/              # Jupyter notebooks
│   └── FakeAccount.ipynb
├── 📂 scripts/                # Utility scripts
│   ├── run_inference.py
│   └── compare_models.py
├── 📂 src/                    # Source code
│   ├── __init__.py
│   ├── feature_engineer.py
│   ├── train.py
│   └── visualize.py
├── 📂 tests/                  # Unit tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_feature_engineer.py
│   └── test_model.py
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ramezaboud/Fake-account.git
   cd Fake-account
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install package in development mode**
   ```bash
   pip install -e .
   ```

## 💻 Usage

### Training the Model

```bash
cd src
python train.py
```

This will:
- Load data from `data/labeled_dataset.csv`
- Train a RandomForest model with GridSearchCV
- Save the pipeline to `models/randomforest_pipeline.joblib`

### Running Inference

```python
import joblib
import pandas as pd

# Load the trained pipeline
pipeline = joblib.load('models/randomforest_pipeline.joblib')

# Prepare your data
user_data = pd.DataFrame([{
    'statuses_count': 100,
    'followers_count': 50,
    'friends_count': 20,
    'favourites_count': 5,
    'listed_count': 1,
    'name': 'John Doe',
    'lang': 'en',
    'created_at': '2020-01-01 12:00:00',
    'description': 'Hello world!',
    'default_profile': 0,
    'verified': 0
}])

# Make prediction
prediction = pipeline.predict(user_data)
probability = pipeline.predict_proba(user_data)[:, 1]

print(f"Prediction: {'Fake' if prediction[0] == 1 else 'Real'}")
print(f"Fake Probability: {probability[0]:.2%}")
```

### API note — decision threshold

The HTTP API exposes a query parameter `decision_threshold` for the `/predict` and `/predict/batch` endpoints. This threshold is applied on the model's predicted probability for the "fake" class (p_fake). If `p_fake >= decision_threshold` the sample is classified as fake.

The project default production threshold has been set to **0.445** to balance precision and recall based on evaluation on the test set `models/test_with_preds.csv`. You can override this per-request by appending `?decision_threshold=0.40` (or any value between 0.0 and 1.0) to the endpoint URL.

### Running Tests

```bash
pytest tests/ -v
```

## 🔧 Features

The model uses 27 engineered features:

### Basic Features
| Feature | Description |
|---------|-------------|
| `statuses_count` | Total number of tweets |
# Fake Account Detection

Comprehensive repository for detecting fake / bot accounts on social media using a RandomForest pipeline and engineered user features.

This README documents how to install, run, and maintain the project, plus notes about the production operating point and reproducibility features (fixed reference time).

---

## Key points (short)

- Production model: RandomForest pipeline stored as `models/randomforest_pipeline.joblib`.
- Default decision threshold (production operating point): **0.445** (balances precision/recall on the test set).
- Feature engineering is deterministic in production if you supply a fixed `reference_time` (recommended). The `FeatureEngineer` accepts an optional `reference_time` argument — use a stable timestamp (e.g. model training time) in production.

---

## Table of contents

- [Overview](#overview)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Quickstart: run the API](#quickstart-run-the-api)
- [API reference](#api-reference)
- [Feature engineering and reproducibility](#feature-engineering-and-reproducibility)
- [Model, threshold & evaluation](#model-threshold--evaluation)
- [Development & tests](#development--tests)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project provides a machine-learning pipeline that classifies social media user accounts as "fake" (bot) or "genuine" (real). It combines handcrafted features derived from user profile fields and activity statistics with a scikit-learn `Pipeline` whose core estimator is a RandomForest classifier.

The goal is to provide a reproducible pipeline for training, serving via a FastAPI application, and evaluating model performance at configurable operating points.

---

## Project structure

(abridged — see repository for full layout)

```
├── app/                       # FastAPI app (app/api.py)
├── config/                    # Configuration files (optional)
├── data/                      # Source datasets (gitignored)
├── docs/                      # Figures, diagrams
├── models/                    # Trained models and evaluation artifacts
│   └── randomforest_pipeline.joblib
├── notebooks/                 # Exploration notebooks
├── scripts/                   # Utility scripts & evaluation helpers
├── src/                       # Feature engineering, training code
├── tests/                     # Pytest tests
├── requirements.txt
└── README.md
```

---

## Installation

Prerequisites
- Python 3.8+
- pip

Install

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

(If you use conda, create a conda env and install similarly.)

---

## Quickstart: run the API

Start the FastAPI app from repository root:

```powershell
uvicorn app.api:app --host 127.0.0.1 --port 8001
```

Open `http://127.0.0.1:8001/docs` for interactive API docs.

Notes:
- The API loads the pipeline from `models/randomforest_pipeline.joblib` at startup.
- If the model file is missing the endpoints that depend on it will return 503 until the model is available.

---

## API reference

Endpoints (high level):

- GET `/` — basic info
- GET `/health` — health + model loaded flag
- POST `/predict` — predict a single user
   - Query parameter: `decision_threshold` (float 0.0-1.0). Default: **0.445** (project default).
- POST `/predict/batch` — predict multiple users in one request (same `decision_threshold` query param)
- GET `/model/info` — information about the loaded pipeline

Request/response schemas are implemented with Pydantic models in `src/schemas.py`.

Example: single prediction (JSON body):

```json
{
   "user": {
      "name": "John Doe",
      "screen_name": "johndoe",
      "statuses_count": 100,
      "followers_count": 50,
      "friends_count": 20,
      "favourites_count": 5,
      "listed_count": 1,
      "created_at": "2020-01-01 12:00:00",
      "description": "Hello world!",
      "lang": "en",
      "default_profile": false,
      "verified": false
   }
}
```

You can override the decision threshold per request:
`POST http://127.0.0.1:8001/predict?decision_threshold=0.40`

---

## Feature engineering and reproducibility

The feature engineering logic is in `src/feature_engineer.py`. Key points:

- The transformer `FeatureEngineer` is sklearn-compatible and exposes `fit()` and `transform()`.
- To ensure reproducible numeric features across runs and avoid nondeterminism from using the current clock at inference time, `FeatureEngineer` accepts an optional `reference_time` parameter.
   - Best practice: set `reference_time` to the model training timestamp (e.g. `MODEL_TRAINING_TIME`) when you build and save the pipeline. This keeps `account_age_days` and `tweets_per_day` consistent between training and production.
   - API / production options:
      - When building the pipeline (training), pass the training timestamp: `FeatureEngineer(reference_time=MODEL_TRAINING_TIME)` and save the pipeline.
      - When loading the pipeline in the API, the pipeline will retain the `reference_time` if it was serialized with the pipeline.
   - If `reference_time` is not provided, the transformer uses the maximum `created_at` in the input batch; if no dates are present it falls back to the current time.

Why this matters
- Using a fixed reference time prevents tiny floating-point differences between repeated `transform()` calls and ensures stable behavior in tests and production.

---

## Model, threshold & evaluation

Model
- The project uses a RandomForest pipeline stored as `models/randomforest_pipeline.joblib`.
- Earlier drafts mentioned multiple models (XGBoost, Logistic Regression), but the maintained production pipeline uses RandomForest only.

Decision threshold
- The model outputs a probability for the "fake" class (p_fake). The API classifies an account as fake when `p_fake >= decision_threshold`.
- Project default (production) decision threshold: **0.445**.
- You can override per request with the `decision_threshold` query parameter.

Evaluation example (test set `models/test_with_preds.csv` at threshold 0.445)

```
Total: 4212
TP: 1880
TN: 1985
FP: 173
FN: 174
Accuracy: 91.7616%
Precision (fake): 91.5733%
Recall (fake): 91.5287%
F1 (fake): 91.5510%
```

Choosing threshold
- Lower threshold → higher recall, lower precision.
- Higher threshold → higher precision, lower recall.
- Choose based on operational tolerance to false positives vs false negatives.

---

## Development & tests

Run tests:

```powershell
pytest -q
```

Linting and formatting
- The project does not enforce a strict linter in this repo, but we recommend using `black` / `flake8` in your workflow.

---

## Useful scripts

- `predict_eval_local.py` — locally evaluates the pipeline over `models/test_with_preds.csv` (loads pipeline directly).
- `predict_eval_threshold.py` — evaluate the pipeline for a given decision threshold and write metrics to a JSON file.
- `predict_threshold_test.py` — quick script to hit the running API at different thresholds.

---

## Contributing

Please file issues or PRs. Tests should be added to `tests/` and run via `pytest`.

---

## License

This project is MIT licensed — see `LICENSE`.

---

If you'd like, I can also:
- Add a small section showing how to save the `reference_time` metadata into the saved pipeline (joblib) and load it in `app/api.py` when starting the server, or
- Add a CI job snippet (GitHub Actions) that runs tests and linters.
