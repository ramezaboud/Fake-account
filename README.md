# 🤖 Fake Account Detection# 🤖 Fake Account Detection



[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7+-orange.svg)](https://scikit-learn.org/)[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Fake account classifier using a Random Forest pipeline with custom feature engineering for text, user info, and activity-based features.

A machine learning system for detecting fake/bot accounts on social media platforms using Random Forest classification with advanced feature engineering.

## 📋 Table of Contents

## 📊 Model Performance

- [Overview](#-overview)

| Metric | Score |- [Project Structure](#-project-structure)

|--------|-------|- [Installation](#-installation)

| **Accuracy** | 92.07% |- [Usage](#-usage)

| **ROC AUC** | 98.47% |- [Features](#-features)

| **F1 Score** | 92.12% |- [Model Performance](#-model-performance)

| **Precision** | 91.57% |- [Contributing](#-contributing)

| **Recall** | 92.67% |- [License](#-license)



*Evaluated on test set with decision threshold = 0.445*## 🎯 Overview



## 🎯 OverviewThis project detects fake/bot accounts on social media platforms using machine learning. It analyzes user profiles based on:

- **Activity patterns** (tweets per day, account age)

This project provides a complete ML pipeline for classifying social media accounts as "fake" (bot) or "genuine" (real). It analyzes user profiles based on:- **Network metrics** (followers, friends, listed count)

- **Profile characteristics** (description length, default profile, verified status)

- **Activity Patterns** — tweets per day, account age, posting frequency- **Demographic inference** (gender from name)

- **Network Metrics** — followers, friends, listed count, engagement ratios- **Ratio-based features** (followers/friends ratio, engagement ratio)

- **Profile Characteristics** — description length, default profile, verified status- **Suspicious patterns** (zero followers, high friend rate)

- **Demographic Inference** — gender estimation from username

- **Suspicious Patterns** — zero followers detection, high friend rate anomalies## 📁 Project Structure

- **Ratio-based Features** — followers/friends ratio, engagement metrics

```

## 📁 Project Structurefake-account/

├── 📂 app/                    # Streamlit/Flask application

```│   └── app.py

ml project 2/├── 📂 config/                 # Configuration files

├── app/│   └── config.yaml

│   └── api.py                 # FastAPI REST API├── 📂 data/                   # Data files (gitignored)

├── config/│   ├── labeled_dataset.csv

│   └── config.yaml            # Configuration settings│   └── .gitkeep

├── data/├── 📂 docs/                   # Documentation

│   └── labeled_dataset_new.csv # Training dataset│   └── figures/

├── docs/├── 📂 models/                 # Trained models (gitignored)

│   └── figures/               # Evaluation plots & visualizations│   ├── randomforest_pipeline.joblib

├── models/│   └── .gitkeep

│   ├── randomforest_pipeline.joblib      # Production model├── 📂 notebooks/              # Jupyter notebooks

│   ├── eval_results.json                 # Evaluation metrics│   └── FakeAccount.ipynb

│   └── test_with_preds.csv               # Test predictions├── 📂 scripts/                # Utility scripts

├── notebooks/│   ├── run_inference.py

│   └── FakeAccount.ipynb      # Exploratory analysis│   └── compare_models.py

├── scripts/├── 📂 src/                    # Source code

│   ├── run_inference.py       # Batch inference script│   ├── __init__.py

│   ├── compare_models.py      # Model comparison utilities│   ├── feature_engineer.py

│   └── generate_evaluation_plots.py  # Generate evaluation visualizations│   ├── train.py

├── src/│   └── visualize.py

│   ├── __init__.py├── 📂 tests/                  # Unit tests

│   ├── feature_engineer.py    # Custom sklearn transformer (29 features)│   ├── __init__.py

│   ├── train.py               # Model training pipeline│   ├── conftest.py

│   ├── schemas.py             # Pydantic data models│   ├── test_feature_engineer.py

│   └── visualize.py           # Visualization utilities│   └── test_model.py

├── tests/├── .gitignore

│   ├── conftest.py            # Pytest fixtures├── LICENSE

│   ├── test_feature_engineer.py├── pyproject.toml

│   ├── test_integration.py├── README.md

│   └── test_model.py└── requirements.txt

├── requirements.txt```

├── pyproject.toml

└── README.md## 🚀 Installation

```

### Prerequisites

## 🚀 Installation

- Python 3.8 or higher

### Prerequisites- pip or conda



- Python 3.8 or higher### Setup

- pip

1. **Clone the repository**

### Setup   ```bash

   git clone https://github.com/ramezaboud/Fake-account.git

1. **Clone the repository**   cd Fake-account

   ```bash   ```

   git clone https://github.com/yourusername/fake-account-detection.git

   cd fake-account-detection2. **Create virtual environment**

   ```   ```bash

   python -m venv venv

2. **Create virtual environment**   source venv/bin/activate  # Linux/Mac

   ```bash   # or

   python -m venv .venv   .\venv\Scripts\activate   # Windows

      ```

   # Windows

   .\.venv\Scripts\activate3. **Install dependencies**

      ```bash

   # Linux/Mac   pip install -r requirements.txt

   source .venv/bin/activate   ```

   ```

4. **Install package in development mode**

3. **Install dependencies**   ```bash

   ```bash   pip install -e .

   pip install -r requirements.txt   ```

   ```

## 💻 Usage

4. **Install package in development mode**

   ```bash### Training the Model

   pip install -e .

   ``````bash

cd src

## 💻 Usagepython train.py

```

### Training the Model

This will:

```bash- Load data from `data/labeled_dataset.csv`

python src/train.py- Train a RandomForest model with GridSearchCV

```- Save the pipeline to `models/randomforest_pipeline.joblib`



This will:### Running Inference

- Load data from `data/labeled_dataset_new.csv`

- Engineer 29 features using `FeatureEngineer````python

- Train a RandomForest model with hyperparameter tuningimport joblib

- Save the pipeline to `models/randomforest_pipeline.joblib`import pandas as pd



### Running the API# Load the trained pipeline

pipeline = joblib.load('models/randomforest_pipeline.joblib')

Start the FastAPI server:

# Prepare your data

```bashuser_data = pd.DataFrame([{

uvicorn app.api:app --host 127.0.0.1 --port 8001    'statuses_count': 100,

```    'followers_count': 50,

    'friends_count': 20,

Open `http://127.0.0.1:8001/docs` for interactive API documentation.    'favourites_count': 5,

    'listed_count': 1,

### API Endpoints    'name': 'John Doe',

    'lang': 'en',

| Endpoint | Method | Description |    'created_at': '2020-01-01 12:00:00',

|----------|--------|-------------|    'description': 'Hello world!',

| `/` | GET | API info |    'default_profile': 0,

| `/health` | GET | Health check & model status |    'verified': 0

| `/predict` | POST | Single user prediction |}])

| `/predict/batch` | POST | Batch predictions |

| `/model/info` | GET | Model information |# Make prediction

prediction = pipeline.predict(user_data)

### Example: Single Predictionprobability = pipeline.predict_proba(user_data)[:, 1]



```pythonprint(f"Prediction: {'Fake' if prediction[0] == 1 else 'Real'}")

import requestsprint(f"Fake Probability: {probability[0]:.2%}")

```

user_data = {

    "user": {### API note — decision threshold

        "name": "John Doe",

        "screen_name": "johndoe",The HTTP API exposes a query parameter `decision_threshold` for the `/predict` and `/predict/batch` endpoints. This threshold is applied on the model's predicted probability for the "fake" class (p_fake). If `p_fake >= decision_threshold` the sample is classified as fake.

        "statuses_count": 100,

        "followers_count": 50,The project default production threshold has been set to **0.445** to balance precision and recall based on evaluation on the test set `models/test_with_preds.csv`. You can override this per-request by appending `?decision_threshold=0.40` (or any value between 0.0 and 1.0) to the endpoint URL.

        "friends_count": 20,

        "favourites_count": 5,### Running Tests

        "listed_count": 1,

        "created_at": "2020-01-01 12:00:00",```bash

        "description": "Hello world!",pytest tests/ -v

        "lang": "en",```

        "default_profile": False,

        "verified": False## 🔧 Features

    }

}The model uses 27 engineered features:



response = requests.post(### Basic Features

    "http://127.0.0.1:8001/predict?decision_threshold=0.445",| Feature | Description |

    json=user_data|---------|-------------|

)| `statuses_count` | Total number of tweets |

print(response.json())# Fake Account Detection

```

Comprehensive repository for detecting fake / bot accounts on social media using a RandomForest pipeline and engineered user features.

### Decision Threshold

This README documents how to install, run, and maintain the project, plus notes about the production operating point and reproducibility features (fixed reference time).

The API uses a configurable decision threshold (default: **0.445**):

- `p_fake >= threshold` → classified as **Fake**---

- `p_fake < threshold` → classified as **Real**

## Key points (short)

Override per request: `POST /predict?decision_threshold=0.40`

- Production model: RandomForest pipeline stored as `models/randomforest_pipeline.joblib`.

| Threshold | Use Case |- Default decision threshold (production operating point): **0.445** (balances precision/recall on the test set).

|-----------|----------|- Feature engineering is deterministic in production if you supply a fixed `reference_time` (recommended). The `FeatureEngineer` accepts an optional `reference_time` argument — use a stable timestamp (e.g. model training time) in production.

| Lower (0.30-0.40) | Higher recall, catch more bots |

| Default (0.445) | Balanced precision/recall |---

| Higher (0.50-0.60) | Higher precision, fewer false positives |

## Table of contents

### Running Inference Locally

- [Overview](#overview)

```python- [Project structure](#project-structure)

import joblib- [Installation](#installation)

import pandas as pd- [Quickstart: run the API](#quickstart-run-the-api)

- [API reference](#api-reference)

# Load pipeline- [Feature engineering and reproducibility](#feature-engineering-and-reproducibility)

pipeline = joblib.load('models/randomforest_pipeline.joblib')- [Model, threshold & evaluation](#model-threshold--evaluation)

- [Development & tests](#development--tests)

# Prepare data- [Contributing](#contributing)

user = pd.DataFrame([{- [License](#license)

    'statuses_count': 100,

    'followers_count': 50,---

    'friends_count': 20,

    'favourites_count': 5,## Overview

    'listed_count': 1,

    'name': 'John Doe',This project provides a machine-learning pipeline that classifies social media user accounts as "fake" (bot) or "genuine" (real). It combines handcrafted features derived from user profile fields and activity statistics with a scikit-learn `Pipeline` whose core estimator is a RandomForest classifier.

    'lang': 'en',

    'created_at': '2020-01-01 12:00:00',The goal is to provide a reproducible pipeline for training, serving via a FastAPI application, and evaluating model performance at configurable operating points.

    'description': 'Hello world!',

    'default_profile': 0,---

    'verified': 0

}])## Project structure



# Predict(abridged — see repository for full layout)

prediction = pipeline.predict(user)

probability = pipeline.predict_proba(user)[:, 1]```

├── app/                       # FastAPI app (app/api.py)

print(f"Prediction: {'Fake' if prediction[0] == 1 else 'Real'}")├── config/                    # Configuration files (optional)

print(f"Fake Probability: {probability[0]:.2%}")├── data/                      # Source datasets (gitignored)

```├── docs/                      # Figures, diagrams

├── models/                    # Trained models and evaluation artifacts

## 🔧 Features│   └── randomforest_pipeline.joblib

├── notebooks/                 # Exploration notebooks

The model uses **29 engineered features**:├── scripts/                   # Utility scripts & evaluation helpers

├── src/                       # Feature engineering, training code

### Profile Features├── tests/                     # Pytest tests

- `statuses_count`, `followers_count`, `friends_count`├── requirements.txt

- `favourites_count`, `listed_count`└── README.md

- `description_length`, `name_length`, `screen_name_length````

- `default_profile`, `verified`

---

### Temporal Features

- `account_age_days` — days since account creation## Installation

- `tweets_per_day` — average posting frequency

Prerequisites

### Ratio Features- Python 3.8+

- `followers_friends_ratio` — followers/friends balance- pip

- `listed_followers_ratio` — list membership relative to followers

- `favourites_statuses_ratio` — engagement patternInstall



### Derived Features```powershell

- `friend_rate` — friends acquired per daypython -m venv .venv

- `follower_rate` — followers acquired per day.\.venv\Scripts\activate

- `engagement_ratio` — overall engagement metricpip install -r requirements.txt

```

### Binary Flags

- `has_description`, `has_url`, `has_location`(If you use conda, create a conda env and install similarly.)

- `is_zero_followers`, `is_zero_friends`

- `is_high_friend_rate`, `is_low_tweet_rate`---

- `gender_*` — inferred from name (male/female/unknown)

## Quickstart: run the API

## 📈 Evaluation Plots

Start the FastAPI app from repository root:

Generate evaluation visualizations:

```powershell

```bashuvicorn app.api:app --host 127.0.0.1 --port 8001

python scripts/generate_evaluation_plots.py```

```

Open `http://127.0.0.1:8001/docs` for interactive API docs.

This creates plots in `docs/figures/`:

- `confusion_matrix.png` — Classification confusion matrixNotes:

- `roc_curve.png` — ROC curve with AUC score- The API loads the pipeline from `models/randomforest_pipeline.joblib` at startup.

- `precision_recall_curve.png` — Precision-Recall tradeoff- If the model file is missing the endpoints that depend on it will return 503 until the model is available.

- `feature_importance.png` — Top 15 important features

- `threshold_analysis.png` — Metrics across decision thresholds---



## 🧪 Testing## API reference



Run all tests:Endpoints (high level):



```bash- GET `/` — basic info

pytest tests/ -v- GET `/health` — health + model loaded flag

```- POST `/predict` — predict a single user

   - Query parameter: `decision_threshold` (float 0.0-1.0). Default: **0.445** (project default).

Run specific test files:- POST `/predict/batch` — predict multiple users in one request (same `decision_threshold` query param)

- GET `/model/info` — information about the loaded pipeline

```bash

pytest tests/test_feature_engineer.py -vRequest/response schemas are implemented with Pydantic models in `src/schemas.py`.

pytest tests/test_model.py -v

```Example: single prediction (JSON body):



## 🔬 Feature Engineering & Reproducibility```json

{

The `FeatureEngineer` transformer supports a `reference_time` parameter for reproducible predictions:   "user": {

      "name": "John Doe",

```python      "screen_name": "johndoe",

from src.feature_engineer import FeatureEngineer      "statuses_count": 100,

      "followers_count": 50,

# Use fixed reference time for consistent results      "friends_count": 20,

fe = FeatureEngineer(reference_time="2025-01-01 00:00:00")      "favourites_count": 5,

```      "listed_count": 1,

      "created_at": "2020-01-01 12:00:00",

This ensures `account_age_days` and `tweets_per_day` remain stable across inference calls.      "description": "Hello world!",

      "lang": "en",

## 📚 Documentation      "default_profile": false,

      "verified": false

- [API Documentation](docs/API.md) — Detailed API reference   }

- [Jupyter Notebook](notebooks/FakeAccount.ipynb) — Exploratory analysis}

```

## 🤝 Contributing

You can override the decision threshold per request:

1. Fork the repository`POST http://127.0.0.1:8001/predict?decision_threshold=0.40`

2. Create a feature branch (`git checkout -b feature/amazing-feature`)

3. Commit changes (`git commit -m 'Add amazing feature'`)---

4. Push to branch (`git push origin feature/amazing-feature`)

5. Open a Pull Request## Feature engineering and reproducibility



See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.The feature engineering logic is in `src/feature_engineer.py`. Key points:



## 📄 License- The transformer `FeatureEngineer` is sklearn-compatible and exposes `fit()` and `transform()`.

- To ensure reproducible numeric features across runs and avoid nondeterminism from using the current clock at inference time, `FeatureEngineer` accepts an optional `reference_time` parameter.

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.   - Best practice: set `reference_time` to the model training timestamp (e.g. `MODEL_TRAINING_TIME`) when you build and save the pipeline. This keeps `account_age_days` and `tweets_per_day` consistent between training and production.

   - API / production options:

## 👨‍💻 Author      - When building the pipeline (training), pass the training timestamp: `FeatureEngineer(reference_time=MODEL_TRAINING_TIME)` and save the pipeline.

      - When loading the pipeline in the API, the pipeline will retain the `reference_time` if it was serialized with the pipeline.

**Ramez Aboud**   - If `reference_time` is not provided, the transformer uses the maximum `created_at` in the input batch; if no dates are present it falls back to the current time.



---Why this matters

- Using a fixed reference time prevents tiny floating-point differences between repeated `transform()` calls and ensures stable behavior in tests and production.

<p align="center">

  <i>Built with ❤️ for detecting fake accounts</i>---

</p>

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
