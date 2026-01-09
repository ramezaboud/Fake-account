# API Reference

This document provides detailed API documentation for the Fake Account Detection project.

## Table of Contents

- [feature_engineer](#feature_engineer)
- [train](#train)
- [visualize](#visualize)

---

## feature_engineer

Feature engineering module for fake account detection.

### Constants

```python
FEATURE_NAMES: List[str]
```
List of all feature names used in the model:
```python
[
    'statuses_count', 'followers_count', 'friends_count',
    'favourites_count', 'listed_count', 'sex_code',
    'lang_code', 'tweets_per_day', 'account_age_days',
    'description_length', 'default_profile', 'verified'
]
```

```python
GENDER_MAPPING: Dict[str, int]
```
Mapping of gender-guesser results to numeric codes:
```python
{
    'male': 2,
    'mostly_male': 1,
    'unknown': 0,
    'andy': 0,
    'mostly_female': -1,
    'female': -2
}
```

### Functions

#### `predict_sex(name: str) -> int`

Predict gender from a name using the gender-guesser library.

**Parameters:**
- `name` (str): The name to analyze

**Returns:**
- `int`: Gender code (-2 to 2)
  - 2: male
  - 1: mostly male
  - 0: unknown/androgynous
  - -1: mostly female
  - -2: female

**Example:**
```python
>>> from feature_engineer import predict_sex
>>> predict_sex("John")
2
>>> predict_sex("Mary")
-2
```

#### `to_bool_int(val: Any) -> int`

Convert a value to boolean integer (0 or 1).

**Parameters:**
- `val` (Any): Value to convert

**Returns:**
- `int`: 1 if truthy, 0 if falsy

**Example:**
```python
>>> from feature_engineer import to_bool_int
>>> to_bool_int(True)
1
>>> to_bool_int(0)
0
```

### Classes

#### `FeatureEngineer`

sklearn-compatible transformer for feature extraction.

```python
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        ...
    
    def fit(self, X, y=None):
        """Fit the transformer (learns language encoding)."""
        ...
    
    def transform(self, X):
        """Transform input DataFrame to feature matrix."""
        ...
```

**Methods:**

##### `fit(X, y=None) -> FeatureEngineer`
Fit the transformer to learn language mappings.

**Parameters:**
- `X` (pd.DataFrame): Input data
- `y` (optional): Ignored

**Returns:**
- `self`: Fitted transformer

##### `transform(X) -> np.ndarray`
Transform input data to feature matrix.

**Parameters:**
- `X` (pd.DataFrame): Input data with columns:
  - `statuses_count`, `followers_count`, `friends_count`
  - `favourites_count`, `listed_count`
  - `name`, `lang`, `created_at`
  - `description`, `default_profile`, `verified`

**Returns:**
- `np.ndarray`: Feature matrix of shape (n_samples, 12)

**Example:**
```python
>>> from feature_engineer import FeatureEngineer
>>> import pandas as pd
>>> 
>>> fe = FeatureEngineer()
>>> df = pd.DataFrame([{
...     'statuses_count': 100,
...     'followers_count': 50,
...     'name': 'John',
...     'lang': 'en',
...     'created_at': '2020-01-01',
...     'description': 'Hello!',
...     'default_profile': 0,
...     'verified': 0,
...     'friends_count': 20,
...     'favourites_count': 5,
...     'listed_count': 1
... }])
>>> 
>>> features = fe.fit_transform(df)
>>> print(features.shape)
(1, 12)
```

---

## train

Training module for fake account detection models.

### Constants

```python
DEFAULT_PARAM_GRID: Dict[str, list]
```
Default hyperparameter grid for GridSearchCV:
```python
{
    'clf__n_estimators': [200, 300, 500],
    'clf__max_depth': [10, 20, 30],
    'clf__min_samples_split': [2, 5],
    'clf__min_samples_leaf': [1, 2],
    'clf__max_features': ['sqrt', 'log2'],
}
```

### Functions

#### `load_data(data_path: str) -> Tuple[pd.DataFrame, np.ndarray]`

Load and prepare the dataset for training.

**Parameters:**
- `data_path` (str): Path to the labeled dataset CSV file

**Returns:**
- `Tuple[pd.DataFrame, np.ndarray]`: (DataFrame, labels array)

**Raises:**
- `FileNotFoundError`: If the data file doesn't exist
- `KeyError`: If the 'label' column is missing

#### `create_pipeline(random_state: int = 42) -> Pipeline`

Create the ML pipeline with feature engineering, scaling, and classification.

**Parameters:**
- `random_state` (int): Random seed for reproducibility

**Returns:**
- `Pipeline`: sklearn Pipeline object

#### `train_model(...) -> GridSearchCV`

Train the model using GridSearchCV for hyperparameter tuning.

**Parameters:**
- `X_train` (pd.DataFrame): Training features
- `y_train` (np.ndarray): Training labels
- `param_grid` (Optional[Dict]): Hyperparameter grid
- `cv` (int): Number of CV folds (default: 5)
- `n_jobs` (int): Parallel jobs (default: -1)
- `verbose` (int): Verbosity level (default: 2)

**Returns:**
- `GridSearchCV`: Fitted GridSearchCV object

#### `evaluate_model(...) -> Dict[str, Any]`

Evaluate the model on test data.

**Parameters:**
- `model` (Pipeline): Trained pipeline
- `X_test` (pd.DataFrame): Test features
- `y_test` (np.ndarray): Test labels

**Returns:**
- `Dict[str, Any]`: Dictionary with metrics (accuracy, roc_auc, classification_report)

#### `save_model(model: Pipeline, output_dir: str) -> Path`

Save the trained model to disk.

**Parameters:**
- `model` (Pipeline): Trained pipeline
- `output_dir` (str): Directory to save the model

**Returns:**
- `Path`: Path to the saved model file

#### `main(...) -> Pipeline`

Main training function.

**Parameters:**
- `data_path` (str): Path to labeled dataset
- `output_dir` (str): Output directory for model
- `test_size` (float): Test set fraction
- `random_state` (int): Random seed

**Returns:**
- `Pipeline`: Trained pipeline

### CLI Usage

```bash
python train.py --data data/labeled_dataset.csv --output models/ --test-size 0.25
```

---

## visualize

Visualization module for model evaluation.

### Functions

#### `plot_feature_importance(...) -> plt.Figure`

Plot feature importance from a trained model.

**Parameters:**
- `model`: Trained model with `feature_importances_`
- `feature_names` (List[str]): Feature names
- `top_n` (int): Number of top features (default: 15)
- `save_path` (Optional[str]): Directory to save figure
- `figsize` (tuple): Figure size (default: (10, 8))

**Returns:**
- `plt.Figure`: matplotlib Figure object

#### `plot_confusion_matrix(...) -> plt.Figure`

Plot confusion matrix heatmap.

**Parameters:**
- `y_true` (np.ndarray): True labels
- `y_pred` (np.ndarray): Predicted labels
- `labels` (List[str]): Class labels
- `save_path` (Optional[str]): Directory to save figure
- `figsize` (tuple): Figure size

#### `plot_roc_curve(...) -> plt.Figure`

Plot ROC curve.

**Parameters:**
- `y_true` (np.ndarray): True labels
- `y_proba` (np.ndarray): Predicted probabilities
- `save_path` (Optional[str]): Directory to save figure
- `figsize` (tuple): Figure size

#### `plot_precision_recall_curve(...) -> plt.Figure`

Plot Precision-Recall curve.

**Parameters:**
- `y_true` (np.ndarray): True labels
- `y_proba` (np.ndarray): Predicted probabilities
- `save_path` (Optional[str]): Directory to save figure
- `figsize` (tuple): Figure size

#### `plot_model_comparison(...) -> plt.Figure`

Plot comparison of multiple models.

**Parameters:**
- `results` (dict): Model results dictionary
- `metric` (str): Metric to compare
- `save_path` (Optional[str]): Directory to save figure
- `figsize` (tuple): Figure size

#### `create_all_evaluation_plots(...) -> dict`

Create all evaluation plots for a model.

**Parameters:**
- `model`: Trained model pipeline
- `X_test`: Test features
- `y_test` (np.ndarray): True test labels
- `feature_names` (List[str]): Feature names
- `save_path` (str): Directory to save figures

**Returns:**
- `dict`: Dictionary of figure objects

---

## Quick Start

```python
from src import (
    FeatureEngineer, 
    load_data, 
    train_model, 
    evaluate_model,
    FEATURE_NAMES
)
from src.visualize import create_all_evaluation_plots

# Load data
df, y = load_data('data/labeled_dataset.csv')

# Train model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25)
gs = train_model(X_train, y_train)
model = gs.best_estimator_

# Evaluate
metrics = evaluate_model(model, X_test, y_test)

# Visualize
figures = create_all_evaluation_plots(
    model, X_test, y_test, FEATURE_NAMES,
    save_path='docs/figures'
)
```
