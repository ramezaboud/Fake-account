"""
Training Module for Fake Account Detection.

This module provides functions to train a Random Forest classifier
with hyperparameter tuning using GridSearchCV.

Usage:
    python train.py --data_path data/labeled_dataset.csv --output_path models/
    python train.py  # Uses default paths

Example:
    >>> from train import train_model, load_data
    >>> df, y = load_data('data/labeled_dataset.csv')
    >>> pipeline = train_model(df, y)
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    f1_score
)
import joblib

from feature_engineer import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_DATA_PATH = 'data/labeled_dataset_new.csv'
DEFAULT_OUTPUT_PATH = 'models'
DEFAULT_MODEL_NAME = 'randomforest_pipeline.joblib'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Hyperparameter grid for GridSearchCV
DEFAULT_PARAM_GRID: Dict[str, list] = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [10, 20, None],
    'clf__min_samples_split': [2, 5],
    'clf__min_samples_leaf': [1, 2],
    'clf__class_weight': ['balanced']
}


def load_data(data_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load and prepare training data.
    
    Args:
        data_path: Path to the labeled dataset CSV file.
        
    Returns:
        Tuple of (features DataFrame, labels array)
        
    Raises:
        FileNotFoundError: If data file doesn't exist.
        ValueError: If required columns are missing.
    """
    logger.info(f"Loading data from {data_path}")
    
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    if 'label' not in df.columns:
        raise ValueError("Dataset must contain 'label' column")
    
    # Remove duplicates
    original_shape = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    
    if df.shape[0] < original_shape:
        logger.info(f"Removed {original_shape - df.shape[0]} duplicate rows")
    
    y = df['label'].values
    X = df.drop(columns=['label'])
    
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Class distribution: Fake (1)={sum(y==1)}, Real (0)={sum(y==0)}")
    
    return X, y


def create_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """
    Create the ML pipeline with feature engineering and classifier.
    
    Args:
        random_state: Random seed for reproducibility.
        
    Returns:
        sklearn Pipeline with FeatureEngineer, StandardScaler, and RandomForestClassifier.
    """
    pipeline = Pipeline([
        ('fe', FeatureEngineer()),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            random_state=random_state,
            class_weight='balanced'
        ))
    ])
    logger.info("Created pipeline: FeatureEngineer -> StandardScaler -> RandomForestClassifier")
    return pipeline


def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    param_grid: Optional[Dict[str, list]] = None,
    cv: int = 5,
    n_jobs: int = -1,
    verbose: int = 1
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train the model with hyperparameter tuning using GridSearchCV.
    
    Args:
        X_train: Training features DataFrame.
        y_train: Training labels.
        param_grid: Hyperparameter grid for search. Uses default if None.
        cv: Number of cross-validation folds.
        n_jobs: Number of parallel jobs (-1 uses all CPUs).
        verbose: Verbosity level.
        
    Returns:
        Tuple of (best fitted pipeline, best parameters dict)
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID
    
    pipeline = create_pipeline()
    
    # Log search info
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    logger.info(f"Total hyperparameter combinations: {total_combinations}")
    logger.info(f"Starting GridSearchCV with {cv}-fold CV")
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        n_jobs=n_jobs,
        scoring='roc_auc',
        verbose=verbose
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV ROC AUC: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained pipeline.
        X_test: Test features DataFrame.
        y_test: Test labels.
        
    Returns:
        Dictionary with evaluation metrics (accuracy, roc_auc, f1_score).
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    logger.info("=" * 50)
    logger.info("Model Evaluation Results:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
    logger.info("=" * 50)
    
    # Note: In the dataset, label 0 = genuine/real, label 1 = fake
    report = classification_report(y_test, y_pred, target_names=['Real (0)', 'Fake (1)'])
    print("\nClassification Report:")
    print(report)
    
    metrics['classification_report'] = report
    
    return metrics


def save_model(
    model: Pipeline,
    output_path: str,
    model_name: str = DEFAULT_MODEL_NAME
) -> Path:
    """
    Save the trained model to disk.
    
    Args:
        model: Trained pipeline to save.
        output_path: Directory to save the model.
        model_name: Filename for the model.
        
    Returns:
        Path to the saved model file.
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / model_name
    joblib.dump(model, model_path)
    
    logger.info(f"Model saved to {model_path}")
    return model_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Fake Account Detection Model'
    )
    parser.add_argument(
        '--data_path', '-d',
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f'Path to labeled dataset CSV (default: {DEFAULT_DATA_PATH})'
    )
    parser.add_argument(
        '--output_path', '-o',
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f'Output directory for model (default: {DEFAULT_OUTPUT_PATH})'
    )
    parser.add_argument(
        '--test_size', '-t',
        type=float,
        default=TEST_SIZE,
        help=f'Test set proportion (default: {TEST_SIZE})'
    )
    parser.add_argument(
        '--cv',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--random_state', '-r',
        type=int,
        default=RANDOM_STATE,
        help=f'Random seed (default: {RANDOM_STATE})'
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Starting Fake Account Detection Model Training")
    logger.info("=" * 60)
    
    # Load data
    X, y = load_data(args.data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    logger.info(f"Class distribution - Train: {pd.Series(y_train).value_counts().to_dict()}")
    
    # Train model
    best_model, best_params = train_model(X_train, y_train, cv=args.cv)
    
    # Evaluate
    metrics = evaluate_model(best_model, X_test, y_test)
    
    # Save model
    model_path = save_model(best_model, args.output_path)
    
    logger.info("Training complete!")
    
    return best_model, metrics


if __name__ == '__main__':
    main()
