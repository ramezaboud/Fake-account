"""
Train module for fake account detection.

This module provides functionality to train a machine learning pipeline
for detecting fake social media accounts. It uses RandomForest with
GridSearchCV for hyperparameter tuning.

Example:
    Run as script:
        $ python train.py --data data/labeled_dataset.csv --output models/
    
    Use as module:
        >>> from train import train_model, load_data
        >>> df, y = load_data('data/labeled_dataset.csv')
        >>> pipeline = train_model(df, y)
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_engineer import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default hyperparameter grid
DEFAULT_PARAM_GRID: Dict[str, list] = {
    'clf__n_estimators': [200, 300, 500],
    'clf__max_depth': [10, 20, 30],
    'clf__min_samples_split': [2, 5],
    'clf__min_samples_leaf': [1, 2],
    'clf__max_features': ['sqrt', 'log2'],
}


def load_data(data_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load and prepare the dataset for training.
    
    Args:
        data_path: Path to the labeled dataset CSV file.
        
    Returns:
        Tuple of (DataFrame, labels array).
        
    Raises:
        FileNotFoundError: If the data file doesn't exist.
        KeyError: If the 'label' column is missing.
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    if 'label' not in df.columns:
        raise KeyError("Dataset must contain a 'label' column")
    
    # Get labels
    y = df['label'].values
    
    # Basic cleaning: remove exact duplicates
    original_shape = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    y = df['label'].values
    
    if df.shape[0] < original_shape:
        logger.info(f"Removed {original_shape - df.shape[0]} duplicate rows")
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Label distribution: Fake (1)={sum(y==1)}, Real (0)={sum(y==0)}")
    
    return df, y


def create_pipeline(random_state: int = 42) -> Pipeline:
    """
    Create the ML pipeline with feature engineering, scaling, and classification.
    
    Args:
        random_state: Random seed for reproducibility.
        
    Returns:
        sklearn Pipeline object.
    """
    return Pipeline([
        ('fe', FeatureEngineer()),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            random_state=random_state,
            class_weight='balanced'
        ))
    ])


def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    param_grid: Optional[Dict[str, list]] = None,
    cv: int = 5,
    n_jobs: int = -1,
    verbose: int = 2
) -> GridSearchCV:
    """
    Train the model using GridSearchCV for hyperparameter tuning.
    
    Args:
        X_train: Training features DataFrame.
        y_train: Training labels.
        param_grid: Hyperparameter grid for GridSearchCV.
        cv: Number of cross-validation folds.
        n_jobs: Number of parallel jobs (-1 uses all CPUs).
        verbose: Verbosity level.
        
    Returns:
        Fitted GridSearchCV object.
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID
    
    pipeline = create_pipeline()
    
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    logger.info(f"Total hyperparameter combinations: {total_combinations}")
    
    gs = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        n_jobs=n_jobs,
        scoring='accuracy',
        verbose=verbose
    )
    
    logger.info("Starting GridSearchCV training...")
    gs.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {gs.best_params_}")
    logger.info(f"Best CV accuracy: {gs.best_score_:.4f}")
    
    return gs


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained pipeline model.
        X_test: Test features DataFrame.
        y_test: Test labels.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(
            y_test, y_pred, target_names=['Real (0)', 'Fake (1)']
        )
    }
    
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    
    logger.info("=" * 50)
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    if 'roc_auc' in metrics:
        logger.info(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info("=" * 50)
    logger.info(f"\nClassification Report:\n{metrics['classification_report']}")
    
    return metrics


def save_model(model: Pipeline, output_dir: str) -> Path:
    """
    Save the trained model to disk.
    
    Args:
        model: Trained pipeline model.
        output_dir: Directory to save the model.
        
    Returns:
        Path to the saved model file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = output_path / 'randomforest_pipeline.joblib'
    joblib.dump(model, model_path)
    
    logger.info(f"Model saved to {model_path}")
    return model_path


def main(
    data_path: str = 'data/labeled_dataset.csv',
    output_dir: str = 'models',
    test_size: float = 0.25,
    random_state: int = 42
) -> Pipeline:
    """
    Main training function.
    
    Args:
        data_path: Path to the labeled dataset.
        output_dir: Directory to save the trained model.
        test_size: Fraction of data to use for testing.
        random_state: Random seed for reproducibility.
        
    Returns:
        Trained pipeline model.
    """
    # Load data
    df, y = load_data(data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train model
    gs = train_model(X_train, y_train)
    best_model = gs.best_estimator_
    
    # Evaluate
    evaluate_model(best_model, X_test, y_test)
    
    # Save
    save_model(best_model, output_dir)
    
    return best_model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train fake account detection model'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='data/labeled_dataset.csv',
        help='Path to labeled dataset CSV'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='models',
        help='Output directory for saved model'
    )
    parser.add_argument(
        '--test-size', '-t',
        type=float,
        default=0.25,
        help='Test set size (default: 0.25)'
    )
    parser.add_argument(
        '--random-state', '-r',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(
        data_path=args.data,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.random_state
    )
