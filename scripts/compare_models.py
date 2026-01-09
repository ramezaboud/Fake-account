"""
Model comparison script for fake account detection.

This script trains and compares multiple ML models (RandomForest, XGBoost)
and generates evaluation visualizations.

Example:
    $ python compare_models.py --data data/labeled_dataset.csv
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Run: pip install xgboost")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from feature_engineer import FeatureEngineer, FEATURE_NAMES
from visualize import (
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_model_comparison,
    create_all_evaluation_plots
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Model configurations
MODELS_CONFIG = {
    'RandomForest': {
        'classifier': RandomForestClassifier(
            random_state=42,
            class_weight='balanced'
        ),
        'param_grid': {
            'clf__n_estimators': [200, 300],
            'clf__max_depth': [10, 20],
            'clf__min_samples_split': [2, 5],
        }
    },
    'LogisticRegression': {
        'classifier': LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        ),
        'param_grid': {
            'clf__C': [0.1, 1, 10],
            'clf__penalty': ['l2'],
        }
    },
}

if XGBOOST_AVAILABLE:
    MODELS_CONFIG['XGBoost'] = {
        'classifier': XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            n_jobs=1  # Avoid parallelization issues
        ),
        'param_grid': {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [3, 5],
            'clf__learning_rate': [0.1],
        }
    }


def load_data(data_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load and prepare dataset."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    y = df['label'].values
    df = df.drop_duplicates().reset_index(drop=True)
    y = df['label'].values
    
    logger.info(f"Dataset: {df.shape[0]} samples, {sum(y==1)} fake, {sum(y==0)} real")
    return df, y


def create_pipeline(classifier) -> Pipeline:
    """Create ML pipeline with given classifier."""
    return Pipeline([
        ('fe', FeatureEngineer()),
        ('scaler', StandardScaler()),
        ('clf', classifier)
    ])


def train_and_evaluate(
    model_name: str,
    classifier,
    param_grid: Dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cv: int = 3
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train a model with GridSearchCV and evaluate it.
    
    Returns:
        Tuple of (best_model, metrics_dict)
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Training {model_name}...")
    logger.info(f"{'='*50}")
    
    pipeline = create_pipeline(classifier)
    
    # Use n_jobs=1 for XGBoost to avoid memory issues
    n_jobs_gs = 1 if model_name == 'XGBoost' else -1
    
    gs = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        n_jobs=n_jobs_gs,
        scoring='accuracy',
        verbose=1
    )
    
    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_
    
    # Predictions
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'best_params': gs.best_params_,
        'cv_score': gs.best_score_,
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    
    logger.info(f"\n{model_name} Results:")
    logger.info(f"  Best CV Score: {metrics['cv_score']:.4f}")
    logger.info(f"  Test Accuracy: {metrics['accuracy']:.4f}")
    if 'roc_auc' in metrics:
        logger.info(f"  Test ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"  Best Params: {metrics['best_params']}")
    
    return best_model, metrics


def compare_models(
    data_path: str = 'data/labeled_dataset.csv',
    output_dir: str = 'docs/figures',
    models_dir: str = 'models',
    test_size: float = 0.25,
    random_state: int = 42
) -> Dict[str, Dict]:
    """
    Compare multiple models and generate visualizations.
    
    Returns:
        Dictionary with model names as keys and results as values.
    """
    # Load data
    df, y = load_data(data_path)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    results = {}
    best_models = {}
    
    # Train all models
    for model_name, config in MODELS_CONFIG.items():
        model, metrics = train_and_evaluate(
            model_name,
            config['classifier'],
            config['param_grid'],
            X_train, X_test,
            y_train, y_test
        )
        results[model_name] = metrics
        best_models[model_name] = model
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = best_models[best_model_name]
    
    logger.info(f"\n{'='*50}")
    logger.info(f"BEST MODEL: {best_model_name}")
    logger.info(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    logger.info(f"{'='*50}")
    
    # Create visualizations
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Model comparison plots
    plot_model_comparison(results, metric='accuracy', save_path=output_dir)
    if all('roc_auc' in r for r in results.values()):
        plot_model_comparison(results, metric='roc_auc', save_path=output_dir)
    
    # Best model evaluation plots
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    plot_confusion_matrix(y_test, y_pred, save_path=output_dir)
    plot_roc_curve(y_test, y_proba, save_path=output_dir)
    plot_precision_recall_curve(y_test, y_proba, save_path=output_dir)
    
    # Feature importance (for tree-based models)
    try:
        plot_feature_importance(best_model, FEATURE_NAMES, save_path=output_dir)
    except Exception as e:
        logger.warning(f"Could not plot feature importance: {e}")
    
    # Save best model
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(models_dir) / f'{best_model_name.lower()}_pipeline.joblib'
    joblib.dump(best_model, model_path)
    logger.info(f"Best model saved to {model_path}")
    
    # Print summary table
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<20} {'Accuracy':>12} {'ROC AUC':>12} {'F1 Score':>12}")
    print("-"*70)
    for name, metrics in results.items():
        roc = metrics.get('roc_auc', 'N/A')
        roc_str = f"{roc:.4f}" if isinstance(roc, float) else roc
        print(f"{name:<20} {metrics['accuracy']:>12.4f} {roc_str:>12} {metrics['f1_score']:>12.4f}")
    print("="*70)
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare ML models for fake account detection')
    parser.add_argument('--data', '-d', default='data/labeled_dataset.csv', help='Data path')
    parser.add_argument('--output', '-o', default='docs/figures', help='Output directory for figures')
    parser.add_argument('--models', '-m', default='models', help='Directory for saved models')
    parser.add_argument('--test-size', '-t', type=float, default=0.25, help='Test set size')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    results = compare_models(
        data_path=args.data,
        output_dir=args.output,
        models_dir=args.models,
        test_size=args.test_size
    )
