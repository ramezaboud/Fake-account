"""
Generate Evaluation Plots for Fake Account Detection Model.

This script loads the trained model and test data, then generates
all evaluation plots and saves them to docs/figures/.

Usage:
    python scripts/generate_evaluation_plots.py
    python scripts/generate_evaluation_plots.py --threshold 0.5
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import logging
import joblib
import numpy as np
import pandas as pd

from feature_engineer import FEATURE_NAMES
from visualize import (
    create_all_evaluation_plots,
    plot_threshold_analysis,
    plot_feature_importance,
    plot_model_comparison,
    DEFAULT_FIGURES_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'randomforest_pipeline.joblib'
TEST_DATA_PATH = PROJECT_ROOT / 'models' / 'test_with_preds.csv'
FIGURES_DIR = PROJECT_ROOT / 'docs' / 'figures'


def load_model_and_data():
    """Load the trained model and test data."""
    logger.info(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    
    logger.info(f"Loading test data from {TEST_DATA_PATH}")
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    logger.info(f"Test data shape: {test_df.shape}")
    logger.info(f"Columns: {list(test_df.columns)}")
    
    return model, test_df


def extract_predictions(test_df: pd.DataFrame, threshold: float = 0.445):
    """
    Extract y_true, y_pred, y_proba from test data.
    
    Args:
        test_df: Test DataFrame with label, y_pred, y_proba columns
        threshold: Decision threshold for predictions
        
    Returns:
        Tuple of (y_true, y_pred, y_proba)
    """
    y_true = test_df['label'].values
    y_proba = test_df['y_proba'].values
    
    # Recalculate predictions based on threshold
    y_pred = (y_proba >= threshold).astype(int)
    
    logger.info(f"Using threshold: {threshold}")
    logger.info(f"Total samples: {len(y_true)}")
    logger.info(f"Genuine (0): {sum(y_true == 0)}, Fake (1): {sum(y_true == 1)}")
    
    return y_true, y_pred, y_proba


def get_feature_importances(model):
    """Extract feature importances from the model."""
    try:
        # Get the classifier from the pipeline
        clf = model.named_steps['clf']
        importances = clf.feature_importances_
        logger.info(f"Extracted {len(importances)} feature importances")
        return importances
    except Exception as e:
        logger.warning(f"Could not extract feature importances: {e}")
        return None


def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate evaluation metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, confusion_matrix
    )
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn)
    }
    
    return metrics


def print_metrics(metrics: dict, threshold: float):
    """Print metrics in a formatted way."""
    print("\n" + "=" * 60)
    print(f"  EVALUATION RESULTS (Threshold: {threshold})")
    print("=" * 60)
    print(f"  Accuracy:   {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1 Score:   {metrics['f1_score']:.4f}")
    print(f"  ROC AUC:    {metrics['roc_auc']:.4f}")
    print("-" * 60)
    print(f"  Confusion Matrix:")
    print(f"    TP (True Fake):     {metrics['TP']}")
    print(f"    TN (True Genuine):  {metrics['TN']}")
    print(f"    FP (False Fake):    {metrics['FP']}")
    print(f"    FN (False Genuine): {metrics['FN']}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Generate evaluation plots')
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.445,
        help='Decision threshold (default: 0.445)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help=f'Output directory (default: {FIGURES_DIR})'
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Starting Evaluation Plot Generation")
    logger.info("=" * 60)
    
    # Load model and data
    model, test_df = load_model_and_data()
    
    # Extract predictions
    y_true, y_pred, y_proba = extract_predictions(test_df, threshold=args.threshold)
    
    # Calculate and print metrics
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    print_metrics(metrics, args.threshold)
    
    # Get feature importances
    importances = get_feature_importances(model)
    
    # Generate all evaluation plots
    logger.info(f"Generating plots and saving to {output_dir}")
    
    figures = create_all_evaluation_plots(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        feature_names=FEATURE_NAMES,
        feature_importances=importances,
        model_name='RandomForest',
        output_dir=output_dir,
        show_plots=False
    )
    
    # Generate threshold analysis plot
    logger.info("Creating threshold analysis plot...")
    plot_threshold_analysis(
        y_true=y_true,
        y_proba=y_proba,
        title="Threshold Analysis - Precision/Recall/F1",
        save_path=output_dir / 'threshold_analysis.png'
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("  GENERATED PLOTS")
    print("=" * 60)
    for plot_name in figures.keys():
        print(f"  ✅ {plot_name}")
    print(f"  ✅ threshold_analysis")
    print("-" * 60)
    print(f"  📁 All plots saved to: {output_dir}")
    print("=" * 60 + "\n")
    
    logger.info("Plot generation complete!")
    
    return figures, metrics


if __name__ == '__main__':
    main()
