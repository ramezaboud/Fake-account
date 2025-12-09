"""
Visualization module for model evaluation and analysis.

This module provides functions to create various plots for model evaluation
including feature importance, confusion matrix, ROC curve, and PR curve.

Example:
    >>> from visualize import plot_feature_importance, plot_confusion_matrix
    >>> plot_feature_importance(model, feature_names, save_path='docs/figures/')
    >>> plot_confusion_matrix(y_test, y_pred, save_path='docs/figures/')
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# Configure logging
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 15,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
) -> plt.Figure:
    """
    Plot feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute.
                Can be a Pipeline (will extract classifier).
        feature_names: List of feature names.
        top_n: Number of top features to display.
        save_path: Directory to save the figure. If None, displays plot.
        figsize: Figure size tuple.
        
    Returns:
        matplotlib Figure object.
    """
    # Extract classifier from pipeline if needed
    if hasattr(model, 'named_steps'):
        clf = model.named_steps.get('clf', model)
    else:
        clf = model
    
    if not hasattr(clf, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))
    
    ax.barh(
        range(top_n),
        importances[indices][::-1],
        color=colors
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1])
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Top Feature Importances', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        output_dir = Path(save_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / 'feature_importance.png'
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {filepath}")
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Class labels for display.
        save_path: Directory to save the figure.
        figsize: Figure size tuple.
        
    Returns:
        matplotlib Figure object.
    """
    if labels is None:
        labels = ['Real (0)', 'Fake (1)']
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={'size': 14}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        output_dir = Path(save_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / 'confusion_matrix.png'
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {filepath}")
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6)
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels.
        y_proba: Predicted probabilities for positive class.
        save_path: Directory to save the figure.
        figsize: Figure size tuple.
        
    Returns:
        matplotlib Figure object.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(
        fpr, tpr,
        color='darkorange',
        lw=2,
        label=f'ROC curve (AUC = {roc_auc:.3f})'
    )
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    ax.fill_between(fpr, tpr, alpha=0.3, color='darkorange')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        output_dir = Path(save_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / 'roc_curve.png'
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"ROC curve saved to {filepath}")
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6)
) -> plt.Figure:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels.
        y_proba: Predicted probabilities for positive class.
        save_path: Directory to save the figure.
        figsize: Figure size tuple.
        
    Returns:
        matplotlib Figure object.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(
        recall, precision,
        color='green',
        lw=2,
        label=f'PR curve (AP = {avg_precision:.3f})'
    )
    
    ax.fill_between(recall, precision, alpha=0.3, color='green')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        output_dir = Path(save_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / 'precision_recall_curve.png'
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"Precision-Recall curve saved to {filepath}")
    
    return fig


def plot_model_comparison(
    results: dict,
    metric: str = 'accuracy',
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plot comparison of multiple models.
    
    Args:
        results: Dictionary with model names as keys and metrics dict as values.
                 Example: {'RandomForest': {'accuracy': 0.95, 'roc_auc': 0.98}}
        metric: Metric to compare ('accuracy', 'roc_auc', etc.)
        save_path: Directory to save the figure.
        figsize: Figure size tuple.
        
    Returns:
        matplotlib Figure object.
    """
    models = list(results.keys())
    scores = [results[m].get(metric, 0) for m in models]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, scores, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.annotate(
            f'{score:.4f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=12, fontweight='bold'
        )
    
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Model Comparison - {metric.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(scores) * 1.15])
    
    plt.tight_layout()
    
    if save_path:
        output_dir = Path(save_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f'model_comparison_{metric}.png'
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {filepath}")
    
    return fig


def create_all_evaluation_plots(
    model,
    X_test,
    y_test: np.ndarray,
    feature_names: List[str],
    save_path: str = 'docs/figures'
) -> dict:
    """
    Create all evaluation plots for a model.
    
    Args:
        model: Trained model pipeline.
        X_test: Test features.
        y_test: True test labels.
        feature_names: List of feature names.
        save_path: Directory to save figures.
        
    Returns:
        Dictionary of figure objects.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    figures = {}
    
    # Feature importance
    try:
        figures['feature_importance'] = plot_feature_importance(
            model, feature_names, save_path=save_path
        )
    except Exception as e:
        logger.warning(f"Could not create feature importance plot: {e}")
    
    # Confusion matrix
    figures['confusion_matrix'] = plot_confusion_matrix(
        y_test, y_pred, save_path=save_path
    )
    
    # ROC and PR curves (only if probabilities available)
    if y_proba is not None:
        figures['roc_curve'] = plot_roc_curve(y_test, y_proba, save_path=save_path)
        figures['pr_curve'] = plot_precision_recall_curve(y_test, y_proba, save_path=save_path)
    
    logger.info(f"Created {len(figures)} evaluation plots in {save_path}")
    
    return figures
