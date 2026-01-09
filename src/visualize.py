"""
Visualization Module for Fake Account Detection.

This module provides functions for creating evaluation plots and visualizations
for the machine learning models.

Functions:
    plot_feature_importance: Plot feature importance from a trained model.
    plot_confusion_matrix: Plot confusion matrix heatmap.
    plot_roc_curve: Plot ROC curve with AUC score.
    plot_precision_recall_curve: Plot Precision-Recall curve.
    plot_model_comparison: Compare multiple models' performance.
    create_all_evaluation_plots: Generate all evaluation plots at once.

Example:
    >>> from visualize import plot_confusion_matrix, plot_roc_curve
    >>> plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png')
    >>> plot_roc_curve(y_true, y_proba, save_path='roc_curve.png')
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
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

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Default figure directory
DEFAULT_FIGURES_DIR = Path(__file__).parent.parent / "docs" / "figures"


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    title: str = "Feature Importance",
    top_n: int = 20,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot feature importance as a horizontal bar chart.
    
    Args:
        feature_names: List of feature names.
        importances: Array of feature importance values.
        title: Plot title.
        top_n: Number of top features to display.
        save_path: Path to save the figure. If None, figure is not saved.
        figsize: Figure size as (width, height).
        
    Returns:
        matplotlib Figure object.
        
    Example:
        >>> importances = model.named_steps['clf'].feature_importances_
        >>> plot_feature_importance(FEATURE_NAMES, importances)
    """
    # Sort features by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bars
    y_pos = np.arange(len(top_features))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
    
    ax.barh(y_pos, top_importances, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()  # Top feature at the top
    ax.set_xlabel('Importance')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, v in enumerate(top_importances):
        ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues"
) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Class labels for display. Defaults to ['Genuine', 'Fake'].
        title: Plot title.
        save_path: Path to save the figure.
        figsize: Figure size.
        cmap: Colormap for the heatmap.
        
    Returns:
        matplotlib Figure object.
        
    Example:
        >>> plot_confusion_matrix(y_test, y_pred, save_path='cm.png')
    """
    if labels is None:
        labels = ['Genuine (0)', 'Fake (1)']
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={'size': 14}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add percentages
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = cm[i, j] / total * 100
            current_text = ax.texts[i * cm.shape[1] + j]
            current_text.set_text(f'{cm[i, j]}\n({percentage:.1f}%)')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {save_path}")
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot ROC curve with AUC score.
    
    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for the positive class.
        title: Plot title.
        save_path: Path to save the figure.
        figsize: Figure size.
        
    Returns:
        matplotlib Figure object.
        
    Example:
        >>> y_proba = model.predict_proba(X_test)[:, 1]
        >>> plot_roc_curve(y_test, y_proba)
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random Classifier')
    
    # Fill area under curve
    ax.fill_between(fpr, tpr, alpha=0.3, color='darkorange')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"ROC curve plot saved to {save_path}")
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot Precision-Recall curve with Average Precision score.
    
    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for the positive class.
        title: Plot title.
        save_path: Path to save the figure.
        figsize: Figure size.
        
    Returns:
        matplotlib Figure object.
        
    Example:
        >>> y_proba = model.predict_proba(X_test)[:, 1]
        >>> plot_precision_recall_curve(y_test, y_proba)
    """
    # Compute Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot PR curve
    ax.plot(recall, precision, color='green', lw=2,
            label=f'PR curve (AP = {avg_precision:.3f})')
    
    # Fill area under curve
    ax.fill_between(recall, precision, alpha=0.3, color='green')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Precision-Recall curve plot saved to {save_path}")
    
    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    title: str = "Model Comparison",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot comparison of multiple models across different metrics.
    
    Args:
        results: Dictionary mapping model names to their metrics.
                 Example: {'RandomForest': {'accuracy': 0.95, 'f1': 0.93}, ...}
        metrics: List of metrics to compare. If None, uses all available metrics.
        title: Plot title.
        save_path: Path to save the figure.
        figsize: Figure size.
        
    Returns:
        matplotlib Figure object.
        
    Example:
        >>> results = {
        ...     'RandomForest': {'accuracy': 0.95, 'f1': 0.93, 'roc_auc': 0.97},
        ...     'XGBoost': {'accuracy': 0.94, 'f1': 0.92, 'roc_auc': 0.96}
        ... }
        >>> plot_model_comparison(results)
    """
    if metrics is None:
        # Get all metrics from the first model
        first_model = list(results.keys())[0]
        metrics = list(results[first_model].keys())
    
    # Prepare data
    model_names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bars for each model
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        values = [results[model_name].get(m, 0) for m in metrics]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name, color=color)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")
    
    return fig


def create_all_evaluation_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    feature_names: Optional[List[str]] = None,
    feature_importances: Optional[np.ndarray] = None,
    model_name: str = "Model",
    output_dir: Optional[Union[str, Path]] = None,
    show_plots: bool = False
) -> Dict[str, plt.Figure]:
    """
    Generate all evaluation plots for a model.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities for the positive class.
        feature_names: List of feature names (for feature importance plot).
        feature_importances: Feature importance values.
        model_name: Name of the model for plot titles.
        output_dir: Directory to save plots. If None, uses default figures directory.
        show_plots: Whether to display plots.
        
    Returns:
        Dictionary mapping plot names to Figure objects.
        
    Example:
        >>> figs = create_all_evaluation_plots(
        ...     y_test, y_pred, y_proba,
        ...     feature_names=FEATURE_NAMES,
        ...     feature_importances=model.named_steps['clf'].feature_importances_,
        ...     model_name='RandomForest',
        ...     output_dir='docs/figures'
        ... )
    """
    if output_dir is None:
        output_dir = DEFAULT_FIGURES_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    # Confusion Matrix
    logger.info("Creating confusion matrix plot...")
    figures['confusion_matrix'] = plot_confusion_matrix(
        y_true, y_pred,
        title=f"{model_name} - Confusion Matrix",
        save_path=output_dir / f"{model_name.lower()}_confusion_matrix.png"
    )
    
    # ROC Curve
    logger.info("Creating ROC curve plot...")
    figures['roc_curve'] = plot_roc_curve(
        y_true, y_proba,
        title=f"{model_name} - ROC Curve",
        save_path=output_dir / f"{model_name.lower()}_roc_curve.png"
    )
    
    # Precision-Recall Curve
    logger.info("Creating Precision-Recall curve plot...")
    figures['pr_curve'] = plot_precision_recall_curve(
        y_true, y_proba,
        title=f"{model_name} - Precision-Recall Curve",
        save_path=output_dir / f"{model_name.lower()}_pr_curve.png"
    )
    
    # Feature Importance (if provided)
    if feature_names is not None and feature_importances is not None:
        logger.info("Creating feature importance plot...")
        figures['feature_importance'] = plot_feature_importance(
            feature_names, feature_importances,
            title=f"{model_name} - Feature Importance",
            save_path=output_dir / f"{model_name.lower()}_feature_importance.png"
        )
    
    if show_plots:
        plt.show()
    else:
        plt.close('all')
    
    logger.info(f"All evaluation plots saved to {output_dir}")
    return figures


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: List[float] = None,
    title: str = "Threshold Analysis",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot how precision, recall, and F1 change with different thresholds.
    
    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for the positive class.
        thresholds: List of thresholds to evaluate. If None, uses 0.1 to 0.9.
        title: Plot title.
        save_path: Path to save the figure.
        figsize: Figure size.
        
    Returns:
        matplotlib Figure object.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)
    
    precisions = []
    recalls = []
    f1_scores = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(thresholds, precisions, 'b-', lw=2, label='Precision')
    ax.plot(thresholds, recalls, 'g-', lw=2, label='Recall')
    ax.plot(thresholds, f1_scores, 'r-', lw=2, label='F1 Score')
    
    # Find best F1 threshold
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    ax.axvline(x=best_thresh, color='gray', linestyle='--', 
               label=f'Best F1 Threshold ({best_thresh:.2f})')
    
    ax.set_xlabel('Decision Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Threshold analysis plot saved to {save_path}")
    
    return fig
