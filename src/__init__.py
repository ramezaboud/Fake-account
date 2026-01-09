"""
Fake Account Detector - Source Package.

This package provides tools for detecting fake social media accounts
using machine learning.

Modules:
    feature_engineer: Feature extraction and transformation.
    train: Model training and evaluation.
    visualize: Visualization and plotting functions.
"""

__version__ = '1.0.0'

# Lazy imports to avoid circular dependencies
# Users can import directly from submodules:
#   from src.feature_engineer import FeatureEngineer
#   from src.train import train_model
#   from src.visualize import plot_confusion_matrix
