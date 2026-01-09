# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-12-09

### Added
- **XGBoost Support**: Added XGBoost as an alternative classifier
- **Model Comparison Script**: `scripts/compare_models.py` for comparing multiple models
- **Visualization Module**: `src/visualize.py` with functions for:
  - Feature importance plots
  - Confusion matrix heatmaps
  - ROC curves
  - Precision-Recall curves
  - Model comparison charts
- **Evaluation Figures**: Saved plots in `docs/figures/`
- **Comprehensive Docstrings**: All functions now have detailed docstrings
- **Type Hints**: Added type hints throughout the codebase
- **Logging**: Replaced print statements with proper logging
- **CLI Arguments**: `train.py` now supports command-line arguments
- **Configuration File**: Added `config/config.yaml`
- **Unit Tests**: Added test structure in `tests/` directory
- **CHANGELOG.md**: This file

### Changed
- **Project Structure**: Reorganized with proper package structure
- **README.md**: Updated with model comparison results and visualizations
- **requirements.txt**: Added xgboost dependency
- **pyproject.toml**: Enhanced with optional dependencies and tool configs
- **feature_engineer.py**: Refactored with constants and better organization
- **train.py**: Modularized into reusable functions

### Fixed
- Data leakage issue by using Pipeline for feature engineering
- Import paths for cross-module usage

## [1.0.0] - 2025-12-08

### Added
- Initial release
- RandomForest classifier with GridSearchCV
- Custom FeatureEngineer transformer
- Feature extraction from user profiles:
  - Activity metrics (statuses, favorites, etc.)
  - Network metrics (followers, friends, listed)
  - Profile characteristics (description, default profile, verified)
  - Gender inference from name
  - Language encoding
  - Account age calculation
- Streamlit web application
- Jupyter notebook for exploration
- Basic inference script

### Technical Details
- scikit-learn Pipeline architecture
- StandardScaler for feature normalization
- class_weight='balanced' for imbalanced data handling
- 5-fold cross-validation
- GridSearchCV hyperparameter tuning

---

## Roadmap

### Planned for v1.2.0
- [ ] Deep learning model (BERT for text analysis)
- [ ] API endpoint with FastAPI
- [ ] Docker containerization
- [ ] MLflow experiment tracking
- [ ] Real-time prediction service

### Planned for v1.3.0
- [ ] Graph neural network for network analysis
- [ ] Multi-language support
- [ ] Active learning for continuous improvement
- [ ] Explainability with SHAP values
