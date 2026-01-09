# Changelog# Changelog



All notable changes to this project will be documented in this file.All notable changes to this project will be documented in this file.



The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),

and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).



## [1.0.0] - 2026-01-09## [1.1.0] - 2025-12-09



### Added### Added

- **RandomForest Pipeline**: Production-ready ML pipeline for fake account detection- **XGBoost Support**: Added XGBoost as an alternative classifier

- **FastAPI REST API**: Full API implementation with endpoints:- **Model Comparison Script**: `scripts/compare_models.py` for comparing multiple models

  - `POST /predict` - Single user prediction- **Visualization Module**: `src/visualize.py` with functions for:

  - `POST /predict/batch` - Batch predictions  - Feature importance plots

  - `GET /health` - Health check  - Confusion matrix heatmaps

  - `GET /model/info` - Model information  - ROC curves

- **Custom Feature Engineering**: 29 engineered features including:  - Precision-Recall curves

  - Activity metrics (tweets per day, account age)  - Model comparison charts

  - Network metrics (followers/friends ratio)- **Evaluation Figures**: Saved plots in `docs/figures/`

  - Profile characteristics (description length, verified status)- **Comprehensive Docstrings**: All functions now have detailed docstrings

  - Gender inference from name- **Type Hints**: Added type hints throughout the codebase

  - Suspicious pattern detection- **Logging**: Replaced print statements with proper logging

- **Visualization Module**: `src/visualize.py` with plotting functions:- **CLI Arguments**: `train.py` now supports command-line arguments

  - Confusion matrix- **Configuration File**: Added `config/config.yaml`

  - ROC curve- **Unit Tests**: Added test structure in `tests/` directory

  - Precision-Recall curve- **CHANGELOG.md**: This file

  - Feature importance

  - Threshold analysis### Changed

- **Evaluation Plots**: Generated plots in `docs/figures/`- **Project Structure**: Reorganized with proper package structure

- **Configurable Decision Threshold**: Default 0.445 for balanced precision/recall- **README.md**: Updated with model comparison results and visualizations

- **Comprehensive Tests**: Unit tests for feature engineering and model- **requirements.txt**: Added xgboost dependency

- **CI Pipeline**: GitHub Actions for automated testing- **pyproject.toml**: Enhanced with optional dependencies and tool configs

- **Documentation**: Full README, API docs, and contributing guidelines- **feature_engineer.py**: Refactored with constants and better organization

- **train.py**: Modularized into reusable functions

### Model Performance

- Accuracy: 92.07%### Fixed

- ROC AUC: 98.47%- Data leakage issue by using Pipeline for feature engineering

- F1 Score: 92.12%- Import paths for cross-module usage

- Precision: 91.57%

- Recall: 92.67%## [1.0.0] - 2025-12-08



### Technical Details### Added

- scikit-learn Pipeline architecture- Initial release

- StandardScaler for feature normalization- RandomForest classifier with GridSearchCV

- class_weight='balanced' for imbalanced data handling- Custom FeatureEngineer transformer

- Reference time support for reproducible predictions- Feature extraction from user profiles:

  - Activity metrics (statuses, favorites, etc.)

---  - Network metrics (followers, friends, listed)

  - Profile characteristics (description, default profile, verified)

## Roadmap  - Gender inference from name

  - Language encoding

### Planned for v1.1.0  - Account age calculation

- [ ] Docker containerization- Streamlit web application

- [ ] MLflow experiment tracking- Jupyter notebook for exploration

- [ ] SHAP explainability integration- Basic inference script

- [ ] Multi-language support improvements

### Technical Details

### Planned for v2.0.0- scikit-learn Pipeline architecture

- [ ] Deep learning model (BERT for text analysis)- StandardScaler for feature normalization

- [ ] Graph neural network for network analysis- class_weight='balanced' for imbalanced data handling

- [ ] Real-time streaming predictions- 5-fold cross-validation

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
