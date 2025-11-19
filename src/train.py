import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from feature_engineer import FeatureEngineer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


# Load data
real = pd.read_csv('data/users.csv')
fake = pd.read_csv('data/fusers.csv')
df_all = pd.concat([real, fake], ignore_index=True, sort=False)
y = np.array([1]*len(real) + [0]*len(fake))

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df_all, y, test_size=0.4, random_state=44, stratify=y)

# pipeline
pipeline = Pipeline([
    ('fe', FeatureEngineer()),
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])

# Hyperparameter tuning
param_dist = {
    'clf__n_estimators': [50, 100, 200, 400],
    'clf__max_depth': [None, 10, 20, 30],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4],
    'clf__max_features': ['sqrt', 'log2', None],
}

rs = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=24, cv=5,
                        n_jobs=-1, scoring='roc_auc', random_state=42, verbose=1)
rs.fit(X_train, y_train)
final_pipeline = rs.best_estimator_

print("Best Hyperparameters:", rs.best_params_)
y_pred = final_pipeline.predict(X_test)
y_proba = final_pipeline.predict_proba(X_test)[:, 1] if hasattr(final_pipeline, 'predict_proba') else None
print('Test Accuracy:', accuracy_score(y_test, y_pred))
if y_proba is not None:
    print('Test ROC AUC:', roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred, target_names=['Fake','Real']))

from pathlib import Path
models_dir = Path('models')
models_dir.mkdir(parents=True, exist_ok=True)
out_path = models_dir / 'randomforest_pipeline.joblib'
joblib.dump(final_pipeline, out_path)
print(f"Saved full pipeline to {out_path}")
