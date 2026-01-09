"""
Tests for the model training and prediction pipeline.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from feature_engineer import FeatureEngineer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


class TestPipeline:
    """Tests for the ML pipeline."""
    
    def test_pipeline_creation(self):
        """Test that pipeline can be created."""
        pipeline = Pipeline([
            ('fe', FeatureEngineer()),
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(random_state=42, n_estimators=10))
        ])
        assert pipeline is not None
        
    def test_pipeline_fit(self, sample_user_data, sample_labels):
        """Test that pipeline can fit data."""
        pipeline = Pipeline([
            ('fe', FeatureEngineer()),
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(random_state=42, n_estimators=10))
        ])
        pipeline.fit(sample_user_data, sample_labels)
        assert hasattr(pipeline, 'predict')
        
    def test_pipeline_predict(self, sample_user_data, sample_labels):
        """Test that pipeline can make predictions."""
        pipeline = Pipeline([
            ('fe', FeatureEngineer()),
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(random_state=42, n_estimators=10))
        ])
        pipeline.fit(sample_user_data, sample_labels)
        predictions = pipeline.predict(sample_user_data)
        assert len(predictions) == len(sample_user_data)
        assert all(p in [0, 1] for p in predictions)
        
    def test_pipeline_predict_proba(self, sample_user_data, sample_labels):
        """Test that pipeline can output probabilities."""
        pipeline = Pipeline([
            ('fe', FeatureEngineer()),
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(random_state=42, n_estimators=10))
        ])
        pipeline.fit(sample_user_data, sample_labels)
        probas = pipeline.predict_proba(sample_user_data)
        assert probas.shape == (len(sample_user_data), 2)
        assert all(0 <= p <= 1 for row in probas for p in row)
