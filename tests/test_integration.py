"""
Integration tests for the Fake Account Detection pipeline.

These tests verify the end-to-end functionality of the ML pipeline,
ensuring all components work together correctly.
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


class TestPipelineIntegration:
    """Integration tests for the full ML pipeline."""
    
    def test_full_pipeline_fit_predict(self, sample_user_data):
        """Test that the full pipeline can fit and predict."""
        # Create labels
        y = np.array([0, 1, 0])
        
        # Create pipeline
        pipeline = Pipeline([
            ('fe', FeatureEngineer()),
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Fit pipeline
        pipeline.fit(sample_user_data, y)
        
        # Predict
        predictions = pipeline.predict(sample_user_data)
        
        assert len(predictions) == len(sample_user_data)
        assert all(p in [0, 1] for p in predictions)
    
    def test_pipeline_predict_proba(self, sample_user_data):
        """Test that pipeline can return probability predictions."""
        y = np.array([0, 1, 0])
        
        pipeline = Pipeline([
            ('fe', FeatureEngineer()),
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        pipeline.fit(sample_user_data, y)
        probas = pipeline.predict_proba(sample_user_data)
        
        assert probas.shape == (len(sample_user_data), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
    
    def test_pipeline_with_missing_values(self):
        """Test pipeline handles missing values gracefully."""
        data = pd.DataFrame({
            'name': ['John Doe', None, 'Jane'],
            'screen_name': ['john', 'unknown', 'jane'],
            'statuses_count': [100, np.nan, 200],
            'followers_count': [50, 100, np.nan],
            'friends_count': [30, 40, 50],
            'favourites_count': [10, 20, 30],
            'listed_count': [5, np.nan, 15],
            'created_at': ['2020-01-01', '2021-06-15', '2019-03-20'],
            'description': ['Hello', None, 'World'],
            'lang': ['en', 'en', None],
            'default_profile': [True, False, True],
            'verified': [False, False, True]
        })
        y = np.array([0, 1, 0])
        
        pipeline = Pipeline([
            ('fe', FeatureEngineer()),
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Should not raise any errors
        pipeline.fit(data, y)
        predictions = pipeline.predict(data)
        
        assert len(predictions) == 3
    
    def test_pipeline_consistency(self, sample_user_data):
        """Test that pipeline gives consistent results."""
        y = np.array([0, 1, 0])
        
        pipeline = Pipeline([
            ('fe', FeatureEngineer()),
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        pipeline.fit(sample_user_data, y)
        
        # Multiple predictions should be identical
        pred1 = pipeline.predict(sample_user_data)
        pred2 = pipeline.predict(sample_user_data)
        
        np.testing.assert_array_equal(pred1, pred2)


class TestFeatureEngineerIntegration:
    """Integration tests for FeatureEngineer with real-world scenarios."""
    
    def test_feature_engineer_output_is_numeric(self, sample_user_data):
        """Verify all output features are numeric."""
        fe = FeatureEngineer()
        fe.fit(sample_user_data)
        result = fe.transform(sample_user_data)
        
        assert result.dtype in [np.float64, np.float32, np.int64, np.int32]
    
    def test_feature_engineer_reproducibility(self, sample_user_data):
        """Test that FeatureEngineer gives reproducible results."""
        fe1 = FeatureEngineer()
        fe2 = FeatureEngineer()
        
        fe1.fit(sample_user_data)
        fe2.fit(sample_user_data)
        
        result1 = fe1.transform(sample_user_data)
        result2 = fe2.transform(sample_user_data)
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_large_dataset_handling(self):
        """Test pipeline can handle larger datasets."""
        n_samples = 1000
        
        data = pd.DataFrame({
            'name': [f'User {i}' for i in range(n_samples)],
            'screen_name': [f'user{i}' for i in range(n_samples)],
            'statuses_count': np.random.randint(0, 10000, n_samples),
            'followers_count': np.random.randint(0, 5000, n_samples),
            'friends_count': np.random.randint(0, 2000, n_samples),
            'favourites_count': np.random.randint(0, 3000, n_samples),
            'listed_count': np.random.randint(0, 100, n_samples),
            'created_at': ['2020-01-01'] * n_samples,
            'description': ['Test description'] * n_samples,
            'lang': ['en'] * n_samples,
            'default_profile': [False] * n_samples,
            'verified': [False] * n_samples
        })
        y = np.random.randint(0, 2, n_samples)
        
        pipeline = Pipeline([
            ('fe', FeatureEngineer()),
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Should complete without errors
        pipeline.fit(data, y)
        predictions = pipeline.predict(data)
        
        assert len(predictions) == n_samples
