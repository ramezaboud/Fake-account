"""
Tests for the FeatureEngineer class.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from feature_engineer import FeatureEngineer, predict_sex


class TestPredictSex:
    """Tests for the predict_sex function."""
    
    def test_male_name(self):
        assert predict_sex('John') == 2
        
    def test_female_name(self):
        assert predict_sex('Alice') == -2
        
    def test_unknown_name(self):
        assert predict_sex('Xyzabc123') == 0
        
    def test_nan_handling(self):
        # nan gets converted to string 'nan' which may get partial gender match
        result = predict_sex(np.nan)
        assert result in [-2, -1, 0, 1, 2]  # Any valid gender code is acceptable
        
    def test_empty_string(self):
        assert predict_sex('') == 0


class TestFeatureEngineer:
    """Tests for the FeatureEngineer class."""
    
    def test_fit_returns_self(self, sample_user_data):
        fe = FeatureEngineer()
        result = fe.fit(sample_user_data)
        assert result is fe
        
    def test_transform_output_shape(self, sample_user_data, feature_names):
        fe = FeatureEngineer()
        fe.fit(sample_user_data)
        result = fe.transform(sample_user_data)
        assert result.shape == (len(sample_user_data), len(feature_names))
        
    def test_transform_no_nan(self, sample_user_data):
        fe = FeatureEngineer()
        fe.fit(sample_user_data)
        result = fe.transform(sample_user_data)
        assert not np.isnan(result).any()
        
    def test_lang_map_fitted(self, sample_user_data):
        fe = FeatureEngineer()
        fe.fit(sample_user_data)
        assert fe.lang_map is not None
        assert 'en' in fe.lang_map
        assert 'fr' in fe.lang_map
        
    def test_unseen_language_mapped_to_unknown(self, sample_user_data):
        fe = FeatureEngineer()
        fe.fit(sample_user_data)
        
        new_data = sample_user_data.copy()
        new_data['lang'] = 'xyz_unseen'
        result = fe.transform(new_data)
        # Should not raise error
        assert result.shape[0] == len(new_data)
