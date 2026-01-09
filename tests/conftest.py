"""
Pytest configuration and fixtures for the test suite.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture
def sample_user_data():
    """Create sample user data for testing."""
    return pd.DataFrame([
        {
            'statuses_count': 100, 'followers_count': 50, 'friends_count': 20,
            'favourites_count': 5, 'listed_count': 1, 'name': 'John Doe',
            'lang': 'en', 'created_at': '2020-01-01 12:00:00',
            'description': 'Hello world!', 'default_profile': 0, 'verified': 0
        },
        {
            'statuses_count': 50, 'followers_count': 5, 'friends_count': 10,
            'favourites_count': 3, 'listed_count': 0, 'name': 'Jane Smith',
            'lang': 'fr', 'created_at': '2021-05-05 08:30:00',
            'description': '', 'default_profile': 1, 'verified': 0
        },
        {
            'statuses_count': 1000, 'followers_count': 500, 'friends_count': 200,
            'favourites_count': 50, 'listed_count': 10, 'name': 'Bot Account',
            'lang': 'en', 'created_at': '2023-01-01 00:00:00',
            'description': 'Follow me!', 'default_profile': 1, 'verified': 0
        }
    ])


@pytest.fixture
def sample_labels():
    """Create sample labels for testing."""
    return np.array([0, 0, 1])  # Two real, one fake


@pytest.fixture
def feature_names():
    """Return the expected feature names."""
    return [
        # Basic counts
        'statuses_count', 'followers_count', 'friends_count', 
        'favourites_count', 'listed_count',
        
        # Encoded features
        'sex_code', 'lang_code',
        
        # Activity metrics
        'tweets_per_day', 'account_age_days',
        
        # Profile features
        'description_length', 'default_profile', 'verified',
        
        # Ratio features
        'followers_to_friends_ratio',
        'friends_to_followers_ratio',
        'engagement_ratio',
        'reputation_score',
        'listed_ratio',
        'favorites_per_tweet',
        
        # Name features
        'is_new_account',
        'has_description',
        'has_url',
        'name_length',
        'screen_name_length',
        'has_digits_in_name',
        'digit_ratio_in_screen_name',
        
        # Suspicious patterns
        'zero_followers',
        'zero_statuses',
        'following_many_no_followers',
        'high_friend_rate',
    ]
