"""
Feature Engineering module for Fake Account Detection.

This module provides the FeatureEngineer class which transforms raw user data
into numerical features suitable for machine learning models.

Classes:
    FeatureEngineer: sklearn-compatible transformer for feature extraction.

Functions:
    predict_sex: Infer gender from name using gender_guesser library.
    to_bool_int: Convert boolean-like values to integers.

Example:
    >>> from feature_engineer import FeatureEngineer
    >>> fe = FeatureEngineer()
    >>> fe.fit(train_df)
    >>> X_train = fe.transform(train_df)
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

import gender_guesser.detector as gender
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

NUMERIC_COLUMNS: List[str] = [
    'statuses_count',
    'followers_count',
    'friends_count',
    'favourites_count',
    'listed_count'
]

FEATURE_NAMES: List[str] = [
    'statuses_count',
    'followers_count',
    'friends_count',
    'favourites_count',
    'listed_count',
    'sex_code',
    'lang_code',
    'tweets_per_day',
    'account_age_days',
    'description_length',
    'default_profile',
    'verified'
]

GENDER_MAPPING: Dict[str, int] = {
    'female': -2,
    'mostly_female': -1,
    'unknown': 0,
    'andy': 0,
    'mostly_male': 1,
    'male': 2
}

# Initialize gender detector globally for efficiency
_detector = gender.Detector(case_sensitive=False)


# =============================================================================
# Helper Functions
# =============================================================================

def predict_sex(name: Union[str, float, None]) -> int:
    """
    Predict gender code from a person's name.
    
    Uses the gender_guesser library to infer gender from the first name,
    then maps it to a numerical code.
    
    Args:
        name: The full name string. Can handle NaN/None values.
        
    Returns:
        int: Gender code ranging from -2 (female) to 2 (male).
            - -2: female
            - -1: mostly_female
            - 0: unknown/andy
            - 1: mostly_male
            - 2: male
            
    Example:
        >>> predict_sex('John Smith')
        2
        >>> predict_sex('Alice Johnson')
        -2
        >>> predict_sex('Unknown123')
        0
    """
    try:
        first_name = str(name).strip().split(' ')[0]
        sex = _detector.get_gender(first_name)
    except Exception:
        sex = 'unknown'
    return GENDER_MAPPING.get(sex, 0)


def to_bool_int(series: pd.Series) -> pd.Series:
    """
    Convert boolean-like column values to integers (0/1).
    
    Handles various representations of boolean values including:
    - String: 'true', 'false', 'True', 'False'
    - Numeric strings: '1', '0', '1.0', '0.0'
    - Actual booleans: True, False
    - NaN/None values (converted to 0)
    
    Args:
        series: A pandas Series containing boolean-like values.
        
    Returns:
        pd.Series: Integer series with values 0 or 1.
        
    Example:
        >>> s = pd.Series(['true', 'false', '1', None])
        >>> to_bool_int(s)
        0    1
        1    0
        2    1
        3    0
        dtype: int64
    """
    bool_map = {'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
    return (
        series.fillna(0)
        .astype(str)
        .str.strip()
        .str.lower()
        .map(bool_map)
        .fillna(0)
        .astype(int)
    )


# =============================================================================
# Feature Engineer Class
# =============================================================================

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for extracting features from user profiles.
    
    This transformer converts raw social media user data into numerical features
    that can be used by machine learning models. It handles:
    - Numeric column normalization
    - Language encoding (fitted on training data)
    - Gender inference from names
    - Activity metrics (tweets per day, account age)
    - Profile characteristics (description length, verified status)
    
    Attributes:
        lang_map (Dict[str, int]): Mapping from language codes to integers.
            Fitted during the fit() method.
        feature_names (List[str]): Names of output features.
    
    Example:
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> 
        >>> pipeline = Pipeline([
        ...     ('fe', FeatureEngineer()),
        ...     ('clf', RandomForestClassifier())
        ... ])
        >>> pipeline.fit(X_train, y_train)
    """
    
    def __init__(self) -> None:
        """Initialize FeatureEngineer with empty language map."""
        self.lang_map: Optional[Dict[str, int]] = None
        self.feature_names: List[str] = FEATURE_NAMES.copy()

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'FeatureEngineer':
        """
        Fit the transformer by learning language mappings from training data.
        
        Args:
            X: Training data as a pandas DataFrame with user profile columns.
            y: Target labels (ignored, present for sklearn compatibility).
            
        Returns:
            self: The fitted transformer instance.
            
        Note:
            The language mapping is learned from the training data to avoid
            data leakage. Unseen languages at transform time are mapped to 'unknown'.
        """
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        if 'lang' in df.columns:
            langs = df['lang'].fillna('unknown').astype(str)
            unique_langs = list(pd.Series(langs).unique())
            self.lang_map = {lang: idx for idx, lang in enumerate(unique_langs)}
            
            # Ensure 'unknown' exists for unseen languages
            if 'unknown' not in self.lang_map:
                self.lang_map['unknown'] = len(self.lang_map)
                
            logger.info(f"Fitted language map with {len(self.lang_map)} languages")
        else:
            self.lang_map = {'unknown': 0}
            
        return self

    def transform(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """
        Transform user data into numerical features.
        
        Args:
            X: Data to transform as a pandas DataFrame.
            
        Returns:
            np.ndarray: 2D array of shape (n_samples, n_features) with
                numerical features.
                
        Raises:
            ValueError: If the transformer hasn't been fitted yet.
        """
        if self.lang_map is None:
            raise ValueError("FeatureEngineer must be fitted before transform")
            
        df = X.copy()
        
        # Process numeric columns
        df = self._process_numeric_columns(df)
        
        # Process categorical columns
        df = self._process_language(df)
        df = self._process_gender(df)
        
        # Process derived features
        df = self._process_activity_features(df)
        df = self._process_description(df)
        df = self._process_boolean_flags(df)
        
        # Select and return features
        features = [f for f in self.feature_names if f in df.columns]
        return df[features].fillna(0).values

    def _process_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate numeric columns."""
        for col in NUMERIC_COLUMNS:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
        return df

    def _process_language(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode language column using fitted mapping."""
        if 'lang' in df.columns:
            df['lang'] = df['lang'].fillna('unknown').astype(str)
            unknown_code = self.lang_map.get('unknown', 0)
            df['lang_code'] = df['lang'].map(self.lang_map).fillna(unknown_code).astype(int)
        else:
            df['lang_code'] = 0
        return df

    def _process_gender(self, df: pd.DataFrame) -> pd.DataFrame:
        """Infer gender from name column."""
        if 'name' in df.columns:
            df['sex_code'] = df['name'].apply(predict_sex)
        else:
            df['sex_code'] = 0
        return df

    def _process_activity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate activity-based features (account age, tweets per day)."""
        if 'created_at' in df.columns:
            created = pd.to_datetime(df['created_at'], errors='coerce')
            
            # Handle timezone-aware datetimes
            try:
                if created.dt.tz is not None:
                    created = created.dt.tz_convert(None)
            except Exception:
                pass
                
            now = pd.Timestamp.now()
            df['account_age_days'] = (now - created).dt.total_seconds().div(3600 * 24).fillna(0)
            
            # Calculate tweets per day (avoid division by zero)
            age_for_division = df['account_age_days'].replace(0, np.nan)
            df['tweets_per_day'] = df['statuses_count'] / age_for_division
            df['tweets_per_day'] = df['tweets_per_day'].replace([np.inf, -np.inf], 0).fillna(0)
        else:
            df['account_age_days'] = 0
            df['tweets_per_day'] = 0
        return df

    def _process_description(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate description length."""
        if 'description' in df.columns:
            df['description_length'] = df['description'].fillna('').astype(str).apply(len)
        else:
            df['description_length'] = 0
        return df

    def _process_boolean_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process boolean flag columns."""
        for col in ['default_profile', 'verified']:
            if col in df.columns:
                df[col] = to_bool_int(df[col])
            else:
                df[col] = 0
        return df

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names for transformation.
        
        Args:
            input_features: Ignored, present for sklearn compatibility.
            
        Returns:
            List[str]: Names of output features.
        """
        return self.feature_names

    def export_lang_map(self, path: str) -> None:
        """
        Export the fitted language mapping to a JSON file.
        
        Args:
            path: File path to save the JSON mapping.
            
        Raises:
            ValueError: If the transformer hasn't been fitted yet.
        """
        if self.lang_map is None:
            raise ValueError("lang_map is not fitted - call fit() first")
            
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.lang_map, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Exported language map to {path}")
