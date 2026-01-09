"""
Feature Engineering Module for Fake Account Detection.

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
import re
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

# Feature names including all advanced features for fake detection
FEATURE_NAMES: List[str] = [
    # Basic counts
    'statuses_count', 'followers_count', 'friends_count', 
    'favourites_count', 'listed_count',
    
    # Encoded features
    'sex_code', 'lang_code',
    
    # Activity metrics
    'tweets_per_day', 'account_age_days',
    
    # Profile features
    'description_length', 'default_profile', 'verified',
    
    # === RATIO FEATURES (critical for fake detection) ===
    'followers_to_friends_ratio',      # Real users: ~1.0, Bots: very low
    'friends_to_followers_ratio',      # Inverse ratio
    'engagement_ratio',                # How much they interact vs post
    
    # Reputation features
    'reputation_score',                # followers / (followers + friends)
    'listed_ratio',                    # listed_count / followers
    
    # Activity patterns
    'favorites_per_tweet',             # Real users like more per tweet
    'is_new_account',                  # Account < 30 days old
    'has_description',                 # Bot accounts often have no description
    'has_url',                         # Real accounts often have URLs
    'name_length',                     # Bot names are often short or random
    'screen_name_length',              # Bot screen names patterns
    'has_digits_in_name',              # Bot names often have numbers
    'digit_ratio_in_screen_name',      # Ratio of digits in screen name
    
    # Suspicious patterns
    'zero_followers',                  # Has zero followers
    'zero_statuses',                   # Never posted
    'following_many_no_followers',     # Following many but no one follows back
    'high_friend_rate',                # Added many friends quickly
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
    - Ratio-based features for fake account detection
    - Suspicious pattern detection
    
    Attributes:
        lang_map (Dict[str, int]): Mapping from language codes to integers.
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
    
    def __init__(self, reference_time: Optional[Union[str, pd.Timestamp]] = None) -> None:
        """Initialize FeatureEngineer with empty language map.

        Args:
            reference_time: Optional fixed reference time to compute account age
                (can be a pandas Timestamp or ISO date string). If provided,
                this fixed time will be used when computing `account_age_days`
                and `tweets_per_day`. If None, the transformer will default to
                using the maximum `created_at` value from the input batch; if
                that is missing it falls back to the current time.
        """
        self.lang_map: Optional[Dict[str, int]] = None
        self.feature_names: List[str] = FEATURE_NAMES.copy()
        # Normalize reference_time to a pandas Timestamp when provided
        if reference_time is None:
            self.reference_time: Optional[pd.Timestamp] = None
        else:
            try:
                self.reference_time = pd.to_datetime(reference_time)
            except Exception:
                # keep None if parsing fails
                self.reference_time = None

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
        
        # Process advanced features for fake detection
        df = self._process_ratio_features(df)
        df = self._process_name_features(df)
        df = self._process_suspicious_patterns(df)
        
        # Select and return features
        features = [f for f in self.feature_names if f in df.columns]
        logger.debug(f"Returning {len(features)} features")
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
                
            # Determine reference time for age calculation. Prefer an explicit
            # reference_time configured on the transformer (production-grade
            # stable reference). Otherwise use the max created_at in the batch
            # to preserve relative ages; finally fall back to current time.
            if getattr(self, 'reference_time', None) is not None:
                reference = self.reference_time
            else:
                # use batch max created timestamp if available
                if created.notna().any():
                    reference = created.max()
                else:
                    reference = pd.Timestamp.now()

            df['account_age_days'] = (reference - created).dt.total_seconds().div(3600 * 24).fillna(0)
            
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

    def _process_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ratio-based features that are strong indicators of fake accounts.
        
        Bots typically have: many friends, few followers, low engagement
        Real users typically have: balanced ratios, higher engagement
        """
        # Followers to Friends ratio
        # Real users: usually close to 1 or higher
        # Bots: very low (they follow many but few follow back)
        df['followers_to_friends_ratio'] = np.where(
            df['friends_count'] > 0,
            df['followers_count'] / df['friends_count'],
            df['followers_count']  # If no friends, just use followers count
        )
        # Cap at 100 to avoid extreme outliers
        df['followers_to_friends_ratio'] = df['followers_to_friends_ratio'].clip(0, 100)
        
        # Friends to Followers ratio (inverse - high value = suspicious)
        df['friends_to_followers_ratio'] = np.where(
            df['followers_count'] > 0,
            df['friends_count'] / df['followers_count'],
            df['friends_count']
        )
        df['friends_to_followers_ratio'] = df['friends_to_followers_ratio'].clip(0, 100)
        
        # Engagement ratio: favorites / statuses
        # Real users engage with content, bots don't
        df['engagement_ratio'] = np.where(
            df['statuses_count'] > 0,
            df['favourites_count'] / df['statuses_count'],
            0
        )
        df['engagement_ratio'] = df['engagement_ratio'].clip(0, 100)
        
        # Reputation score: followers / (followers + friends)
        # High value (close to 1) = influential account
        # Low value (close to 0) = follows many, followed by few = suspicious
        total = df['followers_count'] + df['friends_count']
        df['reputation_score'] = np.where(
            total > 0,
            df['followers_count'] / total,
            0.5  # Default to neutral if no activity
        )
        
        # Listed ratio: how often this account is added to lists per follower
        # Valuable accounts get listed more often
        df['listed_ratio'] = np.where(
            df['followers_count'] > 0,
            df['listed_count'] / df['followers_count'],
            0
        )
        df['listed_ratio'] = df['listed_ratio'].clip(0, 10)
        
        # Favorites per tweet
        df['favorites_per_tweet'] = np.where(
            df['statuses_count'] > 0,
            df['favourites_count'] / df['statuses_count'],
            0
        )
        df['favorites_per_tweet'] = df['favorites_per_tweet'].clip(0, 100)
        
        return df

    def _process_name_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from name and screen_name.
        
        Bot patterns:
        - Short or very long names
        - Names with many digits
        - Random-looking screen names
        """
        # Name length
        if 'name' in df.columns:
            df['name_length'] = df['name'].fillna('').astype(str).apply(len)
            
            # Check if name has digits (bot indicator)
            df['has_digits_in_name'] = df['name'].fillna('').astype(str).apply(
                lambda x: 1 if re.search(r'\d', x) else 0
            )
        else:
            df['name_length'] = 0
            df['has_digits_in_name'] = 0
        
        # Screen name features
        if 'screen_name' in df.columns:
            df['screen_name_length'] = df['screen_name'].fillna('').astype(str).apply(len)
            
            # Ratio of digits in screen name (bots often have names like "user12345678")
            def digit_ratio(s):
                s = str(s)
                if len(s) == 0:
                    return 0
                digits = sum(c.isdigit() for c in s)
                return digits / len(s)
            
            df['digit_ratio_in_screen_name'] = df['screen_name'].fillna('').apply(digit_ratio)
        else:
            df['screen_name_length'] = 0
            df['digit_ratio_in_screen_name'] = 0
        
        # Has URL in profile (real users often have personal/business URLs)
        if 'url' in df.columns:
            df['has_url'] = df['url'].fillna('').astype(str).apply(
                lambda x: 1 if len(x) > 0 and x.lower() not in ['nan', 'null', 'none', ''] else 0
            )
        else:
            df['has_url'] = 0
        
        # Has description
        if 'description' in df.columns:
            df['has_description'] = df['description'].fillna('').astype(str).apply(
                lambda x: 1 if len(x.strip()) > 0 else 0
            )
        else:
            df['has_description'] = 0
        
        return df

    def _process_suspicious_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect suspicious patterns that indicate fake accounts.
        """
        # Is this a new account? (less than 30 days old)
        if 'account_age_days' in df.columns:
            df['is_new_account'] = (df['account_age_days'] < 30).astype(int)
        else:
            df['is_new_account'] = 0
        
        # Zero followers (suspicious for accounts that follow many)
        df['zero_followers'] = (df['followers_count'] == 0).astype(int)
        
        # Zero statuses (never posted - often a bot that just follows)
        df['zero_statuses'] = (df['statuses_count'] == 0).astype(int)
        
        # Following many but no followers (classic bot pattern)
        # True if: friends > 100 AND followers < 10
        df['following_many_no_followers'] = (
            (df['friends_count'] > 100) & (df['followers_count'] < 10)
        ).astype(int)
        
        # High friend rate: added many friends quickly
        # friends_count / account_age_days
        if 'account_age_days' in df.columns:
            df['high_friend_rate'] = np.where(
                df['account_age_days'] > 0,
                df['friends_count'] / df['account_age_days'],
                df['friends_count']  # If account age is 0, use raw count
            )
            # Cap at reasonable value
            df['high_friend_rate'] = df['high_friend_rate'].clip(0, 100)
        else:
            df['high_friend_rate'] = 0
        
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
