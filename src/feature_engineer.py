import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import gender_guesser.detector as gender
import json
from typing import Optional

detector = gender.Detector(case_sensitive=False)

def predict_sex(name: str) -> int:
    try:
        first_name = str(name).strip().split(' ')[0]
        sex = detector.get_gender(first_name)
    except Exception:
        sex = 'unknown'
    mapping = {'female': -2, 'mostly_female': -1, 'unknown': 0, 'andy': 0, 'mostly_male': 1, 'male': 2}
    return mapping.get(sex, 0)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # lang_map: mapping from language string -> integer code
        self.lang_map: Optional[dict] = None

    def fit(self, X: pd.DataFrame, y=None):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if 'lang' in df.columns:
            langs = df['lang'].fillna('unknown').astype(str)
            # preserve observed order of appearance for mapping
            unique_langs = list(pd.Series(langs).unique())
            self.lang_map = {cls: int(i) for i, cls in enumerate(unique_langs)}
            # ensure 'unknown' exists
            if 'unknown' not in self.lang_map:
                self.lang_map['unknown'] = len(self.lang_map)
        else:
            self.lang_map = {'unknown': 0}
        return self

    def transform(self, X: pd.DataFrame):
        df = X.copy()
        # Numeric fill
        numeric_cols = ['statuses_count','followers_count','friends_count','favourites_count','listed_count']
        for c in numeric_cols:
            if c not in df.columns:
                df[c] = 0
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(float)

        # Lang code - use fitted lang_map if present. Map unseen languages to 'unknown'.
        if 'lang' in df.columns:
            df['lang'] = df['lang'].fillna('unknown').astype(str)
            if self.lang_map is not None:
                df['lang_code'] = df['lang'].map(self.lang_map).fillna(self.lang_map.get('unknown', 0)).astype(int)
            else:
                # fallback: create a local map
                langs = df['lang']
                unique_langs = list(pd.Series(langs).unique())
                self.lang_map = {cls: int(i) for i, cls in enumerate(unique_langs)}
                df['lang_code'] = df['lang'].map(self.lang_map).fillna(self.lang_map.get('unknown', 0)).astype(int)
        else:
            df['lang_code'] = 0

        # Sex code
        if 'name' in df.columns:
            df['sex_code'] = df['name'].apply(predict_sex)
        else:
            df['sex_code'] = 0

        # account_age_days & tweets_per_day
        if 'created_at' in df.columns:
            created = pd.to_datetime(df['created_at'], errors='coerce')
            # Try to remove timezone if present (safe)
            try:
                if created.dt.tz is not None:
                    created = created.dt.tz_convert(None)
            except Exception:
                # ignore if attribute access fails or no tz info
                pass
            now = pd.Timestamp.now()
            df['account_age_days'] = (now - created).dt.total_seconds().div(3600*24).fillna(0)
            df['tweets_per_day'] = df['statuses_count'] / (df['account_age_days'].replace(0, np.nan))
            df['tweets_per_day'] = df['tweets_per_day'].replace([np.inf, -np.inf], 0).fillna(0)
        else:
            df['account_age_days'] = 0
            df['tweets_per_day'] = 0

        # Description length
        if 'description' in df.columns:
            df['description_length'] = df['description'].fillna('').astype(str).apply(len)
        else:
            df['description_length'] = 0

        # Flags
        if 'default_profile' in df.columns:
            df['default_profile'] = df['default_profile'].fillna(0).astype(int)
        else:
            df['default_profile'] = 0

        if 'verified' in df.columns:
            df['verified'] = df['verified'].fillna(0).astype(int)
        else:
            df['verified'] = 0

        # Select features
        features = [
            'statuses_count','followers_count','friends_count','favourites_count','listed_count',
            'sex_code','lang_code','tweets_per_day','account_age_days','description_length','default_profile','verified'
        ]
        features = [f for f in features if f in df.columns]
        return df[features].fillna(0).values

    def export_lang_map(self, path: str):
        if self.lang_map is None:
            raise ValueError("lang_map is not fitted")
        lang_map = self.lang_map
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(lang_map, f, ensure_ascii=False, indent=2)