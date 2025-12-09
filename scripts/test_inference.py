import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from feature_engineer import FeatureEngineer

# Sample data
data = pd.DataFrame([
    {
        'statuses_count': 100, 'followers_count': 10, 'friends_count': 20,
        'favourites_count': 5, 'listed_count': 1, 'name': 'Alice Smith',
        'lang': 'en', 'created_at': '2020-01-01 12:00:00',
        'description': 'Hello world!', 'default_profile': 0, 'verified': 0
    },
    {
        'statuses_count': 50, 'followers_count': 5, 'friends_count': 10,
        'favourites_count': 3, 'listed_count': 0, 'name': 'Bob Johnson',
        'lang': 'fr', 'created_at': '2021-05-05 08:30:00',
        'description': '', 'default_profile': 1, 'verified': 0
    }
])

fe = FeatureEngineer()
fe.fit(data)
X_transformed = fe.transform(data)

# Print columns and values
feature_names = [
    'statuses_count','followers_count','friends_count','favourites_count','listed_count',
    'sex_code','lang_code','tweets_per_day','account_age_days','description_length',
    'default_profile','verified'
]
print(pd.DataFrame(X_transformed, columns=feature_names))
