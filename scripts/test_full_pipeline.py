"""
Quick test script to verify the full pipeline works end-to-end.
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import joblib

# Load the model
print("Loading model...")
model = joblib.load('models/randomforest_pipeline.joblib')
print("✅ Model loaded successfully!\n")

# Test cases
test_users = [
    {
        "name": "Ahmed Mohamed",
        "description": "Real user - Normal activity",
        "data": {
            'name': 'Ahmed Mohamed',
            'screen_name': 'ahmed_m',
            'statuses_count': 1500,
            'followers_count': 500,
            'friends_count': 300,
            'favourites_count': 200,
            'listed_count': 5,
            'created_at': '2020-01-15',
            'lang': 'ar',
            'description': 'Software Engineer | Cairo | Tech enthusiast',
            'default_profile': 0,
            'verified': 0
        }
    },
    {
        "name": "John Smith",
        "description": "Real user - High engagement",
        "data": {
            'name': 'John Smith',
            'screen_name': 'johnsmith',
            'statuses_count': 5000,
            'followers_count': 2000,
            'friends_count': 500,
            'favourites_count': 1000,
            'listed_count': 20,
            'created_at': '2018-06-01',
            'lang': 'en',
            'description': 'Tech blogger | Python developer | Open source contributor',
            'default_profile': 0,
            'verified': 1
        }
    },
    {
        "name": "xbot12345",
        "description": "Suspicious - Bot-like behavior",
        "data": {
            'name': 'xbot12345',
            'screen_name': 'xbot12345',
            'statuses_count': 50000,
            'followers_count': 10,
            'friends_count': 5000,
            'favourites_count': 0,
            'listed_count': 0,
            'created_at': '2024-11-01',
            'lang': 'en',
            'description': '',
            'default_profile': 1,
            'verified': 0
        }
    },
    {
        "name": "user_abc123",
        "description": "Suspicious - New account, no activity",
        "data": {
            'name': 'user_abc123',
            'screen_name': 'user_abc123',
            'statuses_count': 5,
            'followers_count': 2,
            'friends_count': 1000,
            'favourites_count': 0,
            'listed_count': 0,
            'created_at': '2025-12-01',
            'lang': 'unknown',
            'description': '',
            'default_profile': 1,
            'verified': 0
        }
    },
    {
        "name": "Sarah Johnson",
        "description": "Real user - Moderate activity",
        "data": {
            'name': 'Sarah Johnson',
            'screen_name': 'sarahj',
            'statuses_count': 800,
            'followers_count': 350,
            'friends_count': 400,
            'favourites_count': 500,
            'listed_count': 3,
            'created_at': '2019-03-20',
            'lang': 'en',
            'description': 'Marketing professional | Coffee lover',
            'default_profile': 0,
            'verified': 0
        }
    }
]

print("=" * 70)
print(f"{'User':<20} {'Type':<30} {'Prediction':<10} {'Confidence'}")
print("=" * 70)

for user in test_users:
    df = pd.DataFrame([user["data"]])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    
    result = "Real ✅" if pred == 1 else "Fake ❌"
    confidence = f"Real: {proba[1]*100:.1f}%, Fake: {proba[0]*100:.1f}%"
    
    print(f"{user['name']:<20} {user['description']:<30} {result:<10} {confidence}")

print("=" * 70)
print("\n✅ All predictions completed successfully!")
