import pandas as pd
import joblib
from pathlib import Path

# sample data
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

# find model
models_dir = Path('models')
candidates = [models_dir / 'randomforest_pipeline.joblib', models_dir / 'randomforest_best_model.joblib']
model_path = None
for c in candidates:
    if c.exists():
        model_path = c
        break
if model_path is None:
    raise SystemExit('No model file found in models/ (tried randomforest_pipeline.joblib and randomforest_best_model.joblib)')

print('Loading model:', model_path)
model = joblib.load(model_path)

# model may be a pipeline or a classifier
if hasattr(model, 'predict'):
    preds = model.predict(data)
    print('Predictions:', preds)
else:
    print('Loaded object has no predict method')

if hasattr(model, 'predict_proba'):
    probs = model.predict_proba(data)
    print('Probabilities:\n', probs)
else:
    # if pipeline, check if final estimator has predict_proba
    try:
        final = model.named_steps.get('clf') if hasattr(model, 'named_steps') else None
        if final is not None and hasattr(final, 'predict_proba'):
            probs = model.predict_proba(data)
            print('Probabilities:\n', probs)
    except Exception:
        pass

# print feature-engineered array shape if pipeline has fe step
if hasattr(model, 'named_steps') and 'fe' in model.named_steps:
    fe = model.named_steps['fe']
    try:
        X = fe.transform(data)
        print('Transformed feature shape:', X.shape)
    except Exception as e:
        print('Could not transform data with fe step:', e)

# --- Human-expected labels comparison ---
# You can edit these expected labels to reflect the ground-truth for your sample rows.
# Use 1 for 'Real' and 0 for 'Fake' (this follows the training script's target encoding).
expected = [1, 1]  # <-- change these to your expected labels for each row

import numpy as _np
preds_arr = _np.array(preds)
expected_arr = _np.array(expected)
if preds_arr.shape[0] != expected_arr.shape[0]:
    print(f"Warning: number of predictions ({preds_arr.shape[0]}) != number of expected labels ({expected_arr.shape[0]})")
else:
    correct = (preds_arr == expected_arr)
    acc = float(correct.sum()) / len(correct)
    print(f"Match accuracy vs expected: {acc:.3f} ({correct.sum()}/{len(correct)})")
    # print mismatches with some context
    mismatches = [i for i, ok in enumerate(correct) if not ok]
    if len(mismatches) == 0:
        print('All predictions match the expected labels.')
    else:
        print('Mismatches at rows:', mismatches)
        for i in mismatches:
            row = data.iloc[i].to_dict()
            prob = None
            try:
                prob = probs[i]
            except Exception:
                pass
            print(f"Row {i}: expected={expected_arr[i]}, predicted={preds_arr[i]}, proba={prob}")
