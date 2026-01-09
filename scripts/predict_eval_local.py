import joblib
import pandas as pd
import json
from pathlib import Path

CSV = Path(r'D:\university\project grad\ml project 2\models\test_with_preds.csv')
MODEL = Path(r'D:\university\project grad\ml project 2\models\randomforest_pipeline.joblib')
OUT = Path(r'D:\university\project grad\ml project 2\models\eval_results_local.json')

def safe_int(x, default=0):
    try:
        if pd.isna(x) or x == '':
            return default
        return int(float(x))
    except Exception:
        return default


def prepare_dataframe(df_csv):
    n = len(df_csv)
    out = pd.DataFrame({
        'name': [f'test_{i}' for i in range(n)],
        'screen_name': [f'test_{i}' for i in range(n)],
        'statuses_count': [100] * n,
        'followers_count': [safe_int(x, 0) for x in df_csv.get('followers_count', [0]*n)],
        'friends_count': [safe_int(x, 0) for x in df_csv.get('friends_count', [0]*n)],
        'favourites_count': [safe_int(x, 0) for x in df_csv.get('favourites_count', [0]*n)],
        'listed_count': [safe_int(x, 0) for x in df_csv.get('listed_count', [0]*n)],
        'created_at': ['2019-01-01 00:00:00'] * n,
        'description': [''] * n,
        'lang': ['en'] * n,
        'default_profile': [0] * n,
        'verified': [int(bool(safe_int(x, 0))) for x in df_csv.get('verified', [0]*n)],
    })
    return out


def main():
    print(f'Loading CSV: {CSV}')
    df = pd.read_csv(CSV)
    print('Loaded rows:', len(df))

    print(f'Loading model: {MODEL}')
    model = joblib.load(MODEL)
    print('Model loaded; type:', type(model))

    X = prepare_dataframe(df)
    # Predict probabilities
    probas = model.predict_proba(X)
    # p_fake is column 1
    p_fake = probas[:, 1]
    preds = (p_fake >= 0.5).astype(int)

    # Compute confusion
    y_true = df['label'].fillna(0).astype(int).values

    TP = int(((y_true == 1) & (preds == 1)).sum())
    FP = int(((y_true == 0) & (preds == 1)).sum())
    TN = int(((y_true == 0) & (preds == 0)).sum())
    FN = int(((y_true == 1) & (preds == 0)).sum())

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    out = {
        'total': total,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    OUT.write_text(json.dumps(out, indent=2))
    print('=== Evaluation summary ===')
    print(out)
    print(f'Wrote results to {OUT}')


if __name__ == '__main__':
    main()
