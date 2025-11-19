import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# Simple Streamlit app to serve the saved pipeline for inference

MODEL_CANDIDATES = [Path('models') / 'randomforest_pipeline.joblib', Path('models') / 'randomforest_best_model.joblib']
LABEL_MAP = {1: 'Real', 0: 'Fake'}


@st.cache(allow_output_mutation=True)
def load_model():
    for p in MODEL_CANDIDATES:
        if p.exists():
            return joblib.load(p), p
    return None, None


def build_single_row(
    statuses_count: int,
    followers_count: int,
    friends_count: int,
    favourites_count: int,
    listed_count: int,
    name: str,
    lang: str,
    created_at: str,
    description: str,
    default_profile: int,
    verified: int,
):
    return pd.DataFrame([
        {
            'statuses_count': statuses_count,
            'followers_count': followers_count,
            'friends_count': friends_count,
            'favourites_count': favourites_count,
            'listed_count': listed_count,
            'name': name,
            'lang': lang,
            'created_at': created_at,
            'description': description,
            'default_profile': default_profile,
            'verified': verified,
        }
    ])


def main():
    st.set_page_config(page_title='ML Inference', layout='wide')
    st.title('RandomForest Pipeline - Inference')

    model, model_path = load_model()
    if model is None:
        st.error(f'No model found. Expected one of: {MODEL_CANDIDATES}')
        return

    st.sidebar.write(f'Loaded model: {model_path}')

    input_mode = st.sidebar.selectbox('Input mode', ['Manual single row', 'Upload CSV'])

    df = None

    if input_mode == 'Manual single row':
        st.sidebar.markdown('Provide feature values for one example')
        statuses_count = st.sidebar.number_input('statuses_count', value=100)
        followers_count = st.sidebar.number_input('followers_count', value=10)
        friends_count = st.sidebar.number_input('friends_count', value=20)
        favourites_count = st.sidebar.number_input('favourites_count', value=5)
        listed_count = st.sidebar.number_input('listed_count', value=1)
        name = st.sidebar.text_input('name', 'Alice Smith')
        lang = st.sidebar.text_input('lang', 'en')
        created_at = st.sidebar.text_input('created_at', '2020-01-01 12:00:00')
        description = st.sidebar.text_area('description', 'Hello world!')
        default_profile = int(st.sidebar.selectbox('default_profile', [0, 1], index=0))
        verified = int(st.sidebar.selectbox('verified', [0, 1], index=0))

        if st.sidebar.button('Create DataFrame'):
            df = build_single_row(
                statuses_count,
                followers_count,
                friends_count,
                favourites_count,
                listed_count,
                name,
                lang,
                created_at,
                description,
                default_profile,
                verified,
            )

    else:
        uploaded = st.file_uploader('Upload CSV with raw columns (lang, created_at, name, etc.)', type=['csv'])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.success(f'Loaded {len(df)} rows from uploaded CSV')
                st.dataframe(df.head())
            except Exception as e:
                st.error(f'Could not read uploaded CSV: {e}')

    if df is None:
        st.info('Provide input (manual or upload CSV) and press Create/Upload to proceed')
        return

    st.subheader('Input (first rows)')
    st.dataframe(df.head())

    # Run inference using the pipeline (model may be a Pipeline that expects raw input)
    try:
        preds = model.predict(df)
    except Exception as e:
        st.error(f'Error during model.predict: {e}')
        return

    probs = None
    try:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(df)
        else:
            # try final estimator
            final = model.named_steps.get('clf') if hasattr(model, 'named_steps') else None
            if final is not None and hasattr(final, 'predict_proba'):
                probs = model.predict_proba(df)
    except Exception:
        probs = None

    # Build results DataFrame
    results = df.copy()
    results['_pred'] = preds
    results['_label'] = results['_pred'].map(LABEL_MAP)
    if probs is not None:
        # assume binary
        results['_prob_0'] = probs[:, 0]
        results['_prob_1'] = probs[:, 1]

    st.subheader('Predictions')
    st.dataframe(results[[c for c in results.columns if c.startswith('_')]].head())

    # Offer download
    csv = results.to_csv(index=False)
    st.download_button('Download results CSV', data=csv, file_name='predictions.csv')

    # Show feature-engineered array shape if possible
    fe_msg = 'FeatureEngineer not found in pipeline'
    try:
        if hasattr(model, 'named_steps') and 'fe' in model.named_steps:
            fe = model.named_steps['fe']
            X = fe.transform(df)
            fe_msg = f'FeatureEngineer transformed shape: {X.shape}'
    except Exception as e:
        fe_msg = f'FeatureEngineer transform failed: {e}'

    st.info(fe_msg)


if __name__ == '__main__':
    main()
