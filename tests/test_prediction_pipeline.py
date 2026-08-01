import pandas as pd
import joblib
import pytest

from src.data_preprocessing import DataPreprocessor


@pytest.fixture
def pipeline_components():
    model = joblib.load('models/best_model.pkl')
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessor('models/preprocessor.pkl')
    return model, preprocessor


def test_single_prediction_returns_probabilities_and_label(pipeline_components):
    model, preprocessor = pipeline_components
    customer = pd.DataFrame([{
        'is_tv_subscriber': 1,
        'is_movie_package_subscriber': 1,
        'subscription_age': 24.0,
        'bill_avg': 30.0,
        'reamining_contract': 12.0,
        'service_failure_count': 0,
        'download_avg': 50.0,
        'upload_avg': 10.0,
        'download_over_limit': 0
    }])

    prepared = preprocessor.prepare_features(customer, is_training=False)
    for col in preprocessor.feature_columns:
        if col not in prepared.columns:
            prepared[col] = 0
    prepared = prepared[preprocessor.feature_columns]

    probability = model.predict_proba(prepared)[0, 1]
    prediction = model.predict(prepared)[0]

    assert 0.0 <= probability <= 1.0
    assert prediction in {0, 1}


def test_batch_prediction_keeps_expected_shape(pipeline_components):
    model, preprocessor = pipeline_components
    batch = pd.DataFrame([
        {
            'is_tv_subscriber': 1,
            'is_movie_package_subscriber': 1,
            'subscription_age': 24.0,
            'bill_avg': 30.0,
            'reamining_contract': 12.0,
            'service_failure_count': 0,
            'download_avg': 50.0,
            'upload_avg': 10.0,
            'download_over_limit': 0
        },
        {
            'is_tv_subscriber': 0,
            'is_movie_package_subscriber': 0,
            'subscription_age': 2.0,
            'bill_avg': 160.0,
            'reamining_contract': 0.1,
            'service_failure_count': 6,
            'download_avg': 2.0,
            'upload_avg': 0.2,
            'download_over_limit': 1
        }
    ])

    prepared = preprocessor.prepare_features(batch, is_training=False)
    for col in preprocessor.feature_columns:
        if col not in prepared.columns:
            prepared[col] = 0
    prepared = prepared[preprocessor.feature_columns]

    probabilities = model.predict_proba(prepared)[:, 1]

    assert len(probabilities) == len(batch)
    assert probabilities.shape[0] == 2
