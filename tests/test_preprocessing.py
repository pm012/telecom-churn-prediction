import pandas as pd
import pytest

from src.data_preprocessing import DataPreprocessor


@pytest.fixture
def preprocessor():
    return DataPreprocessor()


def test_prepare_features_returns_expected_shape(preprocessor):
    df = pd.read_csv('data/raw/internet_service_churn.csv')

    features, target = preprocessor.prepare_features(df, is_training=True)

    assert features.shape[0] == df.shape[0]
    assert target.shape[0] == df.shape[0]
    assert features.shape[1] == len(preprocessor.feature_columns)
    assert set(target.unique()).issubset({0, 1})


def test_missing_values_are_handled(preprocessor):
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'is_tv_subscriber': [1, None, 0],
        'is_movie_package_subscriber': [0, 1, None],
        'subscription_age': [12.0, None, 6.0],
        'bill_avg': [20, 30, None],
        'reamining_contract': [6.0, None, 1.0],
        'service_failure_count': [0, 1, 2],
        'download_avg': [10.0, None, 20.0],
        'upload_avg': [2.0, None, 4.0],
        'download_over_limit': [0, 1, None],
        'churn': [0, 1, 0]
    })

    cleaned = preprocessor.clean_data(df)

    assert cleaned.isnull().sum().sum() == 0
    assert 'id' not in cleaned.columns


def test_split_data_preserves_target_distribution(preprocessor):
    df = pd.read_csv('data/raw/internet_service_churn.csv')
    features, target = preprocessor.prepare_features(df, is_training=True)

    X_train, X_test, y_train, y_test = preprocessor.split_data(features, target)

    assert len(X_train) + len(X_test) == len(target)
    assert len(y_train) + len(y_test) == len(target)
    assert X_train.shape[1] == X_test.shape[1]
