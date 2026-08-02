import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

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
        'remaining_contract': [6.0, None, 1.0],
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


def test_scaler_is_fitted_on_training_split_only(preprocessor):
    df = pd.DataFrame({
        'subscription_age': [0, 0, 100, 100, 100, 100],
        'bill_avg': [10, 10, 100, 100, 100, 100],
        'is_tv_subscriber': [0, 1, 0, 1, 0, 1],
        'is_movie_package_subscriber': [0, 0, 1, 1, 0, 1],
        'download_over_limit': [0, 0, 1, 1, 0, 1],
        'churn': [0, 0, 0, 0, 1, 1],
    })

    features, target = preprocessor.prepare_features(df, is_training=True)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    preprocessor.split_data(features, target)

    expected_mean = X_train_raw['subscription_age'].mean()

    assert preprocessor.scaler.mean_[0] == pytest.approx(expected_mean)


def test_prepare_features_inference_uses_training_feature_schema(preprocessor):
    df = pd.DataFrame({
        'subscription_age': [1, 2, 3, 4, 5, 6],
        'bill_avg': [10, 20, 30, 40, 50, 60],
        'is_tv_subscriber': [0, 1, 0, 1, 0, 1],
        'is_movie_package_subscriber': [0, 0, 1, 1, 0, 1],
        'download_over_limit': [0, 0, 1, 1, 0, 1],
        'churn': [0, 0, 0, 1, 1, 1],
    })

    features, target = preprocessor.prepare_features(df, is_training=True)
    preprocessor.split_data(features, target)

    inference_df = pd.DataFrame({
        'subscription_age': [4.0],
        'bill_avg': [40.0],
        'is_tv_subscriber': [1],
        'download_over_limit': [0]
    })

    prepared = preprocessor.prepare_features(inference_df, is_training=False)

    assert list(prepared.columns) == preprocessor.feature_columns
    assert prepared.shape[1] == len(preprocessor.feature_columns)
    assert prepared.loc[0, 'is_movie_package_subscriber'] == 0
