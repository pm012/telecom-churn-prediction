import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.data_preprocessing import DataPreprocessor


@pytest.fixture
def preprocessor():
    return DataPreprocessor()


def test_preprocessor_fits_on_training_partition_only(preprocessor):
    data = pd.DataFrame({
        'subscription_age': [1.0, 3.0, np.nan, 1000.0],
        'bill_avg': [10.0, 30.0, 50.0, 5000.0],
        'remaining_contract': [2.0, 4.0, np.nan, 500.0],
        'is_tv_subscriber': [0, 1, 0, 1],
        'churn': [0, 1, 0, 1],
    })
    X_train = data.iloc[:2].drop(columns='churn')
    X_test = data.iloc[2:].drop(columns='churn')

    preprocessor.fit(X_train)
    prepared = preprocessor.transform(X_test)

    assert preprocessor.medians['subscription_age'] == pytest.approx(2.0)
    assert preprocessor.medians['remaining_contract'] == pytest.approx(3.0)
    assert prepared.shape == (2, 10)
    assert np.isfinite(prepared.to_numpy()).all()


def test_transform_handles_missing_input_columns_using_training_schema(preprocessor):
    training = pd.DataFrame({
        'subscription_age': [1.0, 2.0, 3.0],
        'bill_avg': [10.0, 20.0, 30.0],
        'remaining_contract': [2.0, 4.0, 6.0],
        'is_tv_subscriber': [0, 1, 1],
        'is_movie_package_subscriber': [1, 0, 1],
        'download_over_limit': [0, 0, 1],
    })
    preprocessor.fit(training)
    inferred = preprocessor.transform(pd.DataFrame({'subscription_age': [4.0]}))

    assert list(inferred.columns) == preprocessor.feature_columns
    assert inferred.shape == (1, 10)
    assert inferred.loc[0, 'is_movie_package_subscriber'] == 1


def test_download_over_limit_is_treated_as_a_numeric_count(preprocessor):
    training = pd.DataFrame({
        'subscription_age': [1.0, 2.0, 3.0],
        'bill_avg': [10.0, 20.0, 30.0],
        'download_over_limit': [0, 2, 7],
    })
    preprocessor.fit(training)
    transformed = preprocessor.transform(training)

    assert 'download_over_limit' in preprocessor.numeric_columns
    assert len(transformed['download_over_limit'].unique()) == 3
    assert transformed['download_over_limit'].iloc[1] != 1


def test_split_data_splits_before_fitting_preprocessor(preprocessor):
    source = pd.read_csv('data/raw/internet_service_churn.csv')
    target = source['churn']
    raw_features = source.drop(columns='churn')
    expected_train, _, _, _ = train_test_split(
        raw_features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    X_train, X_test, y_train, y_test = preprocessor.split_data(
        raw_features,
        target,
    )

    expected_median = expected_train['subscription_age'].median()
    assert preprocessor.scaler.n_samples_seen_ == len(expected_train)
    assert preprocessor.medians['subscription_age'] == pytest.approx(expected_median)
    assert len(X_train) + len(X_test) == len(source)
    assert len(y_train) + len(y_test) == len(source)
    assert X_train.shape[1] == X_test.shape[1] == len(preprocessor.feature_columns)
    assert 'id' not in preprocessor.feature_columns
    assert 'churn' not in preprocessor.feature_columns


def test_prepare_features_uses_fitted_training_values(preprocessor):
    train = pd.DataFrame({
        'subscription_age': [1.0, 3.0],
        'bill_avg': [10.0, 30.0],
        'remaining_contract': [2.0, 4.0],
        'is_tv_subscriber': [0, 1],
        'churn': [0, 1],
    })
    prepared, target = preprocessor.prepare_features(train, is_training=True)
    inference = preprocessor.prepare_features(
        pd.DataFrame({'subscription_age': [4.0], 'bill_avg': [20.0]}),
        is_training=False,
    )

    assert target.tolist() == [0, 1]
    assert prepared.shape == (2, 10)
    assert list(inference.columns) == preprocessor.feature_columns
