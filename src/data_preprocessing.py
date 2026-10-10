import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Fit feature cleaning and scaling only on the rows passed to ``fit``."""

    categorical_columns = [
        'is_tv_subscriber',
        'is_movie_package_subscriber',
    ]
    base_numeric_columns = [
        'subscription_age',
        'bill_avg',
        'remaining_contract',
        'service_failure_count',
        'download_avg',
        'upload_avg',
        'download_over_limit',
    ]
    input_columns = [
        'is_tv_subscriber',
        'is_movie_package_subscriber',
        'subscription_age',
        'bill_avg',
        'remaining_contract',
        'service_failure_count',
        'download_avg',
        'upload_avg',
        'download_over_limit',
    ]

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.numeric_columns = self.base_numeric_columns + ['remaining_contract_missing']
        self.medians = {}
        self.modes = {}

    @staticmethod
    def _features_only(data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Expected customer data as a pandas DataFrame')
        return data.drop(columns=['id', 'churn'], errors='ignore').copy()

    def _with_expected_columns(self, data):
        frame = self._features_only(data)
        for column in self.input_columns:
            if column not in frame.columns:
                frame[column] = np.nan
        frame = frame[self.input_columns]
        frame['remaining_contract_missing'] = frame['remaining_contract'].isna().astype(int)
        for column in self.input_columns:
            frame[column] = pd.to_numeric(frame[column], errors='coerce')
        return frame

    def fit(self, X, y=None):
        _ = y
        frame = self._with_expected_columns(X)
        self.medians = {}
        self.modes = {}

        for column in self.base_numeric_columns:
            median = frame[column].median()
            self.medians[column] = float(median) if pd.notna(median) else 0.0

        for column in self.categorical_columns:
            mode = frame[column].mode(dropna=True)
            self.modes[column] = float(mode.iloc[0]) if not mode.empty else 0.0

        cleaned = self._impute(frame)
        self.feature_columns = self.input_columns + ['remaining_contract_missing']
        self.scaler.fit(cleaned[self.numeric_columns])
        return self

    def _impute(self, frame):
        cleaned = frame.copy()
        for column, value in self.medians.items():
            cleaned[column] = cleaned[column].fillna(value)
        for column, value in self.modes.items():
            cleaned[column] = cleaned[column].fillna(value)
        return cleaned

    def transform(self, X):
        if self.feature_columns is None:
            raise ValueError('Preprocessor is not fitted; call fit before transform')
        frame = self._with_expected_columns(X)
        cleaned = self._impute(frame)
        cleaned[self.numeric_columns] = self.scaler.transform(cleaned[self.numeric_columns])
        return cleaned[self.feature_columns]

    def prepare_features(self, df, is_training=True):
        """Compatibility entry point used by the application and data utility."""
        if is_training:
            target = df['churn'].copy() if 'churn' in df.columns else None
            features = self.fit_transform(df)
            if target is None:
                return features
            return features, target
        return self.transform(df)

    def clean_data(self, df):
        """Return raw customer columns without the identifier and target."""
        return self._features_only(df)

    def split_data(self, features, target, test_size=0.2, random_state=42):
        """Split raw rows first, then learn preprocessing from training rows only."""
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
            stratify=target,
        )
        X_train_prepared = self.fit_transform(X_train)
        X_test_prepared = self.transform(X_test)
        return X_train_prepared, X_test_prepared, y_train, y_test

    def save_processed_data(self, features, target):
        os.makedirs('data/processed', exist_ok=True)
        processed = features.copy()
        processed['churn'] = np.asarray(target)
        processed.to_csv('data/processed/processed_data.csv', index=False)

    def save_preprocessor(self, path='models/preprocessor.pkl'):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        state = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'numeric_columns': self.numeric_columns,
            'medians': self.medians,
            'modes': self.modes,
        }
        joblib.dump(state, path)
        print(f'Preprocessor saved to {path}')

    def load_preprocessor(self, path='models/preprocessor.pkl'):
        loaded = joblib.load(path)
        if isinstance(loaded, DataPreprocessor):
            self.__dict__.update(loaded.__dict__)
            print(f'Preprocessor loaded from {path}')
            return
        required = {'scaler', 'feature_columns', 'numeric_columns', 'medians', 'modes'}
        if not isinstance(loaded, dict) or not required.issubset(loaded):
            raise TypeError(f'Unsupported preprocessor artifact at {path}')
        self.scaler = loaded['scaler']
        self.feature_columns = loaded['feature_columns']
        self.numeric_columns = loaded['numeric_columns']
        self.medians = loaded['medians']
        self.modes = loaded['modes']
        print(f'Preprocessor loaded from {path}')


if __name__ == '__main__':
    print('Preparing processed data with a stratified holdout')
    preprocessor = DataPreprocessor()
    df = pd.read_csv('data/raw/internet_service_churn.csv', na_values=['', ' ', 'NA', 'null', 'NULL'])
    target = df['churn'].copy()
    features = df.drop(columns='churn')
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    preprocessor.fit(X_train)
    prepared = preprocessor.transform(features)
    preprocessor.save_processed_data(prepared, target)
    preprocessor.save_preprocessor()
    print(f'Prepared {len(prepared)} rows; preprocessing fitted on {len(X_train)} training rows')
