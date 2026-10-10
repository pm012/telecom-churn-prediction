import os
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    ParameterGrid,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

try:
    from .data_preprocessing import DataPreprocessor
    from .prediction_policy import CHURN_DECISION_THRESHOLD
except ImportError:
    from data_preprocessing import DataPreprocessor
    from prediction_policy import CHURN_DECISION_THRESHOLD

warnings.filterwarnings('ignore')


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_params = None
        self.best_preprocessor = None
        self.best_cv_score = None
        self.results = {}
        self.trained_models = {}

    def create_models(self):
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(
                random_state=42,
                objective='binary:logistic',
                eval_metric='logloss',
                tree_method='hist',
                n_jobs=1,
            ),
            'LightGBM': lgb.LGBMClassifier(
                random_state=42,
                verbose=-1,
                n_jobs=1,
                force_col_wise=True,
            ),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'KNN': KNeighborsClassifier(),
        }
        return self.models

    def get_param_grid(self, model_name):
        return {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
            },
            'LightGBM': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'num_leaves': [31, 50, 70],
                'max_depth': [5, 10, 15],
            },
            'Decision Tree': {
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski'],
            },
        }.get(model_name, {})

    @staticmethod
    def _build_pipeline(model):
        return Pipeline([
            ('preprocessor', DataPreprocessor()),
            ('model', model),
        ])

    def train_model(self, X_train, y_train, model_name, param_grid=None, cv=None):
        estimator = self._build_pipeline(self.models[model_name])
        parameters = self.get_param_grid(model_name) if param_grid is None else param_grid
        if parameters:
            parameters = {f'model__{name}': values for name, values in parameters.items()}
        if cv is None:
            cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=42)

        if not parameters:
            scores = cross_val_score(
                estimator,
                X_train,
                np.asarray(y_train).ravel(),
                cv=cv,
                scoring='roc_auc',
                n_jobs=4,
            )
            estimator.fit(X_train, np.asarray(y_train).ravel())
            return estimator, {}, float(scores.mean()), float(scores.std())

        print(f'  Searching parameters for {model_name} with repeated stratified 3-fold CV (2 repeats)...')
        candidates = min(12, len(ParameterGrid(parameters)))
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=parameters,
            n_iter=candidates,
            cv=cv,
            scoring='roc_auc',
            n_jobs=4,
            pre_dispatch=4,
            verbose=1,
            random_state=42,
            error_score='raise',
            refit=True,
        )

        search.fit(X_train, np.asarray(y_train).ravel())
        best_params = {
            name.removeprefix('model__'): value
            for name, value in search.best_params_.items()
        }
        print(f'  Best CV ROC-AUC: {search.best_score_:.4f} +/- '
              f'{search.cv_results_["std_test_score"][search.best_index_]:.4f}')
        print(f'  Optimal parameters: {best_params}')
        return (
            search.best_estimator_,
            best_params,
            float(search.best_score_),
            float(search.cv_results_['std_test_score'][search.best_index_]),
        )

    @staticmethod
    def evaluate_model(model, preprocessor, X_test, y_test):
        X_test_prepared = preprocessor.transform(X_test)
        y_true = np.asarray(y_test).ravel()
        probabilities = model.predict_proba(X_test_prepared)[:, 1]
        predictions = probabilities >= CHURN_DECISION_THRESHOLD
        return {
            'accuracy': accuracy_score(y_true, predictions),
            'precision': precision_score(y_true, predictions, zero_division=0),
            'recall': recall_score(y_true, predictions, zero_division=0),
            'f1': f1_score(y_true, predictions, zero_division=0),
            'roc_auc': roc_auc_score(y_true, probabilities),
            'brier_score': brier_score_loss(y_true, probabilities),
        }

    def train_all_models(self, X_train, y_train, X_test, y_test, tune_hyperparams=True):
        self.create_models()
        cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=42)

        for model_name in self.models:
            print(f'\n{"=" * 50}\nTraining model: {model_name}')
            params = self.get_param_grid(model_name) if tune_hyperparams else {}
            fitted_pipeline, best_params, cv_score, cv_std = self.train_model(
                X_train,
                y_train,
                model_name,
                param_grid=params,
                cv=cv,
            )
            fitted_model = fitted_pipeline.named_steps['model']
            fitted_preprocessor = fitted_pipeline.named_steps['preprocessor']
            metrics = self.evaluate_model(fitted_model, fitted_preprocessor, X_test, y_test)

            self.results[model_name] = {
                'model': fitted_model,
                'params': best_params,
                'cv_roc_auc': cv_score,
                'cv_roc_auc_std': cv_std,
                'metrics': metrics,
            }
            self.trained_models[model_name] = fitted_model
            print(f'  Holdout metrics: {metrics}')

            if self.best_cv_score is None or cv_score > self.best_cv_score:
                self.best_model = fitted_model
                self.best_preprocessor = fitted_preprocessor
                self.best_model_name = model_name
                self.best_params = best_params
                self.best_cv_score = cv_score

        print(f'\nSelected by training-only CV: {self.best_model_name}')
        print(f'Best CV ROC-AUC: {self.best_cv_score:.4f}')
        return self.best_model

    def save_all_models(self, base_path='models'):
        os.makedirs(base_path, exist_ok=True)
        for model_name, model in self.trained_models.items():
            safe_name = model_name.lower().replace(' ', '_')
            file_path = os.path.join(base_path, f'{safe_name}_model.pkl')
            joblib.dump(model, file_path)
            print(f'Saved model: {model_name} -> {file_path}')

        if self.best_model is None or self.best_preprocessor is None:
            raise RuntimeError('No trained model is available to save')
        joblib.dump(self.best_model, os.path.join(base_path, 'best_model.pkl'))
        self.best_preprocessor.save_preprocessor(os.path.join(base_path, 'preprocessor.pkl'))
        self.get_results_df().to_csv(os.path.join(base_path, 'model_results.csv'), index=False)

    def get_results_df(self):
        rows = []
        for model_name, data in self.results.items():
            row = {
                'Model': model_name,
                'cv_roc_auc': data['cv_roc_auc'],
                'cv_roc_auc_std': data['cv_roc_auc_std'],
            }
            row.update(data['metrics'])
            row['Optimal Parameters'] = str(data['params'])
            rows.append(row)
        return pd.DataFrame(rows).sort_values('cv_roc_auc', ascending=False)


if __name__ == '__main__':
    print('=' * 60)
    print('TRAINING CHURN MODELS')
    print('=' * 60)

    data = pd.read_csv('data/raw/internet_service_churn.csv', na_values=['', ' ', 'NA', 'null', 'NULL'])
    target = data['churn'].copy()
    features = data.drop(columns='churn')
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    trainer = ModelTrainer()
    trainer.train_all_models(X_train, y_train, X_test, y_test, tune_hyperparams=True)
    trainer.save_all_models('models')
    trainer.best_preprocessor.save_processed_data(
        trainer.best_preprocessor.transform(features),
        target,
    )
    print(f'\nSelected model: {trainer.best_model_name}')
    print(trainer.get_results_df().to_string(index=False))
