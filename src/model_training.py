# src/model_training.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_params = None
        self.results = {}
        
    def create_models(self):
        """Створення словника з різними моделями (без SVM)"""
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'KNN': KNeighborsClassifier()
        }
        self.models = models
        return models
    
    def get_param_grid(self, model_name):
        """Отримання сітки параметрів для моделі"""
        param_grids = {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'LightGBM': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'num_leaves': [31, 50, 70],
                'max_depth': [5, 10, 15]
            },
            'Decision Tree': {
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        }
        return param_grids.get(model_name, {})
    
    def train_model(self, X_train, y_train, model_name, param_grid=None):
        """Навчання однієї моделі з пошуком гіперпараметрів"""
        model = self.models[model_name]
        
        if param_grid is None:
            param_grid = self.get_param_grid(model_name)
        
        if param_grid:
            print(f"Пошук гіперпараметрів для {model_name}...")
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring='roc_auc', 
                n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"Найкращі параметри: {best_params}")
        else:
            model.fit(X_train, y_train)
            best_model = model
            best_params = {}
            
        return best_model, best_params
    
    def evaluate_model(self, model, X_test, y_test):
        """Оцінка моделі за різними метриками"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1': f1_score(y_test, y_pred, average='binary')
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        return metrics
    
    def train_all_models(self, X_train, y_train, X_test, y_test, tune_hyperparams=True):
        """Навчання та оцінка всіх моделей (без SVM)"""
        self.create_models()
        
        for model_name in self.models.keys():
            print(f"\n{'='*50}")
            print(f"Навчання моделі: {model_name}")
            
            try:
                # Навчання
                best_model, best_params = self.train_model(
                    X_train, y_train, model_name, 
                    param_grid=self.get_param_grid(model_name) if tune_hyperparams else None
                )
                
                # Оцінка
                metrics = self.evaluate_model(best_model, X_test, y_test)
                
                # Збереження результатів
                self.results[model_name] = {
                    'model': best_model,
                    'params': best_params,
                    'metrics': metrics
                }
                
                print(f"Метрики для {model_name}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
                
                # Оновлення найкращої моделі
                if self.best_model is None or metrics['roc_auc'] > self.results[self.best_model_name]['metrics']['roc_auc']:
                    self.best_model = best_model
                    self.best_model_name = model_name
                    self.best_params = best_params
                    
            except Exception as e:
                print(f"Помилка при навчанні {model_name}: {e}")
        
        print(f"\n{'='*50}")
        print(f"Найкраща модель: {self.best_model_name}")
        print(f"Найкращі параметри: {self.best_params}")
        
        return self.best_model
    
    def save_model(self, model, path='models/best_model.pkl'):
        """Збереження моделі"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        print(f"Модель збережено в {path}")
    
    def load_model(self, path='models/best_model.pkl'):
        """Завантаження моделі"""
        model = joblib.load(path)
        print(f"Модель завантажено з {path}")
        return model
    
    def get_results_df(self):
        """Отримання DataFrame з результатами всіх моделей"""
        results_data = []
        for model_name, data in self.results.items():
            row = {'Модель': model_name}
            row.update(data['metrics'])
            row['Найкращі параметри'] = str(data['params'])
            results_data.append(row)
        
        return pd.DataFrame(results_data).sort_values('roc_auc', ascending=False)

if __name__ == "__main__":
    print("ЗАПУСК НАВЧАННЯ МОДЕЛЕЙ")
    print("="*50)
    
    # Завантаження даних
    from data_preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessor('models/preprocessor.pkl')
    
    # Завантаження даних
    df = pd.read_csv('data/raw/internet_service_churn.csv', na_values=['', ' '])
    
    # Підготовка даних
    features, target = preprocessor.prepare_features(df, is_training=True)
    
    # Розділення даних
    X_train, X_test, y_train, y_test = preprocessor.split_data(features, target)
    
    # Навчання моделей (без SVM)
    trainer = ModelTrainer()
    best_model = trainer.train_all_models(X_train, y_train, X_test, y_test, tune_hyperparams=True)
    
    # Збереження найкращої моделі
    trainer.save_model(best_model, 'models/best_model.pkl')
    
    # Виведення результатів
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТИ ВСІХ МОДЕЛЕЙ")
    print("="*50)
    results_df = trainer.get_results_df()
    print(results_df.to_string(index=False))
    
    # Збереження результатів
    results_df.to_csv('models/model_results.csv', index=False)
    print("\nРезультати збережено в models/model_results.csv")