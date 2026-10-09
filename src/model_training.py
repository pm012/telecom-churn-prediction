# src/model_training.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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
        self.trained_models = {}
        
    def create_models(self):
        """Create dictionary with different models"""
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),            
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),                        
            'XGBoost': xgb.XGBClassifier(random_state=42, objective='binary:logistic', eval_metric='logloss', tree_method='hist', n_jobs=1),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=1, force_col_wise=True),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'KNN': KNeighborsClassifier()
        }
        self.models = models
        return models
    
    def get_param_grid(self, model_name):
        """Get parameter grid for a model"""
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
        """Train a model using controlled hyperparameter search."""
        y_train = np.asarray(y_train).ravel()

            # === DEBUG ===
        print(f"\n[DEBUG-{model_name}] y_train after ravel: shape={y_train.shape}, unique={np.unique(y_train)[:10]}")
        # === /DEBUG ===

        model = self.models[model_name]

        if param_grid is None:
            param_grid = self.get_param_grid(model_name)

        if not param_grid:
            model.fit(X_train, y_train)
            return model, {}

        print(f"  Searching for optimal hyperparameters for {model_name}...")

        if model_name == "LightGBM":
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=20,
                cv=3,
                scoring="roc_auc",
                n_jobs=4,
                pre_dispatch=4,
                verbose=2,
                random_state=42,
                error_score="raise",
                refit=True
            )
        else:
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=3,
                scoring="roc_auc",
                n_jobs=4,
                pre_dispatch=4,
                verbose=1,
                error_score="raise",
                refit=True
            )

        search.fit(X_train, y_train)

        print(f"  Optimal parameters: {search.best_params_}")
        print(f"  Best CV ROC-AUC: {search.best_score_:.4f}")

        return search.best_estimator_, search.best_params_
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model with different metrics"""
        y_test = np.asarray(y_test).ravel() 
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
        """Train and evaluate all models"""
        self.create_models()
        
        # === DEBUG ===
        print(f"\n[DEBUG] X_train shape: {np.asarray(X_train).shape}, type: {type(X_train)}")
        print(f"[DEBUG] y_train shape: {np.asarray(y_train).shape}, type: {type(y_train)}, dtype: {np.asarray(y_train).dtype}")
        print(f"[DEBUG] y_train first 5: {np.asarray(y_train).flatten()[:5]}")
        print(f"[DEBUG] y_train unique: {np.unique(np.asarray(y_train))}")
        print(f"[DEBUG] y_test shape: {np.asarray(y_test).shape}")
        # === /DEBUG ===
    
        for model_name in self.models.keys():
            for model_name in self.models.keys():
                print(f"\n{'='*50}")
                print(f"Training model: {model_name}")
                
                try:
                    best_model, best_params = self.train_model(
                        X_train, y_train, model_name, 
                        param_grid=self.get_param_grid(model_name) if tune_hyperparams else None
                    )
                    
                    metrics = self.evaluate_model(best_model, X_test, y_test)
                    
                    self.results[model_name] = {
                        'model': best_model,
                        'params': best_params,
                        'metrics': metrics
                    }
                    
                    # Save each trained model
                    self.trained_models[model_name] = best_model
                    
                    print(f"  Metrics for {model_name}:")
                    for metric, value in metrics.items():
                        print(f"    {metric}: {value:.4f}")
                    
                    if self.best_model is None or metrics['roc_auc'] > self.results[self.best_model_name]['metrics']['roc_auc']:
                        self.best_model = best_model
                        self.best_model_name = model_name
                        self.best_params = best_params
                        
                except Exception as e:
                    print(f"  Error occurred while training {model_name}: {e}")
            
            print(f"\n{'='*50}")
            print(f"Best model: {self.best_model_name}")
            print(f"Optimal parameters: {self.best_params}")
            
            return self.best_model
    
    def save_all_models(self, base_path='models'):
        """Saving all trained models"""
        os.makedirs(base_path, exist_ok=True)
        
        # Save each trained model
        for model_name, model in self.trained_models.items():
            safe_name = model_name.lower().replace(' ', '_')
            file_path = f'{base_path}/{safe_name}_model.pkl'
            joblib.dump(model, file_path)
            print(f"Saved model: {model_name} -> {file_path}")
        
        # Saving the best model
        if self.best_model:
            joblib.dump(self.best_model, f'{base_path}/best_model.pkl')
            print(f"Saved the best model: {base_path}/best_model.pkl")
        
        # Saving metrics
        results_df = self.get_results_df()
        results_df.to_csv(f'{base_path}/model_results.csv', index=False)
        print(f"Saved metrics: {base_path}/model_results.csv")
    
    def get_results_df(self):
        """Getting DataFrame with results of all models"""
        results_data = []
        for model_name, data in self.results.items():
            row = {'Model': model_name}
            row.update(data['metrics'])
            row['Optimal Parameters'] = str(data['params'])
            results_data.append(row)
        
        return pd.DataFrame(results_data).sort_values('roc_auc', ascending=False)

if __name__ == "__main__":
    print("="*60)
    print("LAUNCHING TRAINING OF ALL MODELS")
    print("="*60)
    
    # Loading the preprocessor
    try:
        from .data_preprocessing import DataPreprocessor
    except ImportError:
        from data_preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor()
    
    # Checking if the preprocessor exists
    if os.path.exists('models/preprocessor.pkl'):
        preprocessor.load_preprocessor('models/preprocessor.pkl')
    else:
        print("Preprocessor not found. Creating a new one...")
        df_temp = pd.read_csv('data/raw/internet_service_churn.csv', na_values=['', ' '])
        features, target = preprocessor.prepare_features(df_temp, is_training=True)
        preprocessor.save_preprocessor()
    
    # Loading data
    print("\nLoading data...")
    df = pd.read_csv('data/raw/internet_service_churn.csv', na_values=['', ' '])
    features, target = preprocessor.prepare_features(df, is_training=True)
    X_train, X_test, y_train, y_test = preprocessor.split_data(features, target)
    
    # Training models
    print("\nStarting model training...")
    trainer = ModelTrainer()
    best_model = trainer.train_all_models(X_train, y_train, X_test, y_test, tune_hyperparams=True)
    
    # Saving all models
    print("\nSaving all models...")
    trainer.save_all_models('models')
    
    print("\n" + "="*60)
    print(" ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("="*60)