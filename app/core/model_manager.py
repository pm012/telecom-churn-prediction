# app/core/model_manager.py
import joblib
import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_preprocessing import DataPreprocessor


# === Model Name Mapping: filename -> human-readable name ===
MODEL_NAME_MAP = {
    'logistic_regression_model': 'Logistic Regression',
    'random_forest_model': 'Random Forest',
    'gradient_boosting_model': 'Gradient Boosting',
    'xgboost_model': 'XGBoost',
    'lightgbm_model': 'LightGBM',
    'decision_tree_model': 'Decision Tree',
    'knn_model': 'KNN',
}


@st.cache_resource(show_spinner="Loading models...")
def load_models_cached(models_dir='models'):
    """
    Standalone cached function for loading all models, preprocessor and metrics.
    
    Cached by Streamlit — loaded once per session, survives reruns.
    Returns:
        tuple: (models_dict, preprocessor, model_results_df)
    """
    models = {}
    
    # --- Preprocessor ---
    preprocessor = DataPreprocessor()
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
    preprocessor.load_preprocessor(preprocessor_path)
    
    # --- Model results (metrics) ---
    results_path = os.path.join(models_dir, 'model_results.csv')
    if os.path.exists(results_path):
        model_results = pd.read_csv(results_path)
    else:
        model_results = None
    
    # --- All model .pkl files (dynamically) ---
    if not os.path.isdir(models_dir):
        print(f"[ModelManager] Models directory not found: {models_dir}")
        return {}, preprocessor, model_results
    
    for filename in sorted(os.listdir(models_dir)):
        # Skip non-pkl, preprocessor, and best_model (it's a duplicate of the best one)
        if not filename.endswith('.pkl'):
            continue
        if filename in ('preprocessor.pkl', 'best_model.pkl'):
            continue
        
        file_stem = filename.replace('.pkl', '')
        model_name = MODEL_NAME_MAP.get(
            file_stem,
            file_stem.replace('_', ' ').title()
        )
        
        path = os.path.join(models_dir, filename)
        try:
            models[model_name] = joblib.load(path)
            print(f"[ModelManager] Loaded: {model_name} ({filename})")
        except Exception as e:
            print(f"[ModelManager] FAILED to load {filename}: {e}")
    
    print(f"[ModelManager] Total models loaded: {len(models)}")
    print(f"[ModelManager] Models: {list(models.keys())}")
    
    return models, preprocessor, model_results


class ModelManager:
    """Cache Management for Loading and Caching Models"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}
        self.preprocessor = None
        self.model_results = None
    
    def load_all_models(self):
        """
        Load all models via cached function.
        On first call — actually loads. On subsequent calls — returns cached.
        """
        self.models, self.preprocessor, self.model_results = load_models_cached(self.models_dir)
        return self.models, self.preprocessor, self.model_results
    
    def get_available_models(self):
        """Getting the list of available models"""
        if not self.models:
            self.load_all_models()
        return list(self.models.keys())
    
    def get_model(self, model_name='LightGBM'):
        """Getting a specific model"""
        if not self.models:
            self.load_all_models()
        
        model = self.models.get(model_name)
        if model is None and self.models:
            first_name = list(self.models.keys())[0]
            model = self.models[first_name]
            st.warning(f"Model '{model_name}' not found. Using '{first_name}'")
        
        return model, self.preprocessor, self.model_results
    
    def get_model_metrics(self, model_name):
        """Getting metrics for a specific model"""
        if self.model_results is not None:
            row = self.model_results[self.model_results['Model'] == model_name]
            if not row.empty:
                return row.iloc[0].to_dict()
        return None
    
    def save_model(self, model, name, path=None):
        """Saving the model"""
        if path is None:
            path = f'{self.models_dir}/{name.lower().replace(" ", "_")}_model.pkl'
        joblib.dump(model, path)
        print(f"Model saved: {path}")