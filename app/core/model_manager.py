# app/core/model_manager.py
import joblib
import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_preprocessing import DataPreprocessor

class ModelManager:
    """Керування завантаженням та кешуванням моделей"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}
        self.preprocessor = None
        self.model_results = None
        
    @st.cache_resource
    def load_all_models(_self):
        """Завантаження всіх доступних моделей"""
        try:
            # Завантаження препроцесора
            preprocessor = DataPreprocessor()
            preprocessor.load_preprocessor(f'{_self.models_dir}/preprocessor.pkl')
            _self.preprocessor = preprocessor
            
            # Завантаження результатів моделей
            results_path = f'{_self.models_dir}/model_results.csv'
            if os.path.exists(results_path):
                _self.model_results = pd.read_csv(results_path)
            
            # Завантаження окремих моделей
            model_files = {
                'LightGBM': f'{_self.models_dir}/best_model.pkl',  # за замовчуванням
                'XGBoost': f'{_self.models_dir}/xgboost_model.pkl',
                'Random Forest': f'{_self.models_dir}/random_forest_model.pkl',
                'Gradient Boosting': f'{_self.models_dir}/gradient_boosting_model.pkl'
            }
            
            for name, path in model_files.items():
                if os.path.exists(path):
                    _self.models[name] = joblib.load(path)
                    print(f"Завантажено модель: {name}")
                else:
                    print(f"Модель не знайдена: {path}")
            
            # Якщо немає окремих моделей, використовуємо best_model як LightGBM
            if not _self.models and os.path.exists(f'{_self.models_dir}/best_model.pkl'):
                _self.models['LightGBM'] = joblib.load(f'{_self.models_dir}/best_model.pkl')
            
            return _self.models, _self.preprocessor, _self.model_results
            
        except Exception as e:
            st.error(f"Помилка завантаження моделей: {e}")
            return {}, None, None
    
    def get_available_models(self):
        """Отримання списку доступних моделей"""
        if not self.models:
            self.models, self.preprocessor, self.model_results = self.load_all_models()
        return list(self.models.keys())
    
    def get_model(self, model_name='LightGBM'):
        """Отримання конкретної моделі"""
        if not self.models:
            self.models, self.preprocessor, self.model_results = self.load_all_models()
        
        model = self.models.get(model_name)
        if model is None and self.models:
            # Якщо модель не знайдена, повертаємо першу доступну
            model = list(self.models.values())[0]
            st.warning(f"Модель '{model_name}' не знайдена. Використовується {list(self.models.keys())[0]}")
        
        return model, self.preprocessor, self.model_results
    
    def get_model_metrics(self, model_name):
        """Отримання метрик для конкретної моделі"""
        if self.model_results is not None:
            row = self.model_results[self.model_results['Модель'] == model_name]
            if not row.empty:
                return row.iloc[0].to_dict()
        return None
    
    def save_model(self, model, name, path=None):
        """Збереження моделі"""
        if path is None:
            path = f'{self.models_dir}/{name.lower().replace(" ", "_")}_model.pkl'
        joblib.dump(model, path)
        print(f"Модель збережено: {path}")