# app/core/model_manager.py
import joblib
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_preprocessing import DataPreprocessor

class ModelManager:
    """Керування завантаженням та кешуванням моделі"""
    
    def __init__(self, model_path='models/best_model.pkl', preprocessor_path='models/preprocessor.pkl'):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None
        
    @st.cache_resource
    def load_model_and_preprocessor(_self):
        """Завантаження моделі та препроцесора з кешуванням"""
        try:
            model = joblib.load(_self.model_path)
            preprocessor = DataPreprocessor()
            preprocessor.load_preprocessor(_self.preprocessor_path)
            return model, preprocessor
        except Exception as e:
            st.error(f"Помилка завантаження моделі: {e}")
            return None, None
    
    def get_model(self):
        """Отримання моделі"""
        if self.model is None or self.preprocessor is None:
            self.model, self.preprocessor = self.load_model_and_preprocessor()
        return self.model, self.preprocessor
    
    def is_loaded(self):
        """Перевірка завантаження"""
        return self.model is not None and self.preprocessor is not None