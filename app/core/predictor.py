# app/core/predictor.py
import pandas as pd
import numpy as np

class ChurnPredictor:
    """Клас для прогнозування відтоку"""
    
    def __init__(self, model, preprocessor, model_name=None):
        self.model = model
        self.preprocessor = preprocessor
        self.model_name = model_name or "Unknown"
        
    def predict_single(self, customer_data):
        """Прогноз для одного клієнта"""
        if isinstance(customer_data, dict):
            customer_df = pd.DataFrame([customer_data])
        else:
            customer_df = customer_data.copy()
        
        # Підготовка даних
        prepared_data = self.preprocessor.prepare_features(customer_df, is_training=False)
        
        # Переконаємось, що всі ознаки присутні
        for col in self.preprocessor.feature_columns:
            if col not in prepared_data.columns:
                prepared_data[col] = 0
        prepared_data = prepared_data[self.preprocessor.feature_columns]
        
        # Прогноз
        probability = self.model.predict_proba(prepared_data)[0, 1]
        prediction = self.model.predict(prepared_data)[0]
        
        return {
            'probability': probability,
            'prediction': bool(prediction),
            'risk_level': self._get_risk_level(probability),
            'model_used': self.model_name
        }
    
    def predict_batch(self, data):
        """Пакетний прогноз"""
        # Копіюємо дані
        df = data.copy()
        
        # Переконаємось, що всі необхідні колонки є
        for col in self.preprocessor.feature_columns:
            if col not in df.columns:
                df[col] = 0
                print(f"Додано відсутню колонку: {col} зі значенням 0")
        
        # Підготовка даних
        prepared_data = self.preprocessor.prepare_features(df, is_training=False)
        
        # Переконаємось, що всі ознаки присутні в правильному порядку
        for col in self.preprocessor.feature_columns:
            if col not in prepared_data.columns:
                prepared_data[col] = 0
        
        prepared_data = prepared_data[self.preprocessor.feature_columns]
        
        # Прогноз
        probabilities = self.model.predict_proba(prepared_data)[:, 1]
        predictions = self.model.predict(prepared_data)
        
        results = data.copy()
        results['churn_probability'] = probabilities
        results['churn_prediction'] = predictions
        results['risk_level'] = results['churn_probability'].apply(self._get_risk_level)
        results['model_used'] = self.model_name
        
        return results
    
    @staticmethod
    def _get_risk_level(probability):
        """Визначення рівня ризику"""
        if probability >= 0.7:
            return 'Високий'
        elif probability >= 0.4:
            return 'Середній'
        else:
            return 'Низький'
    
    @staticmethod
    def get_risk_color(probability):
        """Отримання кольору для рівня ризику"""
        if probability >= 0.7:
            return '#FF4B4B'
        elif probability >= 0.4:
            return '#FFA500'
        else:
            return '#4CAF50'