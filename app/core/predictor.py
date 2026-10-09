# app/core/predictor.py
import pandas as pd
import numpy as np

class ChurnPredictor:
    """Class for churn prediction"""
    
    def __init__(self, model, preprocessor, model_name=None):
        self.model = model
        self.preprocessor = preprocessor
        self.model_name = model_name or "Unknown"
        
    def predict_single(self, customer_data):
        """Predict for a single client"""
        if isinstance(customer_data, dict):
            customer_df = pd.DataFrame([customer_data])
        else:
            customer_df = customer_data.copy()
        
        # Preparation of data
        prepared_data = self.preprocessor.prepare_features(customer_df, is_training=False)
        
        # Ensure all features are present
        for col in self.preprocessor.feature_columns:
            if col not in prepared_data.columns:
                prepared_data[col] = 0
        prepared_data = prepared_data[self.preprocessor.feature_columns]
        
        # Predict
        probability = self.model.predict_proba(prepared_data)[0, 1]
        prediction = self.model.predict(prepared_data)[0]
        
        return {
            'probability': probability,
            'prediction': bool(prediction),
            'risk_level': self._get_risk_level(probability),
            'model_used': self.model_name
        }
    
    def predict_batch(self, data):
        """Batch prediction"""
        # Copy the data
        df = data.copy()
        
        # Ensure all required columns are present
        for col in self.preprocessor.feature_columns:
            if col not in df.columns:
                df[col] = 0
                print(f"Added missing column: {col} with value 0")
        
        # Prepare the data
        prepared_data = self.preprocessor.prepare_features(df, is_training=False)
        
        # Ensure all features are present in the correct order
        for col in self.preprocessor.feature_columns:
            if col not in prepared_data.columns:
                prepared_data[col] = 0
        
        prepared_data = prepared_data[self.preprocessor.feature_columns]
        
        # Predict
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
        """Determination of risk level"""
        if probability >= 0.7:
            return 'High'
        elif probability >= 0.4:
            return 'Medium'
        else:
            return 'Low'
    
    @staticmethod
    def get_risk_color(probability):
        """Getting the color for the risk level"""
        if probability >= 0.7:
            return '#FF4B4B'
        elif probability >= 0.4:
            return '#FFA500'
        else:
            return '#4CAF50'