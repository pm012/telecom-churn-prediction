# app/core/predictor.py
import pandas as pd

from src.prediction_policy import (
    CHURN_DECISION_THRESHOLD,
    HIGH_RISK_THRESHOLD,
    LOW_RISK_THRESHOLD,
    classify_risk,
    predict_churn,
)

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
        probability = float(self.model.predict_proba(prepared_data)[0, 1])
        prediction = predict_churn(probability)
        
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
        predictions = probabilities >= CHURN_DECISION_THRESHOLD
        
        results = data.copy()
        results['churn_probability'] = probabilities
        results['churn_prediction'] = predictions
        results['risk_level'] = results['churn_probability'].apply(self._get_risk_level)
        results['model_used'] = self.model_name
        
        return results
    
    @staticmethod
    def _get_risk_level(probability):
        return classify_risk(probability)
    
    @staticmethod
    def get_risk_color(probability):
        """Getting the color for the risk level"""
        if probability >= HIGH_RISK_THRESHOLD:
            return '#FF4B4B'
        elif probability >= LOW_RISK_THRESHOLD:
            return '#FFA500'
        else:
            return '#4CAF50'