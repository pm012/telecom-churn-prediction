# test_extreme_cases.py
import pandas as pd
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_preprocessing import DataPreprocessor

def test_extreme_cases():
    """Testing Extreme Cases"""
    
    # Loading the model
    model = joblib.load('models/best_model.pkl')
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessor('models/preprocessor.pkl')
    
    # Extreme test cases
    test_cases = [
        {
            'name': 'Very Low Risk',
            'data': {
                'is_tv_subscriber': 1,
                'is_movie_package_subscriber': 1,
                'subscription_age': 60.0,
                'bill_avg': 10.0,
                'remaining_contract': 24.0,
                'service_failure_count': 0,
                'download_avg': 100.0,
                'upload_avg': 20.0,
                'download_over_limit': 0
            }
        },
        {
            'name': 'High Risk (Extreme)',
            'data': {
                'is_tv_subscriber': 0,
                'is_movie_package_subscriber': 0,
                'subscription_age': 0.5,
                'bill_avg': 300.0,
                'remaining_contract': 0.0,
                'service_failure_count': 20,
                'download_avg': 0.5,
                'upload_avg': 0.0,
                'download_over_limit': 1
            }
        },
        {
            'name': 'Your Test 1',
            'data': {
                'is_tv_subscriber': 0,
                'is_movie_package_subscriber': 0,
                'subscription_age': 12.0,
                'bill_avg': 50.0,
                'remaining_contract': 6.0,
                'service_failure_count': 0,
                'download_avg': 20.0,
                'upload_avg': 5.0,
                'download_over_limit': 0
            }
        },
        {
            'name': 'Your Test 2',
            'data': {
                'is_tv_subscriber': 1,
                'is_movie_package_subscriber': 1,
                'subscription_age': 0.5,
                'bill_avg': 80.0,
                'remaining_contract': 2.0,
                'service_failure_count': 4,
                'download_avg': 2.0,
                'upload_avg': 5.0,
                'download_over_limit': 1
            }
        }
    ]
    
    print("="*60)
    print("TESTING EXTREME CASES")
    print("="*60)
    
    for case in test_cases:
        print(f"\n{case['name']}:")
        print("-" * 40)
        
        df = pd.DataFrame([case['data']])
        prepared_data = preprocessor.prepare_features(df, is_training=False)
        
        for col in preprocessor.feature_columns:
            if col not in prepared_data.columns:
                prepared_data[col] = 0
        prepared_data = prepared_data[preprocessor.feature_columns]
        
        prob = model.predict_proba(prepared_data)[0, 1]
        pred = model.predict(prepared_data)[0]
        
        print(f"Data: {case['data']}")
        print(f"Churn Probability: {prob*100:.2f}%")
        print(f"Prediction: {'Churn' if pred else 'Stay'}")

if __name__ == "__main__":
    test_extreme_cases()