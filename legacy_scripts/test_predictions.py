# test_predictions.py
import pandas as pd
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import DataPreprocessor

def test_predictions():
    """Testing Different Customer Scenarios"""
    
    # Loading the model and preprocessor
    model = joblib.load('models/best_model.pkl')
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessor('models/preprocessor.pkl')
    
    # Test cases
    test_cases = [
        {
            'name': 'Low Risk (Loyal Customer)',
            'data': {
                'is_tv_subscriber': 1,
                'is_movie_package_subscriber': 1,
                'subscription_age': 24.0,
                'bill_avg': 30.0,
                'remaining_contract': 12.0,
                'service_failure_count': 0,
                'download_avg': 50.0,
                'upload_avg': 10.0,
                'download_over_limit': 0
            }
        },
        {
            'name': 'Medium Risk',
            'data': {
                'is_tv_subscriber': 1,
                'is_movie_package_subscriber': 0,
                'subscription_age': 6.0,
                'bill_avg': 80.0,
                'remaining_contract': 2.0,
                'service_failure_count': 2,
                'download_avg': 20.0,
                'upload_avg': 3.0,
                'download_over_limit': 1
            }
        },
        {
            'name': 'High Risk (Problematic Customer)',
            'data': {
                'is_tv_subscriber': 0,
                'is_movie_package_subscriber': 0,
                'subscription_age': 3.0,
                'bill_avg': 150.0,
                'remaining_contract': 0.1,
                'service_failure_count': 5,
                'download_avg': 5.0,
                'upload_avg': 0.5,
                'download_over_limit': 1
            }
        },
        {
            'name': 'Extreme Risk',
            'data': {
                'is_tv_subscriber': 0,
                'is_movie_package_subscriber': 0,
                'subscription_age': 1.0,
                'bill_avg': 200.0,
                'remaining_contract': 0.0,
                'service_failure_count': 10,
                'download_avg': 2.0,
                'upload_avg': 0.1,
                'download_over_limit': 1
            }
        }
    ]
    
    print("="*60)
    print("TESTING DIFFERENT CUSTOMER SCENARIOS")
    print("="*60)
    
    for case in test_cases:
        print(f"\nScenario: {case['name']}")
        print("-" * 40)
        
        # Creating DataFrame
        df = pd.DataFrame([case['data']])
        
        # Preparing data
        prepared_data = preprocessor.prepare_features(df, is_training=False)
        
        # Ensuring all features are present
        for col in preprocessor.feature_columns:
            if col not in prepared_data.columns:
                prepared_data[col] = 0
        prepared_data = prepared_data[preprocessor.feature_columns]
        
        # Prediction
        prob = model.predict_proba(prepared_data)[0, 1]
        pred = model.predict(prepared_data)[0]
        
        print(f"Client Data:")
        for key, value in case['data'].items():
            print(f"  {key}: {value}")
        
        print(f"\nResults:")
        print(f"  Churn Probability: {prob*100:.2f}%")
        print(f"  Prediction: {'Churn' if pred else 'Stay'}")
        print(f"  Risk Level: {'High' if prob >= 0.7 else 'Medium' if prob >= 0.4 else 'Low'}")

if __name__ == "__main__":
    test_predictions()