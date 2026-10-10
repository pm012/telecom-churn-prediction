import numpy as np
import pandas as pd

from app.core.predictor import ChurnPredictor
from src.data_preprocessing import DataPreprocessor
from src.model_evaluation import ModelEvaluator, summarize_contract_missingness
from src.prediction_policy import (
    CHURN_DECISION_THRESHOLD,
    classify_risk,
    predict_churn,
)


class FixedProbabilityModel:
    def predict_proba(self, features):
        probabilities = np.array([0.49, 0.5, 0.75])[:len(features)]
        return np.column_stack((1 - probabilities, probabilities))


def test_equal_cost_decision_cutoff_and_risk_bands_are_explicit():
    assert CHURN_DECISION_THRESHOLD == 0.5
    assert predict_churn(0.4999) is False
    assert predict_churn(0.5) is True
    assert classify_risk(0.29) == 'Low'
    assert classify_risk(0.3) == 'Medium'
    assert classify_risk(0.7) == 'High'


def test_app_predictions_and_evaluation_use_shared_policy():
    training = pd.DataFrame({
        'subscription_age': [1.0, 2.0, 3.0],
        'bill_avg': [10.0, 20.0, 30.0],
        'remaining_contract': [1.0, 2.0, 3.0],
    })
    preprocessor = DataPreprocessor().fit(training)
    customers = pd.DataFrame({
        'subscription_age': [4.0, 5.0, 6.0],
        'bill_avg': [40.0, 50.0, 60.0],
        'remaining_contract': [4.0, 5.0, 6.0],
    })
    predictor = ChurnPredictor(FixedProbabilityModel(), preprocessor)

    results = predictor.predict_batch(customers)
    assert results['churn_prediction'].tolist() == [False, True, True]
    assert results['risk_level'].tolist() == ['Medium', 'Medium', 'High']

    evaluator = ModelEvaluator(FixedProbabilityModel(), preprocessor.feature_columns)
    metrics, _, predictions, probabilities = evaluator.evaluate(
        preprocessor.transform(customers),
        np.array([0, 1, 1]),
    )
    assert predictions.tolist() == [0, 1, 1]
    assert metrics['Brier-Score'] >= 0
    analyzed, _, _ = evaluator.analyze_predictions(
        np.array([0, 1, 1]),
        predictions,
        probabilities,
    )
    assert analyzed['Risk_Level'].tolist() == ['Medium', 'Medium', 'High']


def test_contract_missingness_diagnostics_cover_target_and_segments():
    data = pd.DataFrame({
        'remaining_contract': [None, 1.0, None, 2.0],
        'churn': [1, 1, 0, 0],
        'is_tv_subscriber': [0, 1, 0, 1],
    })

    summary = summarize_contract_missingness(data)
    overall = summary[summary['Segment'] == 'Overall'].iloc[0]
    churned = summary[(summary['Segment'] == 'churn') & (summary['Value'] == 1)].iloc[0]

    assert overall['Customers'] == 4
    assert overall['Missing_Contract'] == 2
    assert overall['Missing_Rate'] == 0.5
    assert churned['Customers'] == 2
    assert churned['Missing_Rate'] == 0.5
