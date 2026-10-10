CHURN_DECISION_THRESHOLD = 0.5
LOW_RISK_THRESHOLD = 0.3
HIGH_RISK_THRESHOLD = 0.7


def classify_risk(probability):
    if probability >= HIGH_RISK_THRESHOLD:
        return 'High'
    if probability >= LOW_RISK_THRESHOLD:
        return 'Medium'
    return 'Low'


def predict_churn(probability):
    return probability >= CHURN_DECISION_THRESHOLD
