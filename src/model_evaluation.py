import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

try:
    from .data_preprocessing import DataPreprocessor
    from .prediction_policy import (
        CHURN_DECISION_THRESHOLD,
        HIGH_RISK_THRESHOLD,
        LOW_RISK_THRESHOLD,
        classify_risk,
    )
except ImportError:
    from data_preprocessing import DataPreprocessor
    from prediction_policy import (
        CHURN_DECISION_THRESHOLD,
        HIGH_RISK_THRESHOLD,
        LOW_RISK_THRESHOLD,
        classify_risk,
    )

warnings.filterwarnings('ignore')


class ModelEvaluator:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def evaluate(self, X_test, y_test, threshold=CHURN_DECISION_THRESHOLD):
        probabilities = self.model.predict_proba(X_test)[:, 1]
        predictions = probabilities >= threshold
        y_true = np.asarray(y_test).ravel()
        metrics = {
            'Accuracy': accuracy_score(y_true, predictions),
            'Precision': precision_score(y_true, predictions, zero_division=0),
            'Recall': recall_score(y_true, predictions, zero_division=0),
            'F1-Score': f1_score(y_true, predictions, zero_division=0),
            'ROC-AUC': roc_auc_score(y_true, probabilities),
            'Brier-Score': brier_score_loss(y_true, probabilities),
        }
        report = classification_report(
            y_true,
            predictions,
            target_names=['No Churn', 'Churn'],
            zero_division=0,
        )
        return metrics, report, predictions.astype(int), probabilities

    @staticmethod
    def plot_confusion_matrix(y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'],
            ax=ax,
        )
        ax.set_title(f'Confusion Matrix (threshold={CHURN_DECISION_THRESHOLD:.2f})')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_roc_curve(y_test, y_pred_proba):
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.05), xlabel='False Positive Rate',
               ylabel='True Positive Rate', title='Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_calibration_curve(y_test, probabilities, n_bins=10):
        observed_fraction, mean_predicted = calibration_curve(
            y_test,
            probabilities,
            n_bins=n_bins,
            strategy='uniform',
        )
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(mean_predicted, observed_fraction, marker='o', label='Model')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
        ax.set(
            xlabel='Mean predicted churn probability',
            ylabel='Observed churn frequency',
            title='Probability Calibration',
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_feature_importance(self, top_n=10):
        if not hasattr(self.model, 'feature_importances_'):
            return None, None
        feature_imp = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_,
        }).sort_values('Importance', ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feature_imp.head(top_n), x='Importance', y='Feature', palette='viridis', ax=ax)
        ax.set(title=f'Top {top_n} Feature Importances', xlabel='Importance')
        fig.tight_layout()
        return fig, feature_imp

    @staticmethod
    def analyze_predictions(y_test, y_pred, y_pred_proba):
        results = pd.DataFrame({
            'True_Label': np.asarray(y_test),
            'Predicted_Label': y_pred,
            'Probability': y_pred_proba,
        })
        results['Is_Correct'] = results['True_Label'] == results['Predicted_Label']
        results['Risk_Level'] = results['Probability'].map(classify_risk)
        error_analysis = {
            'Total_Predictions': len(results),
            'Correct_Predictions': int(results['Is_Correct'].sum()),
            'Incorrect_Predictions': int((~results['Is_Correct']).sum()),
            'Accuracy_Score': float(results['Is_Correct'].mean()),
            'Decision_Threshold': CHURN_DECISION_THRESHOLD,
            'Low_Risk_Upper_Bound': LOW_RISK_THRESHOLD,
            'High_Risk_Lower_Bound': HIGH_RISK_THRESHOLD,
        }
        risk_stats = results.groupby('Risk_Level', observed=False).agg(
            Count=('True_Label', 'count'),
            Churn_Count=('True_Label', 'sum'),
            Churn_Rate=('True_Label', 'mean'),
        ).round(3)
        return results, error_analysis, risk_stats


def summarize_contract_missingness(data):
    """Report missing contract-duration rates overall and across observed segments."""
    if 'remaining_contract' not in data:
        raise ValueError("Input data must contain 'remaining_contract'")

    segment_columns = [
        column for column in (
            'churn',
            'is_tv_subscriber',
            'is_movie_package_subscriber',
            'download_over_limit',
        )
        if column in data.columns
    ]
    rows = [{
        'Segment': 'Overall',
        'Value': 'All',
        'Customers': len(data),
        'Missing_Contract': int(data['remaining_contract'].isna().sum()),
        'Missing_Rate': float(data['remaining_contract'].isna().mean()),
    }]
    for segment in segment_columns:
        grouped = data.groupby(segment, dropna=False)['remaining_contract'].agg(
            Customers='size',
            Missing_Contract=lambda values: values.isna().sum(),
            Missing_Rate=lambda values: values.isna().mean(),
        )
        for value, stats in grouped.iterrows():
            rows.append({
                'Segment': segment,
                'Value': value,
                'Customers': int(stats['Customers']),
                'Missing_Contract': int(stats['Missing_Contract']),
                'Missing_Rate': float(stats['Missing_Rate']),
            })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print('Launching model evaluation on the untouched stratified holdout')
    data = pd.read_csv('data/raw/internet_service_churn.csv', na_values=['', ' ', 'NA', 'null', 'NULL'])
    target = data['churn'].copy()
    features = data.drop(columns='churn')
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessor('models/preprocessor.pkl')
    model = joblib.load('models/best_model.pkl')
    X_test_prepared = preprocessor.transform(X_test)

    evaluator = ModelEvaluator(model, preprocessor.feature_columns)
    metrics, class_report, y_pred, y_pred_proba = evaluator.evaluate(X_test_prepared, y_test)
    print(f'Selected model: {pd.read_csv("models/model_results.csv").iloc[0]["Model"]}')
    print(f'Decision threshold: {CHURN_DECISION_THRESHOLD:.2f} (equal false-positive/false-negative costs)')
    print('\nHoldout metrics:')
    for metric, value in metrics.items():
        print(f'  {metric}: {value:.4f}')
    print('\nClassification report:')
    print(class_report)

    output_dir = 'models/plots'
    os.makedirs(output_dir, exist_ok=True)
    figures = {
        'confusion_matrix.png': evaluator.plot_confusion_matrix(y_test, y_pred),
        'roc_curve.png': evaluator.plot_roc_curve(y_test, y_pred_proba),
        'calibration_curve.png': evaluator.plot_calibration_curve(y_test, y_pred_proba),
    }
    feature_fig, feature_importance = evaluator.plot_feature_importance(top_n=10)
    if feature_fig is not None:
        figures['feature_importance.png'] = feature_fig
        print('\nTop feature importances:')
        print(feature_importance.head(10).to_string(index=False))
    for filename, figure in figures.items():
        figure.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close(figure)
        print(f'Saved {filename}')

    results, errors, risk_stats = evaluator.analyze_predictions(y_test, y_pred, y_pred_proba)
    results.to_csv(os.path.join(output_dir, 'predictions_analysis.csv'), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, 'holdout_metrics.csv'), index=False)
    missingness = summarize_contract_missingness(data)
    missingness.to_csv(os.path.join(output_dir, 'missingness_diagnostics.csv'), index=False)
    print('\nPrediction summary:')
    print(errors)
    print('\nRisk groups:')
    print(risk_stats)
    print('\nContract missingness by target/subscription segment:')
    print(missingness.to_string(index=False))
    print('\nSaved holdout_metrics.csv, predictions_analysis.csv, and missingness_diagnostics.csv')
