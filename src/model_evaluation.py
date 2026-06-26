# src/model_evaluation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        
    def evaluate(self, X_test, y_test, threshold=0.5):
        """Комплексна оцінка моделі"""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['ROC-AUC'] = roc_auc_score(y_test, y_pred_proba)
        
        class_report = classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])
        
        return metrics, class_report, y_pred, y_pred_proba
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """Візуалізація матриці плутанини"""
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        return fig
    
    def plot_roc_curve(self, y_test, y_pred_proba):
        """Візуалізація ROC-кривої"""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def plot_feature_importance(self, top_n=10):
        """Візуалізація важливості ознак"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_imp = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_imp.head(top_n), x='Importance', y='Feature', palette='viridis')
            plt.title(f'Top {top_n} Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            
            return fig, feature_imp
        
        return None, None
    
    def analyze_predictions(self, X_test, y_test, y_pred, y_pred_proba):
        """Детальний аналіз передбачень"""
        results_df = pd.DataFrame({
            'True_Label': y_test,
            'Predicted_Label': y_pred,
            'Probability': y_pred_proba if y_pred_proba is not None else y_pred
        })
        
        results_df['Is_Correct'] = results_df['True_Label'] == results_df['Predicted_Label']
        
        if y_pred_proba is not None:
            results_df['Risk_Level'] = pd.cut(
                results_df['Probability'], 
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Low', 'Medium', 'High']
            )
        
        error_analysis = {
            'Total_Predictions': len(results_df),
            'Correct_Predictions': results_df['Is_Correct'].sum(),
            'Incorrect_Predictions': (~results_df['Is_Correct']).sum(),
            'Accuracy_Score': results_df['Is_Correct'].mean(),
        }
        
        if y_pred_proba is not None:
            risk_stats = results_df.groupby('Risk_Level').agg({
                'True_Label': ['count', 'sum', 'mean']
            }).round(3)
            risk_stats.columns = ['Count', 'Churn_Count', 'Churn_Rate']
            
            return results_df, error_analysis, risk_stats
        
        return results_df, error_analysis, None

if __name__ == "__main__":
    print("ЗАПУСК ОЦІНКИ МОДЕЛІ")
    print("="*50)
    
    # Завантаження препроцесора
    from data_preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessor('models/preprocessor.pkl')
    
    # Завантаження моделі
    model = joblib.load('models/best_model.pkl')
    print("Модель завантажено")
    
    # Завантаження даних
    df = pd.read_csv('data/raw/internet_service_churn.csv', na_values=['', ' '])
    
    # Підготовка даних
    features, target = preprocessor.prepare_features(df, is_training=True)
    
    # Розділення даних
    X_train, X_test, y_train, y_test = preprocessor.split_data(features, target)
    
    # Оцінка моделі
    evaluator = ModelEvaluator(model, preprocessor.feature_columns)
    metrics, class_report, y_pred, y_pred_proba = evaluator.evaluate(X_test, y_test)
    
    print("\nМетрики моделі:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nClassification Report:")
    print(class_report)
    
    # Створення папки для збереження візуалізацій
    os.makedirs('models/plots', exist_ok=True)
    
    # Візуалізація
    print("\nСтворення візуалізацій...")
    
    # Матриця плутанини
    fig1 = evaluator.plot_confusion_matrix(y_test, y_pred)
    fig1.savefig('models/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("Матрицю плутанини збережено в models/plots/confusion_matrix.png")
    
    # ROC-крива
    fig2 = evaluator.plot_roc_curve(y_test, y_pred_proba)
    fig2.savefig('models/plots/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("ROC-криву збережено в models/plots/roc_curve.png")
    
    # Важливість ознак
    fig3, feature_imp = evaluator.plot_feature_importance(top_n=10)
    if fig3 is not None:
        fig3.savefig('models/plots/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print("Важливість ознак збережено в models/plots/feature_importance.png")
        print("\nТоп-10 найважливіших ознак:")
        print(feature_imp.head(10).to_string(index=False))
    
    # Детальний аналіз
    results_df, error_analysis, risk_stats = evaluator.analyze_predictions(
        X_test, y_test, y_pred, y_pred_proba
    )
    
    print("\nСтатистика помилок:")
    for key, value in error_analysis.items():
        print(f"  {key}: {value}")
    
    if risk_stats is not None:
        print("\nРозподіл за рівнями ризику:")
        print(risk_stats)
    
    # Збереження результатів
    results_df.to_csv('models/plots/predictions_analysis.csv', index=False)
    print("\nРезультати аналізу збережено в models/plots/predictions_analysis.csv")
    
    print("\n" + "="*50)
    print("ОЦІНКУ МОДЕЛІ ЗАВЕРШЕНО")