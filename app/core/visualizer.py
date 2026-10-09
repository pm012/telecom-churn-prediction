# app/core/visualizer.py
import matplotlib.pyplot as plt
import numpy as np

class ResultVisualizer:
    """Visualization of prediction results"""
    
    @staticmethod
    def plot_risk_gauge(probability):
        """Displaying the risk gauge"""
        fig, ax = plt.subplots(figsize=(10, 2))
        
        # Gradient from green to red
        gradient = np.linspace(0, 1, 100).reshape(1, -1)
        ax.imshow(gradient, cmap='RdYlGn_r', aspect='auto', extent=[0, 1, 0, 1])
        
        # Marker for the current probability
        ax.axvline(x=probability, color='black', linewidth=3, linestyle='--')
        
        # Text with probability
        ax.text(probability, 0.5, f'{probability*100:.1f}%', 
               ha='center', va='center', fontsize=20, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.set_yticks([])
        ax.set_title('Probability of Customer Churn', fontsize=14, fontweight='bold')
        
        return fig
    
    @staticmethod
    def get_recommendation(probability):
        """Getting recommendations based on probability"""
        if probability >= 0.7:
            return {
                'type': 'error',
                'title': 'Act immediately to retain the customer!',
                'actions': [
                    'Propose a special discount or bonus',
                    'Conduct a satisfaction survey',
                    'Propose improvements to the service package'
                ]
            }
        elif probability >= 0.4:
            return {
                'type': 'warning',
                'title': 'Recommendation: Monitor customer behavior',
                'actions': [
                    'Track changes in service usage',
                    'Periodically send offers',
                    'Maintain communication'
                ]
            }
        else:
            return {
                'type': 'success',
                'title': 'Клієнт задоволений, продовжуйте поточну стратегію',
                'actions': [
                    'Продовжувати якісне обслуговування',
                    'Інформувати про нові послуги'
                ]
            }