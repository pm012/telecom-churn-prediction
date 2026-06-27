# app/core/visualizer.py
import matplotlib.pyplot as plt
import numpy as np

class ResultVisualizer:
    """Візуалізація результатів прогнозування"""
    
    @staticmethod
    def plot_risk_gauge(probability):
        """Відображення індикатора ризику"""
        fig, ax = plt.subplots(figsize=(10, 2))
        
        # Градієнт від зеленого до червоного
        gradient = np.linspace(0, 1, 100).reshape(1, -1)
        ax.imshow(gradient, cmap='RdYlGn_r', aspect='auto', extent=[0, 1, 0, 1])
        
        # Позначка поточної ймовірності
        ax.axvline(x=probability, color='black', linewidth=3, linestyle='--')
        
        # Текст з ймовірністю
        ax.text(probability, 0.5, f'{probability*100:.1f}%', 
               ha='center', va='center', fontsize=20, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.set_yticks([])
        ax.set_title('Ймовірність відтоку клієнта', fontsize=14, fontweight='bold')
        
        return fig
    
    @staticmethod
    def get_recommendation(probability):
        """Отримання рекомендацій на основі ймовірності"""
        if probability >= 0.7:
            return {
                'type': 'error',
                'title': 'Терміново вжити заходів для утримання клієнта!',
                'actions': [
                    'Запропонувати спеціальну знижку або бонус',
                    'Провести опитування задоволеності',
                    'Запропонувати покращення пакету послуг'
                ]
            }
        elif probability >= 0.4:
            return {
                'type': 'warning',
                'title': 'Рекомендується моніторинг поведінки клієнта',
                'actions': [
                    'Відстежувати зміни в використанні послуг',
                    'Періодично надсилати пропозиції',
                    'Підтримувати комунікацію'
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