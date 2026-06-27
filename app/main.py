# app/main.py
import streamlit as st
import sys
import os

# Додаємо кореневу папку проекту до PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Імпорт компонентів
from app.core.model_manager import ModelManager
from app.core.predictor import ChurnPredictor
from app.core.visualizer import ResultVisualizer
from app.components.sidebar import Sidebar
from app.components.single_prediction import SinglePredictionComponent
from app.components.batch_prediction import BatchPredictionComponent

# Налаштування сторінки
st.set_page_config(
    page_title="Прогнозування відтоку клієнтів",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок
st.title("Прогнозування відтоку клієнтів телекомунікаційної компанії")
st.markdown("""
Цей застосунок використовує машинне навчання для прогнозування ймовірності відтоку клієнтів.
Модель навчена на даних телекомунікаційної компанії та дозволяє оцінити ризик втрати клієнта.
""")

# Ініціалізація менеджера моделі
model_manager = ModelManager()
model, preprocessor = model_manager.get_model()

if not model_manager.is_loaded():
    st.error("Не вдалося завантажити модель або препроцесор.")
    st.stop()

# Ініціалізація компонентів
predictor = ChurnPredictor(model, preprocessor)
visualizer = ResultVisualizer()

# Бічна панель
mode = Sidebar.render(
    model_info="""
    **Найкраща модель:** LightGBM
    
    **Метрики:**
    - Точність: 94.3%
    - Precision: 95.5%
    - Recall: 94.0%
    - F1-Score: 94.8%
    - ROC-AUC: 98.3%
    
    **Топ-3 важливі ознаки:**
    1. Середній рахунок
    2. Вік підписки
    3. Залишок контракту
    """
)

# Основний контент
if mode == "Один клієнт":
    SinglePredictionComponent(predictor, visualizer).render()
else:
    BatchPredictionComponent(predictor).render()

# Футер
st.divider()
st.markdown("""
**Технології:** Python, Streamlit, Scikit-learn, LightGBM, Pandas
""")