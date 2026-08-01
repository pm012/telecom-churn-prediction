# app/main.py
import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
Виберіть модель у бічній панелі та введіть дані клієнта для отримання прогнозу.
""")

# Ініціалізація менеджера моделей
model_manager = ModelManager()
models, preprocessor, model_results = model_manager.load_all_models()

if not models:
    st.error("Не вдалося завантажити жодну модель. Переконайтеся, що моделі існують у папці 'models'.")
    st.stop()

# Отримання доступних моделей
available_models = list(models.keys())

# Відображення бічної панелі та вибір моделі
selected_model = Sidebar.render(available_models, model_results)

if selected_model is None:
    selected_model = available_models[0] if available_models else None

if selected_model is None:
    st.error("Немає доступних моделей для прогнозування.")
    st.stop()

# Завантаження обраної моделі
model = models[selected_model]

# Ініціалізація компонентів з ПЕРЕДАЧЕЮ НАЗВИ МОДЕЛІ
predictor = ChurnPredictor(model, preprocessor, model_name=selected_model)  # <-- Ось виправлення!
visualizer = ResultVisualizer()

# Визначення режиму роботи
mode = st.sidebar.radio(
    "Виберіть режим роботи:",
    ["Один клієнт", "Пакетна обробка (CSV файл)"]
)

# Основний контент
if mode == "Один клієнт":
    SinglePredictionComponent(predictor, visualizer).render()
else:
    BatchPredictionComponent(predictor).render()

# Футер
st.divider()
st.markdown(f"""
**Технології:** Python, Streamlit, Scikit-learn, LightGBM, XGBoost, Pandas

**Обрана модель:** {selected_model}
""")