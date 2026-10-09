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

# Page configuration
st.set_page_config(
    page_title="Prediction of Customer Churn",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("Prediction of Customer Churn")
st.markdown("""
This application uses machine learning to predict the probability of customer churn.
Select a model in the sidebar and enter customer data to get a prediction.
""")

# Initialization of the model manager
model_manager = ModelManager()
models, preprocessor, model_results = model_manager.load_all_models()

if not models:
    st.error("Failed to load any models. Please ensure models exist in the 'models' directory.")
    st.stop()

# Getting available models
available_models = list(models.keys())

# Rendering sidebar and selecting model
selected_model = Sidebar.render(available_models, model_results)

if selected_model is None:
    selected_model = available_models[0] if available_models else None

if selected_model is None:
    st.error("No available models for prediction.")
    st.stop()

# Loading the selected model
model = models[selected_model]

# Initializing components with model name
predictor = ChurnPredictor(model, preprocessor, model_name=selected_model)  # <-- Ось виправлення!
visualizer = ResultVisualizer()

# Determining the working mode
mode = st.sidebar.radio(
    "Select working mode:",
    ["Single Client", "Batch Processing (CSV file)"]
)

# Main content
if mode == "Single Client":
    SinglePredictionComponent(predictor, visualizer).render()
else:
    BatchPredictionComponent(predictor).render()

# Footer
st.divider()
st.markdown(f"""
**Technologies:** Python, Streamlit, Scikit-learn, LightGBM, XGBoost, Pandas

**Selected Model:** {selected_model}
""")