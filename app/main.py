# app/main.py
import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Configure import paths for Python
current_dir = os.path.dirname(os.path.abspath(__file__)) # папка 'app'
project_root = os.path.dirname(current_dir)              # корінь проекту
if project_root not in sys.path:
    sys.path.append(project_root)

# Define absolute path to favicon.png inside the app/assets folder
favicon_path = os.path.join(current_dir, "assets", "favicon.png")

# FIRST CALL TO STREAMLIT (Mandatory!)
st.set_page_config(
    page_title="Prediction of Customer Churn",
    page_icon=favicon_path,  
    layout="wide",  
    initial_sidebar_state="expanded"
)

from app.core.model_manager import ModelManager
from app.core.predictor import ChurnPredictor
from app.core.visualizer import ResultVisualizer
from app.components.sidebar import Sidebar
from app.components.single_prediction import SinglePredictionComponent
from app.components.batch_prediction import BatchPredictionComponent


# Header
st.title("Prediction of Customer Churn")

# Initialization of the model manager
model_manager = ModelManager()
models, preprocessor, model_results = model_manager.load_all_models()

if not models:
    st.error("Failed to load any models. Please ensure models exist in the 'models' directory.")
    st.stop()

available_models = list(models.keys())

# Rendering sidebar (model selection + metrics + mode + reset)
selected_model, mode = Sidebar.render(available_models, model_results)

if selected_model is None:
    selected_model = available_models[0] if available_models else None

if selected_model is None:
    st.error("No available models for prediction.")
    st.stop()

# === Selected Model — underer header, the same font as the title ===
st.markdown(f"## \U0001F449 Selected Model: {selected_model}")

st.markdown("""
This application uses machine learning to predict the probability of customer churn.
Select a model in the sidebar and enter customer data to get a prediction.
""")
st.markdown("**Technologies:** Python, Streamlit, Scikit-learn, LightGBM, XGBoost, Pandas")

# Loading the selected model
model = models[selected_model]

# Initializing components
predictor = ChurnPredictor(model, preprocessor, model_name=selected_model)
visualizer = ResultVisualizer()

# Main content
if mode == "Single Client":
    SinglePredictionComponent(predictor, visualizer).render()
else:
    BatchPredictionComponent(predictor).render()

# === Comparison of Models — placed intentionally above the page footer===
st.divider()
Sidebar.render_model_comparison(model_results, selected_model)

# Footer
st.divider()
st.markdown(f"""
Copyright © {datetime.now().year}. All rights reserved. \n
Created by Serhii Kroshka.
""")