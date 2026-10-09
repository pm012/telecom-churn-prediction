# app/components/sidebar.py
import streamlit as st
import pandas as pd

class Sidebar:
    """Component for the sidebar"""
    
    @staticmethod
    def render(available_models, model_results=None):
        """Displaying the sidebar"""
        with st.sidebar:
            st.header("Settings")
            
            # Model selection
            if available_models:
                selected_model = st.selectbox(
                    "Select a model:",
                    available_models,
                    help="Select a model for prediction"
                )
            else:
                selected_model = None
                st.warning("Models not found. Please ensure models exist in the 'models' directory.")
            
            st.divider()
            
            # Displaying metrics for the selected model
            if selected_model and model_results is not None:
                st.header("Model Metrics")
                
                # Searching for metrics for the selected model
                model_row = model_results[model_results['Model'] == selected_model]
                if not model_row.empty:
                    row = model_row.iloc[0]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accuracy", f"{row['accuracy']*100:.1f}%")
                        st.metric("Precision", f"{row['precision']*100:.1f}%")
                    with col2:
                        st.metric("Recall", f"{row['recall']*100:.1f}%")
                        st.metric("ROC-AUC", f"{row['roc_auc']:.4f}")
                    
                    st.progress(row['roc_auc'], text="ROC-AUC")
                else:
                    st.info("Metrics for this model not found")
            
            st.divider()
            
            # Comparison of all models
            if model_results is not None and len(model_results) > 0:
                st.header("Comparison of Models")
                
                # Show top models
                top_models = model_results.head(5)
                for _, row in top_models.iterrows():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        model_name = row['Model']
                        if model_name == selected_model:
                            model_name = f"⭐ {model_name}"
                        st.write(f"**{model_name}**")
                    with col2:
                        st.write(f"{row['roc_auc']:.4f}")
                    st.progress(row['roc_auc'], text="")
                    
                    # Additional details on hover
                    with st.expander(f"Details for {row['Model']}"):
                        st.write(f"Accuracy: {row['accuracy']*100:.2f}%")
                        st.write(f"Precision: {row['precision']*100:.2f}%")
                        st.write(f"Recall: {row['recall']*100:.2f}%")
                        st.write(f"F1: {row['f1']:.4f}")
            
            st.divider()
            
            # Reset button
            if st.button("Reset Results"):
                st.cache_data.clear()
                st.rerun()
            
            return selected_model