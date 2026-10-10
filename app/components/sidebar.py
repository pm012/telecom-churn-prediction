# app/components/sidebar.py
import streamlit as st
import pandas as pd

class Sidebar:
    """Component for the sidebar"""
    
    @staticmethod
    def render(available_models, model_results=None):
        """Displaying the sidebar. Returns (selected_model, mode)."""
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
            
            # === Working mode ===
            st.header("Working Mode")
            mode = st.radio(
                "Select working mode:",
                ["Single Client", "Batch Processing (CSV file)"],
                key="working_mode"
            )
            
            st.divider()
            
            # Reset button
            if st.button("Reset Results", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
            
            return selected_model, mode
    
    @staticmethod   
    def render_model_comparison(model_results, selected_model=None):
        """Displaying model comparison table (rendered in main area)."""
        if model_results is None or len(model_results) == 0:
            return
        
        st.header("📊 Comparison of Models")
        st.markdown("All trained models, sorted by ROC-AUC (best first). The current model is highlighted.")
        
        # Prepare display DataFrame
        display_df = model_results.copy()
        
        # Add star to selected model
        display_df['Model'] = display_df['Model'].apply(
            lambda x: f"⭐ {x}" if x == selected_model else x
        )
        
        # Format numeric columns
        display_df['accuracy'] = display_df['accuracy'].apply(lambda x: f"{x*100:.2f}%")
        display_df['precision'] = display_df['precision'].apply(lambda x: f"{x*100:.2f}%")
        display_df['recall'] = display_df['recall'].apply(lambda x: f"{x*100:.2f}%")
        display_df['f1'] = display_df['f1'].apply(lambda x: f"{x:.4f}")
        display_df['roc_auc'] = display_df['roc_auc'].apply(lambda x: f"{x:.4f}")
        
        # Rename columns
        display_df = display_df.rename(columns={
            'Model': 'Model',
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1': 'F1',
            'roc_auc': 'ROC-AUC',
        })
        
        # Select and order columns
        cols_order = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
        display_df = display_df[[c for c in cols_order if c in display_df.columns]]
        
        # Sort by ROC-AUC descending
        display_df = display_df.sort_values(
            'ROC-AUC',
            ascending=False,
        ).reset_index(drop=True)
        
        # Add rank
        display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
        
        # Render as table — NEW API: width='stretch' instead of use_container_width=True
        st.dataframe(
            display_df,
            width='stretch',       # <- new API
            hide_index=True,
        )
        
        # Bar chart — fallback (to make it more visual)
        with st.expander("📈 Visual comparison (ROC-AUC)", expanded=True):
            # Simplified to built-in horizontal bar chart for maximum compatibility
            chart_df = model_results.set_index('Model')['roc_auc'].sort_values(ascending=False)
            st.bar_chart(chart_df, horizontal=True, height=300, color="#4B8BBE")
            st.caption(f"⭐ Selected: **{selected_model}**")