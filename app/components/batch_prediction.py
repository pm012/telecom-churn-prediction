# app/components/batch_prediction.py
import streamlit as st
import pandas as pd

from src.prediction_policy import (
    CHURN_DECISION_THRESHOLD,
    HIGH_RISK_THRESHOLD,
    LOW_RISK_THRESHOLD,
)


class BatchPredictionComponent:
    """Component for batch prediction"""
    
    REQUIRED_COLUMNS = [
        'is_tv_subscriber', 'is_movie_package_subscriber', 'subscription_age',
        'bill_avg', 'remaining_contract',  # Same as in the training data
        'service_failure_count', 'download_avg', 'upload_avg', 'download_over_limit'
    ]
    
    def __init__(self, predictor):
        self.predictor = predictor
    
    def render(self):
        """Displaying the batch processing form"""
        st.header("Batch Processing of Data")
        st.markdown("""
        Upload a CSV file with customer data for batch prediction.
        
        **Important:** The file must contain the following columns:
        - `is_tv_subscriber`, `is_movie_package_subscriber`, `subscription_age`
        - `bill_avg`, `remaining_contract`
        - `service_failure_count`, `download_avg`, `upload_avg`, `download_over_limit` (integer count, 0-7)
        """)
        st.caption(
            f"Churn cutoff: {CHURN_DECISION_THRESHOLD:.0%} (equal-cost assumption). "
            f"Risk bands: Low < {LOW_RISK_THRESHOLD:.0%}, "
            f"Medium < {HIGH_RISK_THRESHOLD:.0%}, High >= {HIGH_RISK_THRESHOLD:.0%}."
        )
        
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="File must contain the same columns as the training data"
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                self._display_data_preview(data)
                
                if self._validate_columns(data):
                    if st.button("Predict", type="primary"):
                        self._process_batch(data)
                
            except Exception as e:
                st.error(f"Error occurred while processing the file: {e}")
    
    def _display_data_preview(self, data):
        """Displaying the preview of the data"""
        st.subheader("Data Preview")
        st.dataframe(data.head(10))
        st.caption(f"Total customers: {len(data)}")
    
    def _validate_columns(self, data):
        """Validating data columns"""
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in data.columns]
        if missing_cols:
            st.warning(f"Missing columns: {missing_cols}")
            return False
        raw_counts = data['download_over_limit']
        counts = pd.to_numeric(raw_counts, errors='coerce')
        invalid_counts = raw_counts.notna() & (
            counts.isna() | (counts < 0) | (counts > 7) | (counts % 1 != 0)
        )
        if invalid_counts.any():
            st.warning("`download_over_limit` must be an integer count between 0 and 7.")
            return False
        return True
    
    def _process_batch(self, data):
        """Processing batch data"""
        with st.spinner("Processing data..."):
            results = self.predictor.predict_batch(data)
            self._display_batch_results(results)
    
    def _display_batch_results(self, results):
        """Displaying batch processing results"""
        st.subheader("Results Statistics")
        
        # === Metrics ===
        col1, col2, col3 = st.columns(3)
        churn_count = results['churn_prediction'].sum()
        
        # Risk level matching (case-insensitive)
        HIGH_RISK_LABELS = {'high', 'високий'}
        high_risk_mask = results['risk_level'].str.strip().str.lower().isin(HIGH_RISK_LABELS)
        high_risk = high_risk_mask.sum()
        
        with col1:
            st.metric("Predicted Churn", f"{churn_count} customers")
        
        with col2:
            st.metric("Churn Rate", f"{churn_count/len(results)*100:.1f}%")
        
        with col3:
            st.metric("High Risk", f"{high_risk} customers")
        
        if 'model_used' in results.columns:
            st.info(f"**Model Used:** {results['model_used'].iloc[0]}")
        
        # === Visualization ===
        st.subheader("Risk Distribution")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # --- Pie chart risks distribution ---
            try:
                import plotly.express as px
                
                risk_counts = results['risk_level'].value_counts().reset_index()
                risk_counts.columns = ['Risk Level', 'Count']
                
                # Order and colors
                risk_order = ['Low', 'Medium', 'High']
                risk_colors = {
                    'Low': '#2ECC71',      # green
                    'Medium': '#F39C12',   # orange
                    'High': '#E74C3C',     # red
                }
                
                # Normalize names to English for color mapping
                risk_counts['Risk Level Norm'] = risk_counts['Risk Level'].str.strip().str.title()
                
                fig_pie = px.pie(
                    risk_counts,
                    values='Count',
                    names='Risk Level',
                    color='Risk Level',
                    color_discrete_map={
                        'Low': risk_colors['Low'],
                        'Medium': risk_colors['Medium'],
                        'High': risk_colors['High'],
                    },
                    hole=0.4,  # donut
                )
                fig_pie.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>'
                )
                fig_pie.update_layout(
                    showlegend=True,
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=350,
                )
                st.plotly_chart(fig_pie, width='stretch')
            except ImportError:
                # Fallback: simple bar chart, if plotly is not available
                st.bar_chart(
                    results['risk_level'].value_counts(),
                    color="#4B8BBE"
                )
            except Exception as e:
                st.warning(f"Pie chart unavailable: {e}")
        
        with viz_col2:
            try:
                import plotly.express as px
                
                fig = px.histogram(
                    results,
                    x='churn_probability',
                    nbins=20,
                    color='risk_level',
                    color_discrete_map={
                        'Low': '#2ECC71',
                        'Medium': '#F39C12',
                        'High': '#E74C3C',
                    },
                    category_orders={'risk_level': ['Low', 'Medium', 'High']},
                    labels={'churn_probability': 'Churn Probability', 'count': 'Customers'},
                )
                fig.update_layout(
                    xaxis_title='Churn Probability',
                    yaxis_title='Number of Customers',
                    xaxis=dict(tickformat='.0%', range=[0, 1]),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    margin=dict(t=40, b=20, l=20, r=20),
                    height=350,
                    bargap=0.05,
                )
                st.plotly_chart(fig, width='stretch')
            except Exception as e:
                st.warning(f"Histogram unavailable: {e}")
                    
        # === Detailed Results ===
        st.subheader("Detailed Results")
        st.dataframe(results, width='stretch', hide_index=True)
        
        csv = results.to_csv(index=False)
        st.download_button(
            label="Download Results (CSV)",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )