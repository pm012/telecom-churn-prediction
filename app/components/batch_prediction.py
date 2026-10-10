# app/components/batch_prediction.py
import streamlit as st
import pandas as pd

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
        - `service_failure_count`, `download_avg`, `upload_avg`, `download_over_limit`
        """)
        
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
            # --- Bar chart: churn rate by risk levels ---
            try:
                import plotly.express as px
                
                risk_stats = results.groupby('risk_level').agg(
                    total=('churn_prediction', 'size'),
                    churned=('churn_prediction', 'sum'),
                ).reset_index()
                risk_stats['churn_rate'] = risk_stats['churned'] / risk_stats['total']
                
                # Sort by logical order of risk
                risk_stats['_order'] = risk_stats['risk_level'].map(
                    {'Low': 0, 'Medium': 1, 'High': 2}
                ).fillna(99)
                risk_stats = risk_stats.sort_values('_order').drop(columns='_order')
                
                fig_bar = px.bar(
                    risk_stats,
                    x='risk_level',
                    y='churn_rate',
                    color='risk_level',
                    color_discrete_map={
                        'Low': risk_colors['Low'],
                        'Medium': risk_colors['Medium'],
                        'High': risk_colors['High'],
                    },
                    text=risk_stats['churn_rate'].apply(lambda x: f"{x*100:.0f}%"),
                    labels={'risk_level': 'Risk Level', 'churn_rate': 'Churn Rate'},
                )
                fig_bar.update_traces(
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Churn Rate: %{y:.1%}<extra></extra>'
                )
                fig_bar.update_layout(
                    showlegend=False,
                    yaxis_tickformat='.0%',
                    yaxis_range=[0, 1.1],
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=350,
                )
                st.plotly_chart(fig_bar, width='stretch')
            except ImportError:
                # Fallback without plotly
                risk_stats = results.groupby('risk_level')['churn_prediction'].mean()
                st.bar_chart(risk_stats)
            except Exception as e:
                st.warning(f"Bar chart unavailable: {e}")
        
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