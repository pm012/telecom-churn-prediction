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
        
        col1, col2, col3 = st.columns(3)
        churn_count = results['churn_prediction'].sum()
        
        with col1:
            st.metric("Predicted Churn", f"{churn_count} customers")
        
        with col2:
            st.metric("Churn Rate", f"{churn_count/len(results)*100:.1f}%")
        
        with col3:
            high_risk = len(results[results['risk_level'] == 'Високий'])
            st.metric("High Risk", f"{high_risk} customers")
        
        if 'model_used' in results.columns:
            st.info(f"**Model Used:** {results['model_used'].iloc[0]}")
        
        st.subheader("Detailed Results")
        st.dataframe(results)
        
        csv = results.to_csv(index=False)
        st.download_button(
            label="Download Results (CSV)",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )