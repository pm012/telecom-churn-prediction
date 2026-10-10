# app/components/single_prediction.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

class SinglePredictionComponent:
    """Component for single customer prediction"""
    
    def __init__(self, predictor, visualizer):
        self.predictor = predictor
        self.visualizer = visualizer
    
    def render(self):
        """Displaying the form and results"""
        st.header("Prediction for a Single Customer")
        st.markdown("Enter customer data to get a prediction.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Subscription Data")
            is_tv = st.selectbox(
                "TV Subscription", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )
            is_movie = st.selectbox(
                "Movie Package Subscription", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )
            subscription_age = st.number_input(
                "Subscription Duration (months)",
                min_value=0.0, max_value=120.0, value=12.0, step=0.5
            )
            
        
        with col2:
            st.subheader("Usage Data")            
            service_failures = st.number_input(
                "Service Failure Count",
                min_value=0, max_value=50, value=0, step=1
            )
            download_avg = st.number_input(
                "Average Download (Mbps)",
                min_value=0.0, max_value=500.0, value=20.0, step=5.0
            )
            upload_avg = st.number_input(
                "Average Upload (Mbps)",
                min_value=0.0, max_value=100.0, value=5.0, step=1.0
            )
           

        with col3:
            st.subheader("Contract Data")
            remaining_contract = st.number_input(
                "Remaining Contract (months)",
                 min_value=0.0, max_value=36.0, value=6.0, step=0.5
            )
            bill_avg = st.number_input(
                "Average Bill",
                min_value=0.0, max_value=500.0, value=50.0, step=5.0
            )
            download_over_limit = st.selectbox(
                "Download Over Limit", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )



        
        if st.button("Predict Churn", type="primary"):
            customer_data = {
                'is_tv_subscriber': is_tv,
                'is_movie_package_subscriber': is_movie,
                'subscription_age': subscription_age,
                'bill_avg': bill_avg,
                'remaining_contract': remaining_contract,
                'service_failure_count': service_failures,
                'download_avg': download_avg,
                'upload_avg': upload_avg,
                'download_over_limit': download_over_limit
            }
            
            try:
                result = self.predictor.predict_single(customer_data)
                self._display_results(result)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
    
    def _display_results(self, result):
        """Displaying prediction results"""
        st.divider()
        st.header("Prediction Results")
        
        # Information about the model
        model_used = result.get('model_used', 'N/A')
        st.info(f"**Model Used:** {model_used}")
        
        prob = result['probability']
        pred = result['prediction']
        risk = result['risk_level']
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Churn Probability", f"{prob*100:.1f}%")
        
        with col2:
            st.metric("Risk Level", risk)
        
        with col3:
            st.metric("Prediction", "Churn" if pred else "Stay")
        
        with col4:
            st.metric("Confidence in Prediction", f"{max(prob, 1-prob)*100:.1f}%")
        
        # Visualization
        st.subheader("Risk Visualization")
        fig = self.visualizer.plot_risk_gauge(prob)
        st.pyplot(fig)
        plt.close()
        
        # Recommendations
        st.subheader("Recommendations")
        rec = self.visualizer.get_recommendation(prob)
        
        if rec['type'] == 'error':
            st.error(f"**{rec['title']}**")
        elif rec['type'] == 'warning':
            st.warning(f"**{rec['title']}**")
        else:
            st.success(f"**{rec['title']}**")
        
        for action in rec['actions']:
            st.write(f"- {action}")