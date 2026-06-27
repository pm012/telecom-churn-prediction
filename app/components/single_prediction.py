# app/components/single_prediction.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

class SinglePredictionComponent:
    """Компонент для прогнозування одного клієнта"""
    
    def __init__(self, predictor, visualizer):
        self.predictor = predictor
        self.visualizer = visualizer
    
    def render(self):
        """Відображення форми та результатів"""
        st.header("Прогноз для одного клієнта")
        st.markdown("Введіть дані клієнта для отримання прогнозу.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Демографічні дані")
            is_tv = st.selectbox(
                "Підписка на ТБ", [0, 1],
                format_func=lambda x: "Так" if x else "Ні"
            )
            is_movie = st.selectbox(
                "Підписка на кіно", [0, 1],
                format_func=lambda x: "Так" if x else "Ні"
            )
            subscription_age = st.number_input(
                "Термін підписки (місяців)",
                min_value=0.0, max_value=120.0, value=12.0, step=0.5
            )
        
        with col2:
            st.subheader("Дані використання")
            bill_avg = st.number_input(
                "Середній рахунок",
                min_value=0.0, max_value=500.0, value=50.0, step=5.0
            )
            remaining_contract = st.number_input(
                "Залишок контракту (місяців)",
                min_value=0.0, max_value=36.0, value=6.0, step=0.5
            )
            service_failures = st.number_input(
                "Кількість збоїв",
                min_value=0, max_value=50, value=0, step=1
            )
            download_avg = st.number_input(
                "Середній download (Mbps)",
                min_value=0.0, max_value=500.0, value=20.0, step=5.0
            )
            upload_avg = st.number_input(
                "Середній upload (Mbps)",
                min_value=0.0, max_value=100.0, value=5.0, step=1.0
            )
            download_over_limit = st.selectbox(
                "Перевищення ліміту download", [0, 1],
                format_func=lambda x: "Так" if x else "Ні"
            )
        
        if st.button("Прогнозувати відтік", type="primary"):
            customer_data = {
                'is_tv_subscriber': is_tv,
                'is_movie_package_subscriber': is_movie,
                'subscription_age': subscription_age,
                'bill_avg': bill_avg,
                'reamining_contract': remaining_contract,
                'service_failure_count': service_failures,
                'download_avg': download_avg,
                'upload_avg': upload_avg,
                'download_over_limit': download_over_limit
            }
            
            try:
                result = self.predictor.predict_single(customer_data)
                self._display_results(result)
            except Exception as e:
                st.error(f"Помилка при прогнозуванні: {e}")
    
    def _display_results(self, result):
        """Відображення результатів прогнозу"""
        st.divider()
        st.header("Результат прогнозування")
        
        prob = result['probability']
        pred = result['prediction']
        risk = result['risk_level']
        
        # Метрики
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Ймовірність відтоку", f"{prob*100:.1f}%")
        
        with col2:
            st.metric("Рівень ризику", risk)
        
        with col3:
            st.metric("Прогноз", "Відтік" if pred else "Залишиться")
        
        with col4:
            st.metric("Довіра до прогнозу", f"{max(prob, 1-prob)*100:.1f}%")
        
        # Візуалізація
        st.subheader("Візуалізація ризику")
        fig = self.visualizer.plot_risk_gauge(prob)
        st.pyplot(fig)
        plt.close()
        
        # Рекомендації
        st.subheader("Рекомендації")
        rec = self.visualizer.get_recommendation(prob)
        
        if rec['type'] == 'error':
            st.error(f"**{rec['title']}**")
        elif rec['type'] == 'warning':
            st.warning(f"**{rec['title']}**")
        else:
            st.success(f"**{rec['title']}**")
        
        for action in rec['actions']:
            st.write(f"- {action}")