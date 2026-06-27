# app/components/batch_prediction.py
import streamlit as st
import pandas as pd

class BatchPredictionComponent:
    """Компонент для пакетного прогнозування"""
    
    REQUIRED_COLUMNS = [
        'is_tv_subscriber', 'is_movie_package_subscriber', 'subscription_age',
        'bill_avg', 'reamining_contract', 'service_failure_count',
        'download_avg', 'upload_avg', 'download_over_limit'
    ]
    
    def __init__(self, predictor):
        self.predictor = predictor
    
    def render(self):
        """Відображення форми пакетної обробки"""
        st.header("Пакетна обробка даних")
        st.markdown("Завантажте CSV файл з даними клієнтів для масового прогнозування.")
        
        uploaded_file = st.file_uploader(
            "Виберіть CSV файл",
            type=['csv'],
            help="Файл повинен містити ті ж колонки, що й навчальні дані"
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                self._display_data_preview(data)
                
                if self._validate_columns(data):
                    if st.button("Прогнозувати", type="primary"):
                        self._process_batch(data)
                
            except Exception as e:
                st.error(f"Помилка при обробці файлу: {e}")
    
    def _display_data_preview(self, data):
        """Відображення попереднього перегляду даних"""
        st.subheader("Попередній перегляд даних")
        st.dataframe(data.head(10))
        st.caption(f"Загалом {len(data)} клієнтів")
    
    def _validate_columns(self, data):
        """Валідація колонок даних"""
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in data.columns]
        if missing_cols:
            st.warning(f"Відсутні колонки: {missing_cols}")
            return False
        return True
    
    def _process_batch(self, data):
        """Обробка пакетних даних"""
        with st.spinner("Обробка даних..."):
            results = self.predictor.predict_batch(data)
            self._display_batch_results(results)
    
    def _display_batch_results(self, results):
        """Відображення результатів пакетної обробки"""
        st.subheader("Статистика результатів")
        
        col1, col2, col3 = st.columns(3)
        churn_count = results['churn_prediction'].sum()
        
        with col1:
            st.metric("Прогнозований відтік", f"{churn_count} клієнтів")
        
        with col2:
            st.metric("Рівень відтоку", f"{churn_count/len(results)*100:.1f}%")
        
        with col3:
            high_risk = len(results[results['risk_level'] == 'Високий'])
            st.metric("Високий ризик", f"{high_risk} клієнтів")
        
        st.subheader("Детальні результати")
        st.dataframe(results)
        
        # Завантаження результатів
        csv = results.to_csv(index=False)
        st.download_button(
            label="Завантажити результати (CSV)",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )