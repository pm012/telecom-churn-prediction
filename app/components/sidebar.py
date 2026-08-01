# app/components/sidebar.py
import streamlit as st
import pandas as pd

class Sidebar:
    """Компонент бічної панелі"""
    
    @staticmethod
    def render(available_models, model_results=None):
        """Відображення бічної панелі"""
        with st.sidebar:
            st.header("Налаштування")
            
            # Вибір моделі
            if available_models:
                selected_model = st.selectbox(
                    "Виберіть модель:",
                    available_models,
                    help="Виберіть модель для прогнозування"
                )
            else:
                selected_model = None
                st.warning("Моделі не знайдено")
            
            st.divider()
            
            # Відображення метрик для обраної моделі
            if selected_model and model_results is not None:
                st.header("Метрики моделі")
                
                # Пошук метрик для обраної моделі
                model_row = model_results[model_results['Модель'] == selected_model]
                if not model_row.empty:
                    row = model_row.iloc[0]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Точність", f"{row['accuracy']*100:.1f}%")
                        st.metric("Precision", f"{row['precision']*100:.1f}%")
                    with col2:
                        st.metric("Recall", f"{row['recall']*100:.1f}%")
                        st.metric("ROC-AUC", f"{row['roc_auc']:.4f}")
                    
                    st.progress(row['roc_auc'], text="ROC-AUC")
                else:
                    st.info("Метрики для цієї моделі не знайдено")
            
            st.divider()
            
            # Порівняння всіх моделей
            if model_results is not None and len(model_results) > 0:
                st.header("Порівняння моделей")
                
                # Показати топ моделі
                top_models = model_results.head(5)
                for _, row in top_models.iterrows():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        model_name = row['Модель']
                        if model_name == selected_model:
                            model_name = f"⭐ {model_name}"
                        st.write(f"**{model_name}**")
                    with col2:
                        st.write(f"{row['roc_auc']:.4f}")
                    st.progress(row['roc_auc'], text="")
                    
                    # Додаткові деталі при наведенні
                    with st.expander(f"Деталі для {row['Модель']}"):
                        st.write(f"Accuracy: {row['accuracy']*100:.2f}%")
                        st.write(f"Precision: {row['precision']*100:.2f}%")
                        st.write(f"Recall: {row['recall']*100:.2f}%")
                        st.write(f"F1: {row['f1']:.4f}")
            
            st.divider()
            
            # Кнопка очищення
            if st.button("Очистити результати"):
                st.cache_data.clear()
                st.rerun()
            
            return selected_model