# app/components/sidebar.py
import streamlit as st

class Sidebar:
    """Компонент бічної панелі"""
    
    @staticmethod
    def render(model_info=None):
        """Відображення бічної панелі"""
        with st.sidebar:
            st.header("Налаштування")
            
            mode = st.radio(
                "Виберіть режим роботи:",
                ["Один клієнт", "Пакетна обробка (CSV файл)"]
            )
            
            st.divider()
            
            if model_info:
                st.header("Інформація про модель")
                st.info(model_info)
            
            st.divider()
            
            if st.button("Очистити результати"):
                st.cache_data.clear()
                st.rerun()
            
            return mode