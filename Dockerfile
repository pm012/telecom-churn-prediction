# Dockerfile
FROM python:3.12.3-slim

# Встановлення робочої директорії
WORKDIR /app

# Встановлення системних залежностей (необхідно для lightgbm)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копіювання файлів з вимогами
COPY requirements.txt .

# Встановлення залежностей
RUN pip install --no-cache-dir -r requirements.txt

# Копіювання всього проекту
COPY . .

# Створення папки для моделей
#RUN mkdir -p models

# Відкриття порту для Streamlit
EXPOSE 8501

# Команда для запуску
CMD ["streamlit", "run", "app/main.py"]