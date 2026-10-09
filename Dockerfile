# Dockerfile
FROM python:3.12.3-slim

# Installation of working directory
WORKDIR /app

# Installation of system dependencies (required for lightgbm)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copying requirement files
COPY requirements.txt .

# Installing dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copying the entire project
COPY . .

# Creating folder for models
#RUN mkdir -p models

# Exposing port for Streamlit
EXPOSE 8501

# Command for running the Streamlit application
CMD ["streamlit", "run", "app/main.py"]