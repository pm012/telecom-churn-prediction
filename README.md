# Прогнозування відтоку клієнтів для телекомунікаційної компанії

## Мета проєкту

Цей проєкт розробляє модель машинного навчання для прогнозування ймовірності відтоку клієнтів телекомунікаційної компанії на основі історичних даних. Основна мета — допомогти бізнесу виявляти клієнтів з високим ризиком відтоку та вживати вчасні заходи для їхнього утримання.

## Що реалізовано

- аналіз вхідних даних та попередня обробка ознак;
- порівняння кількох моделей класифікації;
- оцінка якості моделей за Accuracy, Precision, Recall, F1-score та ROC-AUC;
- веб-інтерфейс на Streamlit для прогнозування як для одного клієнта, так і для пакетної обробки CSV-файлів;
- контейнеризація проєкту за допомогою Docker.

## Структура проєкту

```text
telecom-churn-prediction/
├── app/
│   ├── components/
│   │   ├── batch_prediction.py
│   │   └── single_prediction.py
│   ├── core/
│   │   ├── model_manager.py
│   │   ├── predictor.py
│   │   └── visualizer.py
│   └── main.py
├── data/
│   ├── processed/
│   │   └── processed_data.csv
│   └── raw/
│       └── internet_service_churn.csv
├── docs/
│   └── eda_analysis.md
├── legacy_scripts/
│   ├── test_extreme_cases.py
│   └── test_predictions.py
├── models/
│   ├── best_model.pkl
│   ├── model_results.csv
│   ├── plots/
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│   │   ├── predictions_analysis.csv
│   │   └── roc_curve.png
│   └── preprocessor.pkl
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   └── model_training.py
├── tests/
│   ├── conftest.py
│   ├── test_prediction_pipeline.py
│   └── test_preprocessing.py
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
└── README.md
```

## Огляд даних

У роботі використано датасет із 72 274 рядками та 11 стовпцями. Цільова змінна — `churn`.

Основні характеристики датасету:

- частка клієнтів, які відмовилися від послуг: приблизно 55.4%;
- є пропуски в полях `reamining_contract`, `download_avg` та `upload_avg`;
- серед ознак присутні як бінарні ознаки підписок, так і числові показники використання послуг і фінансової активності.

## EDA та обґрунтування підходу

На етапі EDA було перевірено:

- розподіл цільової змінної;
- наявність пропусків у даних;
- базові статистичні характеристики числових ознак;
- структуру та типи ознак, необхідних для подальшої обробки.

Ці кроки були важливими, оскільки вони вплинули на вибір стратегій підготовки даних:

- пропуски в числових ознаках було заповнено медіаною, а для категоріальних — модою;
- для типових числових ознак застосовано стандартизацію;
- для категоріальних ознак використано кодування через `LabelEncoder`;
- через відносно збалансований розподіл цільової змінної було використано стратифікований розділ даних на train/test.

## Попередня обробка даних

Процес підготовки даних включає:

1. завантаження CSV-файлу;
2. видалення ідентифікатора `id`;
3. обробку пропусків;
4. кодування категоріальних ознак;
5. стандартизацію числових ознак;
6. розділення на тренувальний і тестовий набори.

## Навчання моделей

Було протестовано кілька алгоритмів:

- Logistic Regression;
- Random Forest;
- Gradient Boosting;
- XGBoost;
- LightGBM;
- Decision Tree;
- KNN.

Найкращий результат показала модель XGBoost. Основні метрики на тестовому наборі:

- Accuracy: 0.9413
- Precision: 0.9550
- Recall: 0.9383
- F1-score: 0.9466
- ROC-AUC: 0.9822

## Запуск проєкту

### Локальний запуск

1. Встановіть залежності:

```bash
pip install -r requirements.txt
```

2. Навчіть модель:

```bash
python src/model_training.py
```

3. Запустіть веб-застосунок:

```bash
streamlit run app/main.py
```

### Запуск через Docker

```bash
docker compose build --no-cache
docker compose up -d
```

Після цього застосунок буде доступний за адресою:

```text
http://localhost:8501
```

## Тестування

Для запуску тестів використовуйте:

```bash
python -m pytest -q tests --disable-warnings
```

Якщо ви працюєте у віртуальному середовищі, спочатку активуйте його:

```bash
source .venv/bin/activate
python -m pytest -q tests --disable-warnings
```

## Приклад використання

У веб-інтерфейсі доступні два режими:

- прогноз для одного клієнта;
- пакетне прогнозування для CSV-файлу.

Приклад вхідних даних для одного клієнта:

```json
{
  "is_tv_subscriber": 1,
  "is_movie_package_subscriber": 0,
  "subscription_age": 12,
  "bill_avg": 45,
  "reamining_contract": 6,
  "service_failure_count": 0,
  "download_avg": 20,
  "upload_avg": 5,
  "download_over_limit": 0
}
```

Результатом буде:

- ймовірність відтоку;
- прогнозний клас (відтік / залишиться);
- рівень ризику;
- рекомендації для роботи з клієнтом.

## English version

## Project goal

This project develops a machine learning model to predict the probability of customer churn in a telecommunications company based on historical customer data. The main objective is to help the business identify customers at high risk of leaving and take timely retention actions.

## What is implemented

- exploratory data analysis and preprocessing of features;
- comparison of several classification models;
- evaluation of model quality using Accuracy, Precision, Recall, F1-score, and ROC-AUC;
- a Streamlit web interface for both single-customer and batch CSV predictions;
- Docker-based containerization.

## Dataset overview

The dataset contains 72,274 rows and 11 columns. The target variable is `churn`.

Key characteristics:

- churn rate is about 55.4%;
- missing values are present in `reamining_contract`, `download_avg`, and `upload_avg`;
- the dataset includes both binary subscription indicators and numeric usage and billing features.

## EDA and modeling rationale

EDA was used to inspect:

- the target class distribution;
- missing values;
- basic descriptive statistics of numerical features;
- the structure and types of available features.

These checks justified the preprocessing strategy:

- missing numerical values were imputed with the median;
- categorical values were filled with the mode;
- numeric features were standardized;
- a stratified train/test split was used because the target is reasonably balanced.

## Model training

Several algorithms were tested, including Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, Decision Tree, and KNN.

The best-performing model is XGBoost with the following test metrics:

- Accuracy: 0.9413
- Precision: 0.9550
- Recall: 0.9383
- F1-score: 0.9466
- ROC-AUC: 0.9822

## Run locally

```bash
pip install -r requirements.txt
python src/model_training.py
streamlit run app/main.py
```

## Run with Docker

```bash
docker compose build --no-cache
docker compose up -d
```

For more details about the EDA rationale, see [docs/eda_analysis.md](docs/eda_analysis.md).
