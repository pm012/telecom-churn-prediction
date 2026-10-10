# Прогнозування відтоку клієнтів для телекомунікаційної компанії

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

- approximately 55.4% of records are churn cases, so the target is mildly imbalanced;
- missing values occur in `remaining_contract` (21,572; 29.85%), `download_avg` (381; 0.53%), and `upload_avg` (381; 0.53%);
- predictors include binary subscription indicators and numeric usage, billing, contract, and over-limit counts (`download_over_limit` ranges from 0 to 7 in the source data).

## EDA and modeling rationale

EDA was used to inspect:

- the target class distribution;
- missing values;
- basic descriptive statistics of numerical features;
- the structure and types of available features.

The preprocessing pipeline adds a `remaining_contract_missing` indicator before median-imputing that field, median-imputes numeric fields, and mode-imputes the binary subscription flags. It fits all preprocessing inside each cross-validation training fold. A stratified 80/20 split preserves target proportions, and the scaler is never fitted on holdout rows.

## Model training

Several algorithms were tested, including Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, Decision Tree, and KNN.

Hyperparameter search uses repeated stratified three-fold cross-validation (two repeats), with preprocessing fitted inside each fold. The selected model is chosen by mean training-fold ROC-AUC; the stratified 20% holdout is used only for final reporting. Calibration is summarized with the Brier score and a reliability plot.

The churn classification cutoff is 0.50 under an equal false-positive/false-negative cost assumption. Risk levels consistently use Low below 0.30, Medium from 0.30 to below 0.70, and High at or above 0.70. These are explicit initial policy choices rather than business-optimized thresholds.

The latest training run evaluated seven models using repeated stratified three-fold cross-validation (two repeats) on training data, then reported metrics on an untouched stratified 20% holdout. LightGBM was selected by mean CV ROC-AUC. Holdout metrics are reporting-only.

| Model | CV ROC-AUC (mean ± std) | Accuracy | Precision | Recall | F1 | Holdout ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | **0.9818 ± 0.0009** | 0.9428 | 0.9555 | **0.9406** | 0.9480 | **0.9829** | 0.0448 |
| Random Forest | 0.9810 ± 0.0004 | 0.9409 | 0.9566 | 0.9358 | 0.9461 | 0.9819 | 0.0461 |
| Gradient Boosting | 0.9808 ± 0.0009 | **0.9438** | 0.9574 | 0.9404 | **0.9489** | 0.9828 | **0.0446** |
| XGBoost | 0.9804 ± 0.0008 | 0.9412 | **0.9576** | 0.9353 | 0.9463 | 0.9811 | 0.0469 |
| Decision Tree | 0.9708 ± 0.0011 | 0.9376 | 0.9504 | 0.9362 | 0.9433 | 0.9739 | 0.0507 |
| KNN | 0.9584 ± 0.0005 | 0.9202 | 0.9369 | 0.9177 | 0.9272 | 0.9606 | 0.0656 |
| Logistic Regression | 0.9334 ± 0.0011 | 0.8758 | 0.8745 | 0.9057 | 0.8899 | 0.9309 | 0.0974 |

On the 14,455-row holdout, LightGBM classified 13,628 cases correctly (94.28%), with 6,094 true negatives, 351 false positives, 476 false negatives, and 7,534 true positives. The Brier score is 0.0448. Gradient Boosting has the highest holdout Accuracy and F1 and the lowest Brier score; XGBoost has the highest Precision. See [the comprehensive results report](docs/report.md) and [EDA analysis](docs/eda_analysis.md) for plots, segment diagnostics, and caveats.

## Run locally
The project pins Python 3.12.14 in `.python-version`. Create an environment and install dependencies:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
python src/model_training.py
python src/model_evaluation.py
streamlit run app/main.py
```

To run preprocessing, model creation, and evaluation together without deleting existing model files:

```bash
PYTHON=./.venv/bin/python bash run_all.sh
```

## Run with Docker

```bash
docker compose build --no-cache
docker compose up -d
```

For the model evaluation, saved plots, and interpretation caveats, see [docs/report.md](docs/report.md). The [EDA analysis](docs/eda_analysis.md) and [technical specification](docs/technical_spec_en.md) provide the analysis rationale and project requirements.

## Project structure

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
│   ├── eda_analysis.md
│   ├── report.md
│   └── technical_spec_en.md
├── models/
│   ├── best_model.pkl
│   ├── model_results.csv
│   ├── plots/
│   │   ├── calibration_curve.png
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│   │   ├── holdout_metrics.csv
│   │   ├── missingness_diagnostics.csv
│   │   ├── predictions_analysis.csv
│   │   └── roc_curve.png
│   └── preprocessor.pkl
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   ├── prediction_policy.py
│   └── model_training.py
├── tests/
│   ├── conftest.py
│   ├── test_prediction_policy.py
│   ├── test_prediction_pipeline.py
│   └── test_preprocessing.py
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
└── README.md
```

# Ukrainian version
## Мета проекту

Цей проєкт реалізує модель машинного навчання для прогнозування ймовірності відтоку клієнтів телекомунікаційної компанії на основі історичних даних. Основна мета - допомогти бізнесу виявляти клієнтів з високим ризиком відтоку та вживати вчасні заходи для їхнього утримання.

## Що реалізовано

- аналіз вхідних даних та попередня обробка ознак;
- порівняння кількох моделей класифікації;
- оцінка якості моделей за Accuracy, Precision, Recall, F1-score та ROC-AUC;
- веб-інтерфейс на Streamlit для прогнозування як для одного клієнта, так і для пакетної обробки CSV-файлів;
- контейнеризація проєкту за допомогою Docker.

## Огляд даних

У роботі використано датасет із 72 274 рядками та 11 стовпцями. Цільова змінна — `churn`.

Основні характеристики датасету:

- частка клієнтів, які відмовилися від послуг: приблизно 55.4% (помірний дисбаланс класів);
- пропуски є у `remaining_contract` (21 572; 29.85%), `download_avg` (381; 0.53%) та `upload_avg` (381; 0.53%);
- `download_over_limit` є числовим лічильником зі значеннями від 0 до 7, а не бінарною ознакою.

## EDA та обґрунтування підходу

На етапі EDA було перевірено:

- розподіл цільової змінної;
- наявність пропусків у даних;
- базові статистичні характеристики числових ознак;
- структуру та типи ознак, необхідних для подальшої обробки.

Попередня обробка створює ознаку `remaining_contract_missing` і заповнює пропуски. Її параметри навчаються лише на тренувальних частинах CV-фолдів; holdout не використовується для підбору. Пропуски `remaining_contract` пов'язані з цільовою змінною: 49.24% записів із відтоком і 5.75% записів без відтоку мають відсутнє значення. Це потребує перевірки джерела даних, але не доводить причинного зв'язку.

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

Модель обирається за середнім ROC-AUC повторної CV: обрано LightGBM (0.9818 ± 0.0009). На holdout із 14 455 записів Gradient Boosting має найвищі Accuracy та F1 і найнижчий Brier score; LightGBM має найвищий Recall, XGBoost — найвищий Precision. Універсального переможця за всіма метриками немає.

| Модель | CV ROC-AUC (середнє ± std) | Accuracy | Precision | Recall | F1 | Holdout ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | **0.9818 ± 0.0009** | 0.9428 | 0.9555 | **0.9406** | 0.9480 | **0.9829** | 0.0448 |
| Random Forest | 0.9810 ± 0.0004 | 0.9409 | 0.9566 | 0.9358 | 0.9461 | 0.9819 | 0.0461 |
| Gradient Boosting | 0.9808 ± 0.0009 | **0.9438** | 0.9574 | 0.9404 | **0.9489** | 0.9828 | **0.0446** |
| XGBoost | 0.9804 ± 0.0008 | 0.9412 | **0.9576** | 0.9353 | 0.9463 | 0.9811 | 0.0469 |
| Decision Tree | 0.9708 ± 0.0011 | 0.9376 | 0.9504 | 0.9362 | 0.9433 | 0.9739 | 0.0507 |
| KNN | 0.9584 ± 0.0005 | 0.9202 | 0.9369 | 0.9177 | 0.9272 | 0.9606 | 0.0656 |
| Logistic Regression | 0.9334 ± 0.0011 | 0.8758 | 0.8745 | 0.9057 | 0.8899 | 0.9309 | 0.0974 |

Матриця помилок для обраної LightGBM: 6 094 true negatives, 351 false positives, 476 false negatives і 7 534 true positives. Поріг класифікації 0.5 відповідає припущенню рівної вартості хибнонегативних і хибнопозитивних рішень. Деталі й застереження наведено у [звіті](docs/report.md) та [документі EDA](docs/eda_analysis.md).

## Запуск проєкту

### Локальний запуск

1. Встановіть залежності:

```bash
pip install -r requirements.txt
```

2. Навчіть моделі та створіть артефакти:

```bash
python src/model_training.py
```

3. Оновіть оцінку та графіки:

```bash
python src/model_evaluation.py
```

4. Запустіть веб-застосунок:

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
  "remaining_contract": 6,
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
