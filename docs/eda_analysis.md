# EDA and rationale for the approach

## 1. Analysis goal

The EDA stage was conducted to understand the data structure, identify potential issues before preprocessing, and justify the choice of feature preparation methods for the model.

## 2. Key observations

### 2.1. Dataset size

The dataset contains 72,274 rows and 11 columns. The target variable is `churn`.

### 2.2. Target variable distribution

The share of customers with churn is approximately 55.4%, which means the class is balanced enough for applying standard classification approaches.

### 2.3. Missing data

The dataset contains missing values in the `remaining_contract` field (21,572), as well as in `download_avg` and `upload_avg` (381 each). This is important because incomplete data can significantly affect model quality.

### 2.4. Basic statistical characteristics

Among the numerical features, the following stand out:

- `remaining_contract` — has a negative correlation with the target variable;
- `download_avg` — also shows a noticeable relationship with the probability of churn;
- `service_failure_count` has a weak but positive correlation with `churn`.

### 2.5. Relationship between categorical features and churn

The sample shows a significant difference between groups:

- customers without a TB subscription have much higher churn: approximately 89.6% versus 47.7% for those who have such a subscription;
- customers without a movie subscription also churn more often: 66.2% versus 33.9%.

This confirms that binary subscription features contain useful information for the model.

## 3. Rationale for preprocessing

Based on the EDA, the following decisions were made:

1. Missing values in numerical features were filled with the median, which is more robust to outliers than the mean.
2. For categorical features, missing values were filled with the mode.
3. Standardization was applied to numerical features to avoid giving an advantage to features with a larger scale.
4. Categorical features were encoded so the model could work with them as numerical data.
5. A stratified train/test split was used because the target variable is not strongly imbalanced.

## 4. Rationale for model selection

After preparing the data, several algorithms were tested. The best results were shown by LightGBM and XGBoost: LightGBM slightly outperformed in Accuracy, Recall, and F1-score, while XGBoost showed the highest ROC-AUC.

The best metrics obtained from evaluation are:

- LightGBM: Accuracy 0.9436, Precision 0.9564, Recall 0.9412, F1-score 0.9487, ROC-AUC 0.9826
- XGBoost: Accuracy 0.9429, Precision 0.9565, Recall 0.9396, F1-score 0.9480, ROC-AUC 0.9829

## 5. Conclusions

EDA helped not only to understand the data, but also to justify the approach to preprocessing it. As a result, high-quality metrics were achieved, particularly for LightGBM and XGBoost, and the project became not only a technical implementation, but also a practically applicable solution for customer churn analysis.
