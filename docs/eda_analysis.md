# Exploratory data analysis and modeling rationale

## Objective and evidence

The EDA notebook, `notebooks/01_eda_and_modeling.ipynb`, examines the source customer dataset and motivates the model-preparation steps. This summary uses the checked-in notebook and the saved evaluation artifacts under `models/`; it does not treat feature associations as causal effects.

## Dataset profile

The source data contains 72,274 rows and 11 columns: an identifier (`id`), nine customer/service predictors, and the binary target `churn`. The preprocessing code drops `id` before model training.

The target is mildly imbalanced rather than exactly balanced: approximately 55.4% of records are churn cases. A stratified 80/20 train/test split is used to preserve the target proportions. The saved evaluation set contains 14,455 rows.

Three fields contain missing values:

| Feature | Missing rows | Missing share |
|---|---:|---:|
| `remaining_contract` | 21,572 | 29.85% |
| `download_avg` | 381 | 0.53% |
| `upload_avg` | 381 | 0.53% |

The missingness in `remaining_contract` is substantial and should not be treated as interchangeable with an ordinary numeric value. The preprocessing code therefore creates `remaining_contract_missing` before imputing the original feature with its median. The other numeric missing values are median-imputed. Missing values in the two binary subscription indicators are mode-imputed. `download_over_limit` is a numeric count in the source data, with observed values from 0 to 7, rather than a binary flag.

## Preparation and modeling approach

The current implementation:

1. Separates `churn` from predictors, and removes `id` inside the preprocessing pipeline.
2. Uses a stratified 80/20 split (`random_state=42`) before learning any imputation or scaling values.
3. Fits missing-value medians/modes, the `remaining_contract_missing` indicator behavior, and `StandardScaler` within each training fold of repeated stratified three-fold cross-validation (two repeats). The binary subscription flags use mode imputation; `download_over_limit` is treated as a numeric count.
4. Tunes seven classifiers with ROC-AUC as the cross-validation search score and selects the final candidate by mean training-fold CV ROC-AUC.
5. Refits the chosen model and preprocessing on the complete training partition, then evaluates the untouched holdout once.

Scaling is important for Logistic Regression and KNN; tree-based models generally do not require it, but a shared pipeline keeps their input handling consistent. The churn decision cutoff is 0.5 under the requested equal false-negative/false-positive cost assumption. Risk bands are consistently Low below 0.3, Medium from 0.3 to below 0.7, and High at or above 0.7. These are configurable policy assumptions, not estimated business-optimal thresholds.

## Model comparison

The regenerated `models/model_results.csv` contains repeated-CV ROC-AUC (mean and standard deviation across six folds) and final holdout metrics. Values below are rounded to four decimal places. Model selection uses mean CV ROC-AUC; the holdout columns are reporting-only.

| Model | CV ROC-AUC (mean ± std) | Accuracy | Precision | Recall | F1 | Holdout ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | **0.9818 ± 0.0009** | 0.9428 | 0.9555 | **0.9406** | 0.9480 | 0.9829 | 0.0448 |
| Random Forest | 0.9810 ± 0.0004 | 0.9409 | 0.9566 | 0.9358 | 0.9461 | 0.9819 | 0.0461 |
| Gradient Boosting | 0.9808 ± 0.0009 | **0.9438** | 0.9574 | 0.9404 | **0.9489** | 0.9828 | **0.0446** |
| XGBoost | 0.9804 ± 0.0008 | 0.9412 | **0.9576** | 0.9353 | 0.9463 | 0.9811 | 0.0469 |
| Decision Tree | 0.9708 ± 0.0011 | 0.9376 | 0.9504 | 0.9362 | 0.9433 | 0.9739 | 0.0507 |
| KNN | 0.9584 ± 0.0005 | 0.9202 | 0.9369 | 0.9177 | 0.9272 | 0.9606 | 0.0656 |
| Logistic Regression | 0.9334 ± 0.0011 | 0.8758 | 0.8745 | 0.9057 | 0.8899 | 0.9309 | 0.0974 |

LightGBM was selected by repeated-CV ROC-AUC, with tuned parameters `num_leaves=50`, `n_estimators=100`, `max_depth=15`, and `learning_rate=0.1`. Gradient Boosting has the highest holdout Accuracy and F1 and the lowest Brier score; LightGBM has the highest Recall, while XGBoost has the highest Precision. Differences among leading models are small, and selection is based only on training CV.

The selected LightGBM model correctly classified 13,628 of 14,455 holdout cases (94.28%):

- True negatives: 6,094
- False positives: 351
- False negatives: 476
- True positives: 7,534

At the fixed 0.5 cutoff, 5.9% of actual churners were missed and 5.4% of actual non-churners were flagged as churn. The cutoff assumes equal false-negative and false-positive costs; business outcomes should be monitored to revisit it.

## Feature-importance interpretation

The selected model's feature-importance plot ranks `bill_avg`, `subscription_age`, `remaining_contract`, `download_avg`, and `upload_avg` highest. `remaining_contract_missing` is much lower in importance than these features. Importance is model-specific; it is not an effect size, direction of relationship, or causal explanation.

The selected model's holdout ROC-AUC is 0.9829. The Brier score is 0.0448. The reliability plot tracks the diagonal overall but shows deviations in some probability bands; ROC-AUC alone does not establish calibration.

The missingness diagnostic shows a strong association with the target: `remaining_contract` is missing for 49.24% of churn records and 5.75% of non-churn records. Missingness also varies by subscription segment: 59.63% for customers without TV versus 23.10% with TV, and 36.62% without the movie package versus 16.39% with it. These descriptive rates point to data-lineage follow-up, but do not establish why the values are missing.

## Artifacts

- [ROC curve](../models/plots/roc_curve.png)
- [Confusion matrix](../models/plots/confusion_matrix.png)
- [Feature importance](../models/plots/feature_importance.png)
- [Probability calibration](../models/plots/calibration_curve.png)
- [Prediction-level analysis](../models/plots/predictions_analysis.csv)
- [Holdout metrics](../models/plots/holdout_metrics.csv)
- [Missing-contract diagnostics by segment](../models/plots/missingness_diagnostics.csv)
- [Full model comparison and parameters](../models/model_results.csv)
- [Detailed evaluation report](report.md)

The analysis CSV and Streamlit predictor use the same 0.3 and 0.7 risk-band cutoffs.

## Conclusions

The dataset has usable signal for churn prediction. Missing contract duration is strongly associated with churn and with service segments, so its origin needs investigation. Repeated CV informs model choice and an untouched holdout supplies final metrics, calibration, and error analysis. The source has no timestamp, so temporal validation was not possible; results should not be treated as a guarantee of future performance.
