# Healthcare Readmission Prediction Report

## 1. Introduction

This project addresses the prediction of hospital readmissions using the UCI Medical Readmission Dataset. The goal is to build a reliable, interpretable machine learning model to identify patients at risk of readmission within 30 days, supporting clinical decision-making.

---

## 2. Data Overview

- **Source:** UCI Machine Learning Repository
- **Records:** $(df.shape[0])$ patients, $(df.shape[1])$ features (after preprocessing)
- **Target:** `readmitted_flag` (1 = readmitted <30 days, 0 = otherwise)

---

## 3. Data Preprocessing

- **Missing Values:**  
  - Replaced '?' with NaN.
  - Numerical columns imputed using IterativeImputer.
  - Categorical columns imputed with the most frequent value.
- **Feature Engineering:**  
  - Dropped identifiers (`encounter_id`, `patient_nbr`).
  - Target variable binarized.
- **Encoding:**  
  - Numerical features scaled.
  - Categorical features one-hot encoded.

---

## 4. Feature Selection

- Used domain knowledge and model-based selection (e.g., RFECV, permutation importance).
- Selected features include clinical and utilization variables.

---

## 5. Model Development

- **Models Compared:**  
  - Random Forest  
  - XGBoost  
  - SVM  
- **Pipeline:**  
  - Combined preprocessing and model in a scikit-learn Pipeline.
- **Hyperparameter Tuning:**  
  - RandomizedSearchCV with cross-validation.

---

## 6. Model Calibration

- Used `CalibratedClassifierCV` (isotonic method) to ensure reliable probability estimates.
- Evaluated with Brier score and calibration plots.

---

## 7. Evaluation Metrics

- **ROC-AUC:** Measures discrimination.
- **Brier Score:** Measures calibration.
- **Calibration Plot:** Visualizes predicted vs. actual probabilities.

---

## 8. Results

- **Best Model:** Random Forest (after tuning and calibration)
- **Test ROC-AUC:** Refer to notebook output.
- **Test Brier Score:** Refer to notebook output.
- **Calibration Plot:** See notebook for visualization.

---

## 9. Model Interpretability

- Used SHAP values for feature importance.
- Provided summary plots with feature names.
- Key predictors: [List top features from SHAP summary].

---

## 10. Clinical Utility

- Model can identify high-risk patients for targeted interventions.
- Feature importance aligns with clinical expectations (e.g., prior admissions, length of stay).

---

## 11. Artifacts

- **Trained Model:** [`readmission_model.pkl`](readmission_model.pkl)
- **Preprocessing Pipeline:** [`preprocessor.pkl`](preprocessor.pkl)
- **Notebook:** [`medical.ipynb`](medical.ipynb)

---

## 12. References

- UCI Machine Learning Repository: Diabetes 130-US hospitals for years 1999–2008 Data Set
- scikit-learn, XGBoost, SHAP documentation

---

## 13. Future Work

- Explore temporal models for time-series data.
- Incorporate additional clinical variables.
- Deploy as a clinical decision support tool.
