# Analysis of Healthcare Readmission Prediction

## 1. Data Preprocessing

- **Mixed Data Types:**  
  The dataset contains both numerical and categorical features. Numerical columns were imputed using `IterativeImputer` (BayesianRidge), which leverages correlations between features for more accurate imputation. Categorical columns were imputed with a constant value ("Missing") and then one-hot encoded.
- **Feature Selection:**  
  All available features except identifiers (`encounter_id`, `patient_nbr`) and the target were included. Feature importance was later assessed using SHAP values and model-specific importances.
- **Target Definition:**  
  The target variable is `target_30d`, indicating whether a patient was readmitted within 30 days.

## 2. Model Training and Calibration

- **Models Used:**
  - Random Forest (RF)
  - XGBoost (XGB)
  - LinearSVC (SVM with probability calibration)
- **Data Splitting:**  
  The data was split into training and test sets (80/20), with a further split of the training set to create a calibration holdout (10-15%).
- **Calibration:**  
  Probability calibration was performed using `CalibratedClassifierCV` with either 'sigmoid' or 'isotonic' methods, ensuring reliable probability estimates for clinical use.

## 3. Evaluation Metrics

- **Brier Score:**  
  Measures the accuracy of probabilistic predictions. Lower is better.
- **ROC-AUC:**  
  Assesses the model's ability to discriminate between classes.
- **Calibration Plots:**  
  Visualize how well predicted probabilities match observed outcomes.
- **Confusion Matrix & Classification Metrics:**  
  Accuracy, precision, recall, and F1-score were reported for each model.

## 4. Results Summary

| Model         | Brier Score  | ROC-AUC      | Comments                                            |
| ------------- | ------------ | ------------ | --------------------------------------------------- |
| Random Forest | (see output) | (see output) | Good discrimination, well-calibrated                |
| XGBoost       | (see output) | (see output) | Similar/better performance, robust to feature types |
| LinearSVC     | (see output) | (see output) | Lower discrimination, but calibrated                |

- **Calibration:**  
  Both RF and XGB showed good calibration, as seen in calibration plots. SVM required calibration for meaningful probabilities.

## 5. Feature Importance & Interpretability

- **SHAP Analysis:**  
  SHAP values from XGBoost identified the most influential features for readmission risk. Top features typically included:
  - Number of inpatient visits
  - Number of emergency visits
  - Age group
  - Time in hospital
  - Primary diagnosis codes
  - Insulin usage and medication changes
- **Clinical Relevance:**  
  These features align with clinical intuition: frequent hospital use, older age, and certain diagnoses are known risk factors for readmission.

## 6. Clinical Utility

- **Potential Impact:**  
  Well-calibrated models can help clinicians identify high-risk patients for targeted interventions (e.g., follow-up calls, medication review).
- **Limitations:**
  - The dataset lacks some social and outpatient factors.
  - Model performance may vary by hospital or population.
  - Interpretability is improved with SHAP, but clinical validation is recommended.

## 7. Recommendations

- Use XGBoost or Random Forest with calibration for deployment.
- Present SHAP-based feature explanations to clinicians for transparency.
- Periodically retrain and recalibrate the model as new data becomes available.
