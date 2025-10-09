
# 🏥 UCI Medical Readmission Prediction Project

This project predicts hospital **readmission among diabetic patients** using the **UCI Medical Readmission Dataset** (Diabetes 130-US Hospitals for Years 1999–2008).  
The notebook implements multiple machine learning models, preprocessing pipelines, and explainability methods to understand the factors influencing early readmissions.

---

## 📘 Project Overview

- **Dataset:** UCI Diabetes 130-US Hospitals (1999–2008)  
  Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)  
  File used: `diabetic_data.csv`

- **Objective:** Predict whether a patient will be readmitted within 30 days of discharge.

- **Tech Stack:**
  - Python
  - scikit-learn
  - XGBoost
  - SHAP (for explainability)
  - Pandas, NumPy, Matplotlib, Seaborn

---

## ⚙️ Key Features

1. **Data Loading via API**
   ```python
   import requests
   import pandas as pd
   from io import StringIO

   url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00339/diabetic_data.csv"
   response = requests.get(url)
   df = pd.read_csv(StringIO(response.text))
   ```

2. **Preprocessing Pipeline**
   - Handles missing values using `SimpleImputer` and `IterativeImputer`
   - Encodes categorical variables (`OneHotEncoder`, `OrdinalEncoder`)
   - Scales numeric features (`StandardScaler`)
   - Built with `Pipeline` and `ColumnTransformer`

3. **Models Implemented**
   - Random Forest Classifier
   - XGBoost Classifier
   - Support Vector Machine (SVM)

4. **Hyperparameter Tuning**
   - Added via `GridSearchCV` for Random Forest and XGBoost
   - Optimized for ROC-AUC using 5-fold cross-validation

5. **Model Evaluation**
   - Metrics: ROC-AUC, Brier Score, Calibration Curve
   - Cross-validation for robustness

6. **Explainability**
   - SHAP (SHapley Additive Explanations) used to interpret model predictions
   - Identifies top features influencing patient readmission

---

## 🎯 Target Variable

The column **`readmitted`** has three categories:  
- `"<30"` → Readmitted within 30 days  
- `">30"` → Readmitted after 30 days  
- `"NO"` → Not readmitted  

Binary transformation used for modeling:
```python
df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)
```

---

## 📊 Workflow Summary

1. Fetch dataset via API from UCI Repository  
2. Clean and preprocess data  
3. Split into training and test sets  
4. Train baseline models (RandomForest, XGBoost, SVM)  
5. Perform hyperparameter tuning using GridSearchCV  
6. Evaluate models using ROC-AUC and calibration metrics  
7. Interpret results with SHAP for explainability  

---

## 🧠 Insights

- Certain lab results, diagnosis codes, and medication changes are key indicators for early readmission.
- XGBoost typically achieves the best ROC-AUC after tuning.
- SHAP analysis helps medical practitioners understand *why* a patient might be at high risk of readmission.

---

## 🗓️ Last Updated
October 09, 2025

---

## 👤 Author
Project prepared by an aspiring data scientist proficient in Python, ML, and cloud technologies (AWS).

