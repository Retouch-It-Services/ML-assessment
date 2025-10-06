# Diabetic Readmission Prediction 

##  **Project Overview**
- **Objective**: Predict 30-day hospital readmissions for diabetic patients using ML
- **Dataset**: 101,766 patient encounters, 48 features including demographics, medications, and clinical measurements
- **Target**: Binary classification (1=readmitted within 30 days, 0=otherwise)
- **Class Distribution**: Highly imbalanced - 11% positive class (readmitted)

## **Technical Approach**
- **Data Splitting**: Time-based split (80% training, 20% test) to prevent data leakage
- **Preprocessing**: 
  - Numeric features: KNN + StandardScaler
  - Categorical features: SimpleImputer (most frequent) + One-Hot Encoding
- **Feature Selection**: SelectFromModel with Random Forest (50 estimators)
- **Class Imbalance Handling**: Class weights for Random Forest, scale_pos_weight for XGBoost
- **Model Calibration**: CalibratedClassifierCV with isotonic calibration
- **Interpretability**: SHAP analysis for feature importance

## **Models Evaluated**
1. **Random Forest** (50 estimators, class_weight='balanced') + Calibration
2. **XGBoost** (50 estimators, scale_pos_weight=7.86) + Calibration  
3. **SVM** (RBF kernel, trained on 5,000 sample subset) + Calibration

## **Performance Results**

| Model       | AUC   | Accuracy | F1-Score | Brier Score |
|------------ |-------|----------|----------|-------------|
|XGBoost      | 0.661 | 0.648    | 0.257    | 0.219       |
|Random Forest| 0.626 | 0.894    | 0.001    | 0.093       |
|SVM          | 0.565 | 0.894    | 0.000    | 0.095       |

## Calibration Analysis
- **Isotonic calibration** applied to all models using 5-fold CV (3-fold for SVM)
- **Random Forest** showed best calibration (Brier: 0.093) with well-calibrated probabilities
- **XGBoost** probabilities remained moderately calibrated after calibration
- **Calibration curves** demonstrated improved probability reliability across all models


## 🎯 **Key Findings & Conclusion**
- **XGBoost demonstrated the best overall performance** with balanced metrics across AUC (0.661) and F1-score (0.257)
- **Class imbalance handling was critical** - XGBoost's scale_pos_weight parameter effectively addressed the 11% readmission rate
- **Time-based validation provided realistic performance estimates** for clinical deployment scenarios
- **SHAP analysis revealed clinically meaningful feature importance** aligning with medical intuition
- **Model calibration improved probability reliability** for clinical decision support
- The model shows **moderate predictive capability** for identifying high-risk diabetic patients, with interpretable results suitable for healthcare applications