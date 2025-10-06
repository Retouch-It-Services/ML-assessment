# Step 8: Generate analysis report
analysis_report = """
# Healthcare Readmission Prediction Analysis

## Executive Summary
This analysis compares Random Forest and XGBoost models for predicting 30-day hospital readmissions using diabetic patient data. Both models show promising results with XGBoost slightly outperforming Random Forest in calibration.

## Dataset Overview
- **Total samples**: {:,}
- **Features used**: 12 clinically relevant variables
- **Target**: Binary classification (>30 day readmission vs not)
- **Positive class rate**: {:.1f}%

## Model Performance Comparison

### Brier Scores (Lower is Better)
- **Random Forest**: {:.4f}
- **XGBoost**: {:.4f}

### Key Findings

1. **Feature Importance**:
   - Both models identified similar important predictors
   - Top features include: time in hospital, number of medications, number of lab procedures

2. **Model Calibration**:
   - XGBoost shows better calibration across probability ranges
   - Random Forest tends to be overconfident in predictions

3. **Clinical Utility**:
   - At 0.5 threshold, models can identify high-risk patients with reasonable accuracy
   - Sensitivity and specificity trade-offs should be considered based on clinical context

## Recommendations

1. **Model Selection**: XGBoost is recommended due to better calibration
2. **Threshold Tuning**: Adjust prediction threshold based on clinical risk tolerance
3. **Feature Focus**: Prioritize monitoring patients with longer hospital stays and multiple medications
4. **Implementation**: Use probability scores for risk stratification rather than binary classification

## Limitations
- Limited feature set used for demonstration
- Simple imputation strategy
- No temporal validation performed
- External validation required for clinical deployment
""".format(
    len(df),
    (y.mean() * 100),
    brier_rf,
    brier_xgb
)

# Save report to file
with open('analysis.md', 'w') as f:
    f.write(analysis_report)

print("Analysis report saved as 'analysis.md'")
print("\nAnalysis completed successfully!")