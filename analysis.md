# Project analysis — ML-assessment

This document summarizes the analysis of the notebook `Healthcare.ipynb`, the dataset `diabetic_data.csv`, and the outputs in `predictions_diabetes_readmission.csv`. It lists what the notebook does, quick findings, potential bugs, recommended fixes, and suggested next steps for reproducible experiments and model improvement.

## Quick summary

- Dataset: `diabetic_data.csv` (patient encounter records, `readmitted` target mapped to `target_30d` where `'<30'` -> 1 else 0).
- Notebook: loads data, builds preprocessing pipeline (IterativeImputer + StandardScaler for numeric, SimpleImputer + OneHotEncoder for categorical), trains RandomForest, XGBoost and LinearSVC, calibrates models, evaluates (Brier, ROC-AUC, calibration plots), saves predictions, and computes SHAP for XGBoost.
- Output: `predictions_diabetes_readmission.csv` saved from the test set with predicted probabilities for RF and XGB.

## What the notebook does (high-level)

1. Read CSV and create binary target `target_30d` from `readmitted`.
2. Identify numeric and categorical columns automatically.
3. Build `preproc` ColumnTransformer:
   - Numeric: `IterativeImputer(BayesianRidge)` -> `StandardScaler`.
   - Categorical: `SimpleImputer` (constant 'Missing') -> `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
4. Train two styles of pipelines earlier and a later "transform-once" workflow that fits `preproc` then trains classifiers on transformed arrays.
5. Models: RandomForest, XGBoost (XGBClassifier), LinearSVC. Models are calibrated using `CalibratedClassifierCV` with a small holdout.
6. Evaluation: Brier score, ROC-AUC, accuracy/precision/recall/F1, ROC and calibration plots.
7. Save test predictions to CSV.
8. Compute SHAP values (subset of test data) and plot summary and beeswarm for XGBoost.

## Quick findings / observations

- The project already implements strong best-practices:
  - Uses IterativeImputer for numeric missing values.
  - Separates preprocessing and model training so transforms are reusable.
  - Calibrates probabilistic outputs.
  - Uses multiple model families.

- Potential data issues to check (not performed in notebook):
  - Class imbalance in `target_30d` (likely skewed) — use stratified splits (the notebook does this) but also consider resampling or class weights.
  - High-cardinality categorical variables — one-hot encoding may blow up feature count and memory.

## Noted bugs / fragile code (actionable)

1. clean_feature_names generation (in SHAP section):
   - Current code uses `.replace('', '_')` which will insert underscores between every character and is incorrect.
   - The list comprehension has a complex chain that may yield unexpected empty names or duplicated names.
   - Recommendation: use a safer normalization that replaces problematic characters and collapses repeated underscores.

   Suggested replacement code (safe, minimal change):

   ```python
   import re
   raw_feature_names = preproc.get_feature_names_out()
   def _clean(name):
       # replace [, ], < with underscore, drop other non-word chars, collapse underscores
       s = re.sub(r"[\[\]<]", "_", name)
       s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
       s = re.sub(r"_+", "_", s).strip("_")
       return s or "feature_"
   clean_feature_names = [_clean(n) for n in raw_feature_names]
   ```

2. SHAP API usage (compatibility with modern shap versions):
   - Newer SHAP versions prefer `explainer(X)` instead of `explainer.shap_values(X)` and return a `shap.Explanation` object.
   - For TreeExplainer with XGBoost, you can do:

   ```python
   explainer = shap.TreeExplainer(xgb_clf)
   shap_exp = explainer(X_shap_df)
   shap.summary_plot(shap_exp, X_shap_df)
   ```

   - If using older SHAP versions, `explainer.shap_values(X)` may still work. If you see deprecation warnings, switch to `explainer(X)`.

3. Using preproc.get_feature_names_out():
   - When `OneHotEncoder` is present, feature names may contain characters like `x0_A`, brackets, or the transformer names. Cleaning is necessary before creating a DataFrame with columns.
   - Also ensure `X_shap_arr` is a 2D numpy array with shape (n_samples, n_features) when creating DataFrame.

4. SHAP and sampling size:
   - Computing SHAP for large test sets is slow. Notebook uses `n_shap = min(1000, X_test_t.shape[0])` which is fine; consider using `shap.sample()` or a stratified sample to preserve class balance.

## Suggested reproducibility & quality gates

- Add an `environment` or `requirements.txt` capturing package versions (pinned versions for sklearn, xgboost, shap, pandas, numpy).
- Add a small smoke test or unit that:
  - Loads a tiny slice of `diabetic_data.csv`, runs preprocessing transform and ensures shapes are consistent.
  - Trains a very small model on reduced data and ensures `predict_proba` exists.

- Run notebooks from a clean environment to catch deprecation warnings and missing dependencies.

## Next experiments and improvements

1. Hyperparameter tuning:
   - Use `RandomizedSearchCV` or `Optuna` with cross-validation for XGBoost and RandomForest.
2. Class imbalance:
   - Evaluate SMOTE/undersampling, or tune class weights and decision thresholds using expected utility / cost function.
3. Feature engineering:
   - Collapse high-cardinality features, create meaningful aggregations (counts, frequency encoding) before one-hot encoding.
4. Model explainability:
   - Use SHAP interaction values for important feature pairs.
   - Add per-sample SHAP reports for problematic predictions.
5. Calibration & deployment:
   - Keep calibration step as part of a final pipeline; save both the preprocessor and calibrated model (joblib/pickle). Document expected input schema.

## Minimal actionable fixes to apply now

- Fix the `clean_feature_names` code in the SHAP section (see suggested snippet above).
- Update SHAP calls to the modern API if your `shap` package prints deprecation warnings.
- Ensure the DataFrame used to compute SHAP has correct column names and matches the transformed array shape.

## Files of interest

- `Healthcare.ipynb` — main notebook (see SHAP section near the end for the name-cleaning bug).
- `diabetic_data.csv` — dataset.
- `predictions_diabetes_readmission.csv` — saved test predictions.
- `README.md` — project README (review and expand with environment and run instructions).

## Example notes for `README.md` (copy to README if desired)

- How to run the notebook (conda/env and pip list), e.g.:

```powershell
# create env, install minimal deps
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Start jupyter lab or run notebook
jupyter lab
```

- Pip `requirements.txt` should include pinned versions for reproducibility, for example:

```
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.2.2
xgboost==1.7.6
shap==0.41.0
matplotlib==3.7.1
seaborn==0.12.2
```

(Adjust exact versions to match your environment.)

## Closing / Next steps

- I created this summary to capture immediate observations and quick fixes. If you want, I can:
  - Apply the safe `clean_feature_names` fix directly into `Healthcare.ipynb` (make a small patch to the SHAP cell).
  - Add a `requirements.txt` and a small smoke test notebook or script.
  - Run the notebook (or selected cells) to verify the changes.

If you'd like me to apply any of the fixes or add the environment file, tell me which one and I'll proceed.
