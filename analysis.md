# Analysis of Healthcare_Prediction.ipynb

## Objective
This notebook builds and evaluates models to predict 30-day hospital readmission (`target_30d`) from a healthcare dataset (`data.csv`). The main goal is to train reliable classifiers, calibrate their probability outputs, and explain model behavior using SHAP.

## Data
- Source: `data.csv` (loaded at start).
- Target: `target_30d` — derived from column `readmitted` where "<30" maps to 1 and other values map to 0.
- ID columns excluded: `encounter_id`, `patient_nbr`.

## Feature engineering & preprocessing
- Features separated into numeric and categorical using pandas dtypes.
- Numeric pipeline:
  - `IterativeImputer` with `BayesianRidge` (max_iter=10) to impute missing numeric values.
  - `StandardScaler` to standardize numeric features.
- Categorical pipeline:
  - `SimpleImputer` filling missing values with the string `"Missing"`.
  - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to encode categorical variables.
- Preprocessing combined via `ColumnTransformer` named `preproc`.
- The notebook shows two approaches:
  1. A `Pipeline`-based approach where preprocessing is part of full model pipelines (used when training and calibrating using `CalibratedClassifierCV` wrapped around the full pipeline).
  2. A separate fit/transform approach where `preproc` is fit once and reused to transform datasets for downstream classifiers. This is used for faster SHAP computations and repeated model training.

## Train / test split
- Initial split: 80/20 stratified split on `target_30d`.
- Additional small holdout (~10-15% of training set) is used for calibration via `CalibratedClassifierCV`.

## Models trained
- RandomForestClassifier (various configs: n_estimators=200 in pipeline, later 100 with max_depth=12 and class_weight='balanced').
- XGBoost (`xgb.XGBClassifier`) with tuned-ish defaults (`n_estimators=100, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method='hist'`).
- LinearSVC (fast linear SVM) used with `CalibratedClassifierCV` to obtain probabilities.

Calibration
- Models are calibrated using `CalibratedClassifierCV` with `method='isotonic'` or `method='sigmoid'` and `cv='prefit'` by fitting on a held-out subset from the training data.

## Evaluation metrics & plots
- Metrics computed: Brier score, ROC-AUC, accuracy, precision, recall, F1, confusion matrix.
- Visuals produced:
  - ROC curves (with AUC displayed)
  - Calibration plots (observed fraction positive vs mean predicted probability)
  - For the pipeline approach, additional calibration and evaluation were printed for both RF and XGB.

## Explainability (SHAP)
- SHAP is used with the trained `xgb_clf` (XGBoost model) to produce feature importance and beeswarm plots.
- Steps performed:
  - Preprocessed test data (`X_test_t`) is subset (up to 1000 rows) for SHAP (`X_shap_arr`).
  - The raw feature names are obtained via `preproc.get_feature_names_out()` and cleaned to remove forbidden characters like `[`, `]`, `<` and to avoid double underscores. A robust fallback name is provided if a feature name becomes empty.
  - A pandas DataFrame `X_shap_df` is constructed with clean feature names and passed to SHAP's `TreeExplainer` for `xgb_clf`.
  - Plots generated: SHAP summary bar plot and SHAP beeswarm.

Notes on SHAP usage
- The notebook constructs feature names carefully because `ColumnTransformer` + `OneHotEncoder` often produces complex names (including brackets or special characters). The cleaning routine replaces problematic characters and ensures valid column labels.
- `shap.TreeExplainer` is used and `explainer_xgb.shap_values(X_shap_df)` is called. Depending on SHAP and XGBoost versions, the recommended API now uses `explainer(X)` or `explainer.shap_values` may return different shapes; the notebook's usage is consistent with older SHAP versions.

## Key findings (implicit)
- Multiple classifiers were trained and calibrated; metrics were computed and visualized for selection.
- XGBoost and Random Forest both produce probability estimates that benefit from calibration.
- SHAP provides feature-level explanations which can guide feature selection or model interpretation.

## Recommendations / Next steps
1. Validate calibration with more bins (e.g., 20) and inspect reliability diagrams across different population segments.
2. Look for data leakage or target leakage: verify temporal splits or ensure that `encounter_id`/`patient_nbr` are not indirectly present in features.
3. For SHAP:
   - Ensure SHAP and XGBoost versions are compatible; prefer `explainer(X).values` for modern SHAP versions.
   - Consider using `shap.Explainer` which handles multiple model types automatically.
4. Add cross-validation experiments (e.g., StratifiedKFold) to estimate generalization variability rather than a single train-test split.
5. Save calibrated models and preprocessing (joblib) and add a small script to load a model and run predictions on new data.
6. Run class-specific calibration if class imbalance changes across subpopulations.

## Files referenced / produced
- `Healthcare_Prediction.ipynb` — main notebook with code.
- `data.csv` — input data loaded.
- `predictions_diabetes_readmission.csv` — (present in repo root) possibly used for storing predictions.

## Minimal reproduction steps
1. Install dependencies: numpy, pandas, scikit-learn, xgboost, shap, matplotlib, seaborn.
2. Open `Healthcare_Prediction.ipynb` and run cells in order. Ensure a machine with enough memory to fit models (RandomForest/XGBoost).

## Conclusion
The notebook provides a practical pipeline for imputation, preprocessing, model training, calibration, evaluation, and SHAP-based explainability. The artifacts and code are arranged to allow quick iterations. The immediate next steps are to verify calibration robustness, add cross-validation, persist models, and make SHAP usage compatible with the installed SHAP/XGBoost versions.
