# Copilot / AI Agent Instructions for this repository

Overview
- Purpose: This repo holds CMAPSS RUL prediction experiments (data files + `rul_prediction.py`).
- Primary workflow: load text datasets (train/test/RUL), preprocess, build statistical & sequence features, train multiple models, evaluate and save plots.

Key files
- `rul_prediction.py`: single entry-point implementing the full pipeline (data loading, feature engineering, multiple models, evaluation, plotting).
- data files: `train_FD*.txt`, `test_FD*.txt`, `RUL_FD*.txt` located in repository root (`d:/CMAPSSData/` in current environment).

Quick run
- Install deps (recommended virtualenv):
  - `pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow`
- Run the default end-to-end pipeline:
  - `python rul_prediction.py` (defaults to `dataset_name='FD001'`, `window_size=20`, `max_rul=125`).

Important implementation details and conventions (for code edits)
- Data columns: `rul_prediction.py` defines columns: `unit_number, time_cycles, op_setting_1..3, sensor_1..sensor_21`.
  - Note: the provided `readme.txt` describes 26 sensors; the script currently uses 21 sensor columns — verify column count before changing parsing.
- Feature sets:
  - `feature_cols` = `op_setting_1..3` + sensors with non-zero std in the training set.
  - Two parallel representations are used: sequence tensors (`X_*_seq`) for DL models and aggregated statistical features (`X_*_stat`) for traditional ML models.
  - Sliding window: default `window_size=20`. Test handling uses the last window per unit; if a unit has fewer cycles than the window, the last row is repeated to pad to window size.
- Models and mapping to feature types:
  - RandomForest / XGBoost / SVR → use `X_*_stat` (statistical features)
  - LSTM / GRU / CNN-LSTM / Transformer → use `X_*_seq` (sequence features)
- Scaling: `StandardScaler` is fit on training features and applied to test features. Keep this pattern when adding new models or pipelines.
- PHM scoring: the repo implements the official PHM competition scoring function — lower score is better and is used to select the "best" model for plotting/optimization.

Output locations
- Saved images: `d:/CMAPSSData/{dataset_name}_metrics_comparison.png` and `{dataset_name}_{best_model}_predictions.png`.

Editing and experimentation notes for agents
- If you change column parsing, update both `cols` in `load_data()` and anywhere `feature_cols` is inferred.
- When adding models, follow the repo pattern: implement a `build_<name>_model()` method, add to `self.models` and ensure predictions use the correct feature type.
- Long-running training (DL) assumes CPU by default; run on GPU-enabled environment for faster experiments and hyperparameter searches.

Safety checks for PRs
- Confirm dataset parsing matches actual files (count of sensor columns). Small mismatch will silently misalign features.
- Do not change file paths to relative locations unless you update `load_data()` path construction (currently hard-coded to `d:/CMAPSSData/...`).

If something is unclear
- Ask which dataset (`FD001..FD004`) to use and whether to change `window_size` or `max_rul` defaults.

Examples
- To run only preprocessing and inspect feature shapes, edit `if __name__ == "__main__"` at bottom to run `load_data()`, `preprocess_data()`, `feature_engineering()` and then print shapes.
- To add a new ML model, create `build_my_model()` that trains on `self.X_train_stat` and registers it as `self.models['MyModel']`.

---
Please review and tell me if you want these instructions in Chinese, or if you want me to add a short `requirements.txt` and a README run snippet.
