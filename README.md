# Breast Cancer Prediction

End-to-end machine learning project for classifying breast tumors as **Malignant** or **Benign** from diagnostic measurements. The project compares classical ML models and neural networks, tunes hyperparameters, explains model behavior, saves reproducible artifacts, and ships a Streamlit prediction app.

## Problem Statement

Breast cancer diagnosis can be supported by automated pattern recognition on tumor measurements. The goal is to build a scientifically responsible binary classifier that predicts whether a tumor is malignant or benign while avoiding data leakage and reporting medically relevant metrics such as recall, specificity, false negatives, ROC-AUC, and PR-AUC.

## Dataset

The project uses the Wisconsin Diagnostic Breast Cancer style dataset stored at `data/raw/breast-cancer.csv`.

- Rows: 569 samples
- Target: `diagnosis`, encoded as `M = 1` and `B = 0`
- Features: 30 numeric tumor measurements
- Identifier column: `id`, removed before modeling
- Class distribution: 357 benign and 212 malignant samples

## Project Workflow

1. Load and validate the dataset.
2. Encode the target and remove non-predictive identifiers.
3. Perform exploratory data analysis and save professional figures.
4. Split data with a reproducible stratified train-test split.
5. Train leakage-safe scikit-learn pipelines with outlier clipping, scaling, optional PCA, and classifiers.
6. Tune models with cross-validation and model-specific grids.
7. Evaluate with ROC-AUC, PR-AUC, recall, precision, F1, sensitivity, specificity, confusion matrices, training time, and inference time.
8. Tune the classification threshold with recall awareness.
9. Save the best model with `joblib`.
10. Generate interpretation artifacts and launch a Streamlit app.

## Technologies Used

Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, joblib, Streamlit, pytest. An optional TensorFlow/Keras trainer is included for deeper neural-network experimentation.

## Models Compared

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- K-Nearest Neighbors
- Support Vector Machine
- Neural Network using `MLPClassifier`

## Best Model Results

The best holdout model from the generated run is:

**Neural Network with threshold tuning**

| Metric | Score |
|---|---:|
| Accuracy | 0.982 |
| Precision | 0.976 |
| Recall / Sensitivity | 0.976 |
| F1-score | 0.976 |
| ROC-AUC | 0.998 |
| PR-AUC | 0.996 |
| Specificity | 0.986 |
| False Positives | 1 |
| False Negatives | 1 |
| Inference time | 0.022 ms/sample |

The untuned model comparison table is saved to `reports/metrics/model_metrics.csv`, and the threshold-tuned table is saved to `reports/metrics/model_metrics_with_threshold_tuning.csv`.

## Key Insights

- The strongest models reached high ROC-AUC, showing strong separability between malignant and benign classes.
- Threshold tuning improved the neural network recall from 0.905 at the default 0.50 threshold to 0.976 at a tuned threshold of 0.24.
- Important predictors repeatedly include worst-case perimeter, area, concave points, radius, and texture measurements.
- Error analysis is saved in `reports/metrics/error_analysis.csv` to inspect false positives and false negatives directly.

## Generated Reports

Figures are saved in `reports/figures/`:

- Target distribution
- Correlation heatmap
- Top feature correlations
- PCA explained variance
- ROC curve comparison
- Precision-recall curve comparison
- Calibration curve
- Best model confusion matrix
- Model performance comparison
- Permutation importance
- Tree-based feature importance
- Logistic regression coefficients
- Neural-network training curve

Metrics and interpretation CSV files are saved in `reports/metrics/`.

## How to Run

Create and activate an environment, then install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[app,advanced,dev]"
```

Run the full training pipeline:

```bash
PYTHONPATH=src python3 run_training.py
```

Run tests:

```bash
PYTHONPATH=src:. python3 -m pytest -q
```

## Streamlit App

Start the interactive prediction app:

```bash
PYTHONPATH=src:. streamlit run app/streamlit_app.py
```

The app lets a user enter tumor diagnostic measurements, returns the predicted class, shows malignant/benign probabilities, displays model metrics, and includes an educational-use warning.

## Example Prediction

Using the first sample from the dataset:

- Actual class: Malignant
- Predicted class: Malignant
- Malignant probability: 99.99%
- Tuned decision threshold: 0.24

## Project Structure

```text
.
├── app/
│   ├── predict.py
│   └── streamlit_app.py
├── data/
│   ├── raw/breast-cancer.csv
│   └── processed/breast-cancer-processed.csv
├── models/
│   └── best_model.joblib
├── notebooks/
│   └── BreastCancer_original.ipynb
├── reports/
│   ├── figures/
│   └── metrics/
├── src/
│   └── breast_cancer_prediction/
├── tests/
├── run_training.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## CV Highlights

- Built an end-to-end machine learning pipeline for breast cancer classification.
- Compared classical ML models, gradient boosting, XGBoost, SVM, KNN, and neural networks using robust evaluation metrics.
- Applied stratified splitting, train-only preprocessing, PCA, cross-validation, hyperparameter tuning, threshold tuning, and model persistence.
- Added model explainability with permutation importance, tree-based feature importance, and logistic regression coefficients.
- Deployed an interactive Streamlit prediction app with probability output and safety disclaimer.
- Implemented reproducible training, generated reports, saved artifacts, and basic unit tests.

## Future Improvements

- Add external validation on a separate clinical dataset.
- Add SHAP plots when the optional SHAP dependency is installed.
- Calibrate probabilities with Platt scaling or isotonic regression and compare deployment thresholds.
- Package the Streamlit app with Docker.
- Add CI to run tests and regenerate a lightweight smoke-test report.
