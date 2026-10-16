# ============================================================
# Converted from Google Colab / Jupyter Notebook via LLM
# Original file : strategic-financial-aid-yield-optimizer-train.ipynb
# LLM model     : openai/gpt-oss-120b
# Pipeline Step 1: Colab to Local
# ============================================================

"""
Standardized Pipeline Implementation
"""
import argparse
import subprocess
import sys

# ---------------------------------------------------------------
# Cell 2 – Install XGBoost (converted to uv)
subprocess.run(
    ["uv", "pip", "install", "--python", sys.executable, "-q", "xgboost"],
    check=True,
)
# ---------------------------------------------------------------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier

import pickle
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------
# Cell 4 – Colab file upload (replaced by local discovery)
# from google.colab import files  # removed for local execution
# import io  # not needed after replacement
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Cell 5 – Target column & basic info (moved into preprocess_data)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Cell 6 – Feature lists (moved into preprocess_data)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Cell 7 – Train‑test split (moved into preprocess_data)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Cell 8 – Pipeline definition (moved into preprocess_data)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Cell 9 – Model training (moved into execute)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Cell 10 – Predictions & metric prints (moved into execute)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Cell 11 – Classification report (moved into execute)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Cell 12 – Confusion matrix plot (moved into execute)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Cell 13 – ROC curve plot (moved into execute)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Cell 14 – Predicted probability scatter plot (moved into execute)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Cell 15 – Feature importance plot (moved into execute)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Cell 16 – Model persistence (moved into generate_output)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Cell 17 – Download model (colab specific – commented out)
# files.download(MODEL_PATH)  # removed for local execution
# ---------------------------------------------------------------

# STAGE 0 - ARGUMENTS / CONFIGURATION
def parse_arguments():
    parser = argparse.ArgumentParser(description="Pipeline implementation")
    args, _ = parser.parse_known_args()
    return vars(args)


# STAGE 1 - INPUT
def load_data(config):
    """
    Mimics the original Colab upload step by searching the current working
    directory for a single CSV file.  The function prints the same informational
    messages that the notebook displayed.
    """
    print("Please provide the dataset file (CSV) in the current directory.")
    csv_files = list(Path(".").glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError("No CSV files found in the current directory.")
    if len(csv_files) > 1:
        raise RuntimeError(
            f"Multiple CSV files found: {[p.name for p in csv_files]}. "
            "Please ensure only one dataset CSV is present."
        )

    dataset_path = csv_files[0]
    df = pd.read_csv(dataset_path)

    print("\nDataset loaded successfully.")
    print("File name:", dataset_path.name)
    print("Dataset shape:", df.shape)
    return df


# STAGE 2 - PREPROCESSING / FEATURE ENGINEERING
def preprocess_data(raw_data, config):
    """
    Replicates the notebook's preprocessing, feature list creation,
    train‑test split, and pipeline construction.
    """
    TARGET_COL = "enrolled"
    print("Shape:", raw_data.shape)
    print("Target column:", TARGET_COL)
    print("Target distribution:")
    print(raw_data[TARGET_COL].value_counts())
    print("\nMissing values in sat_act_percentile:", raw_data["sat_act_percentile"].isna().sum())

    # No categorical features in this dataset (already processed)
    CATEGORICAL_FEATURES = []

    NUMERICAL_FEATURES = [
        c for c in raw_data.columns if c != TARGET_COL
    ]

    X = raw_data[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = raw_data[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERICAL_FEATURES),
        ],
        remainder="drop",
    )

    xgb_classifier = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", xgb_classifier),
    ])

    processed_data = {
        "TARGET_COL": TARGET_COL,
        "CATEGORICAL_FEATURES": CATEGORICAL_FEATURES,
        "NUMERICAL_FEATURES": NUMERICAL_FEATURES,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "pipeline": pipeline,
    }
    return processed_data


# STAGE 3 - INITIALIZATION / SETUP
def initialize(config, processed_data):
    """
    No additional state is required beyond what preprocess_data produced.
    """
    state = {}
    return state


# STAGE 4 - CORE EXECUTION
def execute(state, processed_data, config):
    """
    Trains the model, makes predictions, prints metrics, and generates all
    visualisations exactly as the notebook did.
    """
    pipeline = processed_data["pipeline"]
    X_train = processed_data["X_train"]
    y_train = processed_data["y_train"]
    X_test = processed_data["X_test"]
    y_test = processed_data["y_test"]

    # Training
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Metric prints
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")

    # Classification report
    print("Classification Report")
    print("=" * 60)
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Declined (0)", "Enrolled (1)"],
        )
    )

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Declined", "Enrolled"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix — XGBoost Enrollment Classifier")
    plt.show()

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"XGBoost (AUC = {roc_auc_score(y_test, y_proba):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Enrollment Prediction")
    plt.legend()
    plt.show()

    # Predicted probability scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, label="Actual Enrollment", alpha=0.6)
    plt.scatter(range(len(y_proba)), y_proba, label="Predicted Probability", alpha=0.6)
    plt.xlabel("Student Index")
    plt.ylabel("Enrollment / Probability")
    plt.title("Actual Enrollment vs Predicted Probability")
    plt.legend()
    plt.show()

    # Feature importance plot
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["classifier"].feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    plt.figure(figsize=(8, 6))
    plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    plt.xlabel("Importance")
    plt.title("Top 15 Feature Importances — XGBoost")
    plt.tight_layout()
    plt.show()

    results = {
        "pipeline": pipeline,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "importance_df": importance_df,
    }
    return results


# STAGE 5 - VALIDATION / EVALUATION
def evaluate(state, results, processed_data, config):
    """
    No separate evaluation step is required because all metrics were
    displayed during execution.
    """
    return {"status": "not_applicable"}


# STAGE 6 - OUTPUT / PERSISTENCE
def generate_output(state, results, evaluation, config):
    """
    Saves the trained pipeline to disk, mirroring the notebook's final cell.
    """
    MODEL_PATH = "strategic-_model_pipeline.pkl"
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(results["pipeline"], f)

    print(f"Saved trained pipeline to {MODEL_PATH}")
    # In a Colab environment the next line would trigger a download;
    # it is omitted here because the script runs locally.
    return {"model_path": MODEL_PATH}


# STAGE 7 - ORCHESTRATION
def main():
    config = parse_arguments()
    raw_data = load_data(config)
    processed_data = preprocess_data(raw_data, config)
    state = initialize(config, processed_data)
    results = execute(state, processed_data, config)
    evaluation = evaluate(state, results, processed_data, config)
    output = generate_output(state, results, evaluation, config)
    return output


if __name__ == "__main__":
    main()
