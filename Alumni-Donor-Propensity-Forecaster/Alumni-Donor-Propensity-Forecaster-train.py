# ============================================================
# Converted from Google Colab / Jupyter Notebook via LLM
# Original file : Alumni-Donor-Propensity-Forecaster-train.ipynb
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
# Cell 3 – Install required packages (converted to uv)
subprocess.run(
    ["uv", "pip", "install", "--python", sys.executable,
     "-q", "pandas", "numpy", "scikit-learn", "xgboost", "catboost"],
    check=True,
)
# ---------------------------------------------------------------
# Cell 5 – Install required machine‑learning libraries (converted to uv)
subprocess.run(
    ["uv", "pip", "install", "--python", sys.executable,
     "-q", "scikit-learn", "xgboost", "catboost"],
    check=True,
)

# ---------------------------------------------------------------
# Original imports (preserved, placed after install calls)
import os
import pickle
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ---------------------------------------------------------------
# Cell 8 – Colab file‑upload (removed, replaced by local discovery)
# from google.colab import files  # removed for local execution

# ---------------------------------------------------------------
# Helper: discover a single file with the given extension in the cwd
def _discover_file(extension: str = ".csv"):
    """
    Search the current working directory for files ending with *extension*.
    Returns a dict mimicking ``files.upload()`` output: {filename: filename}.
    Raises FileNotFoundError if none found, RuntimeError if multiple found.
    """
    matches = [f for f in os.listdir() if f.lower().endswith(extension.lower())]
    if not matches:
        raise FileNotFoundError(f"No *{extension}* file found in the current directory.")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple *{extension}* files found: {matches}")
    filename = matches[0]
    return {filename: filename}


# ---------------------------------------------------------------
# STAGE 0 - ARGUMENTS / CONFIGURATION
def parse_arguments():
    parser = argparse.ArgumentParser(description="Pipeline implementation")
    args, _ = parser.parse_known_args()
    return vars(args)


# ---------------------------------------------------------------
# STAGE 1 - INPUT
def load_data(config):
    """
    Loads the synthetic alumni donor dataset from a local CSV file.
    Performs column validation and prints basic dataset information.
    """
    expected_filename = "Alumni-Donor-Propensity-Forecaster-synthetic-dataset.csv"
    print(f"Looking for {expected_filename} in the current directory...")

    uploaded = _discover_file(".csv")
    file_name = next((name for name in uploaded if name.lower().endswith(".csv")), None)
    if file_name is None:
        raise FileNotFoundError("Please ensure the generated .csv dataset is present.")

    data = pd.read_csv(file_name)

    required_columns = {
        "alumni_id", "graduation_year", "age", "events_attended",
        "emails_received", "emails_opened", "newsletter_clicks",
        "volunteer_events", "previous_donations", "total_donation_amount",
        "donation_frequency", "days_since_last_donation", "career_level",
        "industry", "days_since_last_interaction", "donated_next_12_months"
    }
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    print("\nDataset loaded successfully!")
    print("File name:", file_name)
    print("Dataset shape:", data.shape)
    print("Column names:", data.columns.tolist())
    print(data.head())
    return data


# ---------------------------------------------------------------
# STAGE 2 - PREPROCESSING / FEATURE ENGINEERING
REFERENCE_YEAR = 2026


def engineer_features(frame, reference_year=REFERENCE_YEAR):
    frame = frame.copy()
    frame["years_since_graduation"] = reference_year - frame["graduation_year"]

    frame["email_open_rate"] = np.where(
        frame["emails_received"] > 0,
        frame["emails_opened"] / frame["emails_received"],
        0.0,
    ).clip(0, 1)

    frame["average_donation_amount"] = np.where(
        frame["previous_donations"] > 0,
        frame["total_donation_amount"] / frame["previous_donations"],
        0.0,
    )
    frame["donation_recency_score"] = 1 / (1 + frame["days_since_last_donation"])
    frame["event_engagement"] = (frame["events_attended"] / 10).clip(0, 1)
    frame["newsletter_engagement"] = (frame["newsletter_clicks"] / 15).clip(0, 1)
    frame["volunteer_engagement"] = (frame["volunteer_events"] / 5).clip(0, 1)
    frame["interaction_recency"] = (
        1 - frame["days_since_last_interaction"] / 500
    ).clip(0, 1)
    frame["engagement_score"] = (
        0.30 * frame["event_engagement"]
        + 0.30 * frame["email_open_rate"]
        + 0.15 * frame["newsletter_engagement"]
        + 0.15 * frame["volunteer_engagement"]
        + 0.10 * frame["interaction_recency"]
    ) * 100
    return frame


def preprocess_data(raw_data, config):
    """
    Performs feature engineering, splits data, and builds the preprocessing pipeline.
    Returns a dictionary with all artifacts needed by later stages.
    """
    # Feature engineering
    data = engineer_features(raw_data)

    # Separate features and target
    X = data.drop(columns=["alumni_id", "donated_next_12_months"])
    y = data["donated_next_12_months"]

    print("Features shape (X):", X.shape)
    print("Target shape (y):", y.shape)

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    print("Numerical Features:")
    print(numeric_features)
    print("\nCategorical Features:")
    print(categorical_features)

    # Train‑test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    print("✅ Preprocessing pipeline created successfully!")

    processed_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "preprocessor": preprocessor,
    }
    return processed_data


# ---------------------------------------------------------------
# STAGE 3 - INITIALIZATION / SETUP
def initialize(config, processed_data):
    """
    Trains all four classification models and selects the logistic regression
    model for downstream inference.
    Returns a state dictionary containing the trained pipelines and the
    selected model.
    """
    X_train = processed_data["X_train"]
    y_train = processed_data["y_train"]
    preprocessor = processed_data["preprocessor"]

    # Logistic Regression pipeline
    logistic_model = Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    logistic_model.fit(X_train, y_train)
    print("✅ Logistic Regression trained successfully!")

    # Random Forest pipeline
    rf_model = Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
            )),
        ]
    )
    rf_model.fit(X_train, y_train)
    print("✅ Random Forest trained successfully!")

    # XGBoost pipeline
    xgb_model = Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            ("model", XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
                eval_metric="logloss",
            )),
        ]
    )
    xgb_model.fit(X_train, y_train)
    print("✅ XGBoost trained successfully!")

    # CatBoost pipeline
    cat_model = Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            ("model", CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                verbose=False,
                random_seed=42,
            )),
        ]
    )
    cat_model.fit(X_train, y_train)
    print("✅ CatBoost trained successfully!")

    # Select logistic regression for inference (interpretability priority)
    selected_model = logistic_model

    state = {
        "logistic_model": logistic_model,
        "rf_model": rf_model,
        "xgb_model": xgb_model,
        "cat_model": cat_model,
        "selected_model": selected_model,
        "X_test": processed_data["X_test"],
    }
    return state


# ---------------------------------------------------------------
# Helper: model evaluation (preserved from original notebook)
def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate binary classification, including minority‑class and ranking metrics."""
    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        if not (hasattr(model, "named_steps") and isinstance(model.named_steps.get("model"), CatBoostClassifier)):
            raise
        transformed_test = model.named_steps["preprocessor"].transform(X_test)
        estimator = model.named_steps["model"]
        y_pred = np.asarray(estimator.predict(transformed_test)).reshape(-1)
        y_prob = estimator.predict_proba(transformed_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "PR-AUC": average_precision_score(y_test, y_prob),
        "Predicted Positive Rate": float(np.mean(y_pred == 1)),
    }

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)
    for metric, value in metrics.items():
        if metric != "Model":
            print(f"{metric:22s}: {value:.4f}")
    print("\nConfusion Matrix (rows=true labels [0,1], columns=predicted [0,1]):")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=[0, 1], zero_division=0))
    return metrics


# ---------------------------------------------------------------
# STAGE 4 - CORE EXECUTION
def execute(state, processed_data, config):
    """
    Evaluates all trained models, aggregates results, and prepares sample
    predictions for the output stage.
    """
    X_test = processed_data["X_test"]
    y_test = processed_data["y_test"]

    results = []
    results.append(evaluate_model(state["logistic_model"], X_test, y_test, "Logistic Regression"))
    results.append(evaluate_model(state["rf_model"], X_test, y_test, "Random Forest"))
    results.append(evaluate_model(state["xgb_model"], X_test, y_test, "XGBoost"))
    results.append(evaluate_model(state["cat_model"], X_test, y_test, "CatBoost"))

    results_df = pd.DataFrame(results).sort_values(
        by=["Balanced Accuracy", "PR-AUC"], ascending=False
    ).reset_index(drop=True)
    print("\nModel comparison (sorted by balanced accuracy, then PR-AUC):")
    print(results_df)

    # Sample inference using the selected model
    sample_alumni_data = X_test.iloc[:5]
    sample_probabilities = state["selected_model"].predict_proba(sample_alumni_data)[:, 1]

    exec_output = {
        "results_df": results_df,
        "sample_alumni_data": sample_alumni_data,
        "sample_probabilities": sample_probabilities,
    }
    return exec_output


# ---------------------------------------------------------------
# STAGE 5 - VALIDATION / EVALUATION
def evaluate(state, results, processed_data, config):
    """
    No additional evaluation beyond what is performed in ``execute``.
    """
    return {"status": "not_applicable"}


# ---------------------------------------------------------------
# STAGE 6 - OUTPUT / PERSISTENCE
def generate_output(state, results, evaluation, config):
    """
    Saves the selected model pipeline to disk and prints a short inference
    example. All console output mirrors the original notebook.
    """
    model_path = "Alumni-Donor-Propensity-Forecaster-model_pipeline.pkl"
    with open(model_path, "wb") as file:
        pickle.dump(state["selected_model"], file)

    print("Model pipeline saved successfully!")
    print("File saved at:", os.path.abspath(model_path))

    file_exists = os.path.exists(model_path)
    print(f"File exists: {file_exists}")
    if file_exists:
        file_size_bytes = os.path.getsize(model_path)
        file_size_kb = round(file_size_bytes / 1024, 2)
        print(f"File size: {file_size_kb} KB")
    else:
        print("File not found.")

    # ----------------------------------------------------------------
    # Inference example (original Cell 32)
    print("--- Alumni Donation Propensity Prediction Example ---")
    sample_alumni_data = results["sample_alumni_data"]
    sample_probabilities = results["sample_probabilities"]

    print("\nSample donation propensities (estimated likelihood for the next 12 months):")
    for i, probability in enumerate(sample_probabilities):
        propensity_score = round(float(probability) * 100, 2)
        if propensity_score >= 80:
            category, interpretation = "High", "Likely to Donate"
        elif propensity_score >= 50:
            category, interpretation = "Medium", "Moderate Donation Likelihood"
        else:
            category, interpretation = "Low", "Lower Donation Likelihood"
        print(f"Alumni {i + 1}: {propensity_score}% (Category: {category} - {interpretation})")

    print("\nImportant: propensity is an estimated likelihood, not a guarantee.")
    # ----------------------------------------------------------------
    # Cell 38 – download (colab specific) – removed for local execution
    # from google.colab import files  # removed
    # files.download(model_path)      # removed

    return {"model_path": model_path, "evaluation": evaluation}


# ---------------------------------------------------------------
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
