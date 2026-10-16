# ============================================================
# Converted from Google Colab / Jupyter Notebook via LLM
# Original file : Predictive-Grant-Funding-Success-Classifier-train.ipynb
# LLM model     : openai/gpt-oss-120b
# Pipeline Step 1: Colab to Local
# ============================================================

"""
Standardized Pipeline Implementation
"""
import argparse
import subprocess
import sys

# ================================================================
# Original notebook cell 2
# ================================================================
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# ================================================================
# Original notebook cell 4 – Colab upload replaced by local discovery
# ================================================================
# (Handled inside load_data)

# ================================================================
# Original notebook cell 5 – reading CSV (moved into load_data)
# ================================================================

# ================================================================
# Original notebook cell 6 – feature / target definitions
# ================================================================
# (Handled inside preprocess_data)

# ================================================================
# Original notebook cell 7 – train‑test split (handled inside preprocess_data)
# ================================================================

# ================================================================
# Original notebook cell 9 – first pipeline creation
# ================================================================
# (Handled inside preprocess_data)

# ================================================================
# Original notebook cell 10 – second pipeline creation (overwrites first)
# ================================================================
# (Handled inside preprocess_data)

# ================================================================
# Original notebook cell 11 – first fit (handled inside execute)
# ================================================================

# ================================================================
# Original notebook cell 12 – second fit & first set of metrics/prints
# ================================================================

# ================================================================
# Original notebook cell 13 – additional metrics & prints
# ================================================================

# ================================================================
# Original notebook cell 14 – classification report & confusion matrix
# ================================================================

# ================================================================
# Original notebook cell 15 – feature importance extraction
# ================================================================

# ================================================================
# Original notebook cell 16 – model persistence
# ================================================================
# (Handled inside generate_output)


# STAGE 0 - ARGUMENTS / CONFIGURATION
def parse_arguments():
    parser = argparse.ArgumentParser(description="Pipeline implementation")
    args, _ = parser.parse_known_args()
    return vars(args)


# STAGE 1 - INPUT
def load_data(config):
    """
    Replaces the Colab upload with a local CSV discovery.
    Returns the raw pandas DataFrame.
    """
    # Discover CSV files in the current working directory
    csv_files = list(Path(".").glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            "No CSV file found in the current directory. "
            "Please place Predictive-Grant-Funding-Success-Classifier-synthetic-dataset.csv here."
        )
    if len(csv_files) > 1:
        raise RuntimeError(
            f"Multiple CSV files found: {[str(p) for p in csv_files]}. "
            "Please ensure only one dataset CSV is present."
        )
    dataset_file = csv_files[0]
    print(f"Dataset file found: {dataset_file}")

    # Read the CSV dataset
    df = pd.read_csv(dataset_file)

    # Display the number of rows and columns in the dataset.
    print(df.shape)

    return df


# STAGE 2 - PREPROCESSING / FEATURE ENGINEERING
def preprocess_data(raw_data, config):
    """
    Creates feature lists, splits the data, builds the preprocessing
    transformer and the two pipeline definitions (as in the original notebook).
    Returns a dictionary with all artifacts needed downstream.
    """
    # ------------------------------------------------------------
    # Original notebook cell 6 – target & feature definitions
    # ------------------------------------------------------------
    TARGET = "funded"

    numeric_features = [
        "proposal_length",
        "requested_budget",
        "strategic_alignment_score",
        "team_size",
        "avg_team_experience_years",
        "prior_grants_won",
        "avg_publication_count",
        "reviewer_focus_match",
    ]

    categorical_features = [
        "methodology_type",
        "agency",
        "applicant_state",
    ]

    # ------------------------------------------------------------
    # Original notebook cell 6 – X / y creation
    # ------------------------------------------------------------
    X = raw_data[numeric_features + categorical_features]
    y = raw_data[TARGET]

    # ------------------------------------------------------------
    # Original notebook cell 7 – train / test split
    # ------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # ------------------------------------------------------------
    # Original notebook cell 9 – preprocessor definition
    # ------------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    # ------------------------------------------------------------
    # Original notebook cell 9 – first pipeline creation
    # ------------------------------------------------------------
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    print("Regularized Random Forest pipeline created successfully.")

    # ------------------------------------------------------------
    # Original notebook cell 10 – second pipeline creation (overwrites first)
    # ------------------------------------------------------------
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    print("Random Forest pipeline created successfully.")

    processed_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "preprocessor": preprocessor,
        "pipeline": pipeline,
    }
    return processed_data


# STAGE 3 - INITIALIZATION / SETUP
def initialize(config, processed_data):
    """
    Places the final pipeline into the mutable state dictionary.
    """
    state = {
        "pipeline": processed_data["pipeline"],
    }
    return state


# STAGE 4 - CORE EXECUTION
def execute(state, processed_data, config):
    """
    Trains the model (twice, as in the notebook), computes all
    metrics, prints the same console output, extracts feature
    importances and assembles a results dictionary.
    """
    pipeline = state["pipeline"]
    X_train = processed_data["X_train"]
    X_test = processed_data["X_test"]
    y_train = processed_data["y_train"]
    y_test = processed_data["y_test"]
    numeric_features = processed_data["numeric_features"]
    categorical_features = processed_data["categorical_features"]

    # ------------------------------------------------------------
    # Original notebook cell 11 – first fit
    # ------------------------------------------------------------
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # ------------------------------------------------------------
    # Original notebook cell 12 – second fit & first metric block
    # ------------------------------------------------------------
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    accuracy_gap = train_accuracy - test_accuracy

    print("=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print(f"Training Accuracy : {train_accuracy:.4f}")
    print(f"Test Accuracy     : {test_accuracy:.4f}")
    print(f"Accuracy Gap      : {accuracy_gap:.4f}")
    print(f"Accuracy Gap (%)  : {accuracy_gap * 100:.2f}%")

    # ------------------------------------------------------------
    # Original notebook cell 13 – ROC‑AUC metrics & second metric block
    # ------------------------------------------------------------
    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]

    train_roc_auc = roc_auc_score(y_train, y_train_proba)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)

    accuracy_gap = train_accuracy - test_accuracy  # recomputed for clarity

    print("=" * 60)
    print("              MODEL PERFORMANCE")
    print("=" * 60)

    print("\nTRAINING DATA")
    print(f"Accuracy : {train_accuracy:.4f}")
    print(f"ROC-AUC  : {train_roc_auc:.4f}")

    print("\nTEST DATA")
    print(f"Accuracy : {test_accuracy:.4f}")
    print(f"ROC-AUC  : {test_roc_auc:.4f}")

    print("\nOVERFITTING CHECK")
    print(f"Accuracy Gap : {accuracy_gap:.4f}")
    print(f"Accuracy Gap : {accuracy_gap * 100:.2f}%")
    print("=" * 60)

    # ------------------------------------------------------------
    # Original notebook cell 14 – classification report & confusion matrix
    # ------------------------------------------------------------
    print("=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print("\nTRAINING DATA")
    print(f"Accuracy : {train_accuracy:.4f}")
    print(f"ROC-AUC  : {train_roc_auc:.4f}")

    print("\nTEST DATA")
    print(f"Accuracy : {test_accuracy:.4f}")
    print(f"ROC-AUC  : {test_roc_auc:.4f}")

    print("\nOVERFITTING CHECK")
    print(f"Accuracy Gap : {accuracy_gap:.4f}")
    print(f"Accuracy Gap : {accuracy_gap * 100:.2f}%")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT - TEST DATA")
    print("=" * 60)
    print(classification_report(y_test, y_test_pred))

    print("=" * 60)
    print("CONFUSION MATRIX - TEST DATA")
    print("=" * 60)
    print(confusion_matrix(y_test, y_test_pred))

    # ------------------------------------------------------------
    # Original notebook cell 15 – feature importance extraction
    # ------------------------------------------------------------
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + cat_feature_names
    importances = pipeline.named_steps["classifier"].feature_importances_

    importance_df = (
        pd.DataFrame({"feature": all_feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print(importance_df)

    # ------------------------------------------------------------
    # Assemble results
    # ------------------------------------------------------------
    results = {
        "pipeline": pipeline,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_roc_auc": train_roc_auc,
        "test_roc_auc": test_roc_auc,
        "accuracy_gap": accuracy_gap,
        "classification_report": classification_report(y_test, y_test_pred),
        "confusion_matrix": confusion_matrix(y_test, y_test_pred),
        "feature_importance": importance_df,
    }
    return results


# STAGE 5 - VALIDATION / EVALUATION
def evaluate(state, results, processed_data, config):
    """
    No additional evaluation beyond what is printed in execute.
    """
    return {"status": "not_applicable"}


# STAGE 6 - OUTPUT / PERSISTENCE
def generate_output(state, results, evaluation, config):
    """
    Saves the trained pipeline to a pickle file and prints confirmation.
    """
    MODEL_PATH = Path("Predictive-Grant-Funding-Success-Classifier-model-pipeline.pkl")
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(results["pipeline"], model_file)

    print("Model pipeline saved successfully in pickle (.pkl) format.")
    print(f"Model path: {MODEL_PATH.resolve()}")

    output = {"model_path": str(MODEL_PATH.resolve())}
    return output


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
