# ============================================================
# Converted from Google Colab / Jupyter Notebook via LLM
# Original file : Predictive-Campus-Life-Wellness-Sentinel-train.ipynb
# LLM model     : openai/gpt-oss-120b
# Pipeline Step 1: Colab to Local
# ============================================================

"""
Standardized Pipeline Implementation
"""
import argparse

# ================================================================
# Original notebook cell 2
# ================================================================
import io
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

TARGET_COLUMN = "Risk_Level"
DROP_COLUMNS = ["Student_ID", "Week"]
MODEL_PATH = "Predictive-Campus-Life-Wellness-Sentinel-model-pipeline.pkl"
SEED = 42

# ================================================================
# Original notebook cell 3
# ================================================================
def _discover_csv_file() -> str:
    """
    Search the current working directory for exactly one CSV file.
    Returns the filename if found, otherwise raises an error.
    """
    from pathlib import Path

    csv_files = [p for p in Path(".").iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the current directory.")
    if len(csv_files) > 1:
        raise RuntimeError(
            f"Multiple CSV files found: {[str(p) for p in csv_files]}. "
            "Please ensure only one dataset CSV is present."
        )
    return str(csv_files[0])

# ================================================================
# Boilerplate – Argument parsing
# ================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Pipeline implementation")
    args, _ = parser.parse_known_args()
    return vars(args)


# ================================================================
# Stage 1 – INPUT
# ================================================================
def load_data(config):
    # Libraries imported successfully.
    print("Libraries imported successfully.")

    # Discover and load the dataset CSV.
    dataset_name = _discover_csv_file()
    df = pd.read_csv(dataset_name)

    # Validate required columns.
    required_columns = {"Student_ID", "Week", TARGET_COLUMN}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    if df.empty:
        raise ValueError("Uploaded dataset is empty.")

    print(f"Dataset loaded: {dataset_name}")
    print("Dataset shape:", df.shape)
    print(df.head())
    return df


# ================================================================
# Stage 2 – PREPROCESSING / FEATURE ENGINEERING
# ================================================================
def preprocess_data(raw_data, config):
    df = raw_data

    # Prepare features and target. ID/time columns are excluded from model inputs.
    X = df.drop(columns=[TARGET_COLUMN] + [c for c in DROP_COLUMNS if c in df.columns])
    y = df[TARGET_COLUMN].astype(str)

    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numerical_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    if "Student_ID" in df.columns:
        student_ids = df["Student_ID"].unique()
        train_students, test_students = train_test_split(
            student_ids, test_size=0.20, random_state=SEED
        )
        train_mask = df["Student_ID"].isin(train_students)
        test_mask = df["Student_ID"].isin(test_students)
        X_train, X_test = X.loc[train_mask], X.loc[test_mask]
        y_train, y_test = y.loc[train_mask], y.loc[test_mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=SEED, stratify=y
        )

    print("Feature count:", X.shape[1])
    print("Categorical features:", categorical_features)
    print("Numerical features:", numerical_features)
    print("Training records:", len(X_train), "| Testing records:", len(X_test))

    processed_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
    }
    return processed_data


# ================================================================
# Stage 3 – INITIALIZATION / SETUP
# ================================================================
def initialize(config, processed_data):
    # No additional state required beyond what is passed forward.
    state = {}
    return state


# ================================================================
# Stage 4 – CORE EXECUTION
# ================================================================
def execute(state, processed_data, config):
    pipeline = Pipeline(
        [
            ("preprocessor", processed_data["preprocessor"]),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=SEED,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(processed_data["X_train"], processed_data["y_train"])
    y_pred = pipeline.predict(processed_data["X_test"])

    # Metrics & reporting (preserved exactly as in the notebook)
    print(f"Accuracy : {accuracy_score(processed_data['y_test'], y_pred):.4f}")
    print(
        f"Precision (weighted): {precision_score(processed_data['y_test'], y_pred, average='weighted', zero_division=0):.4f}"
    )
    print(
        f"Recall (weighted)   : {recall_score(processed_data['y_test'], y_pred, average='weighted', zero_division=0):.4f}"
    )
    print(
        f"F1-score (weighted) : {f1_score(processed_data['y_test'], y_pred, average='weighted', zero_division=0):.4f}"
    )
    print("\nClassification report:")
    print(classification_report(processed_data["y_test"], y_pred, zero_division=0))
    print("Confusion matrix (class order shown below):")
    print("Classes:", list(pipeline.named_steps["classifier"].classes_))
    print(
        confusion_matrix(
            processed_data["y_test"], y_pred, labels=pipeline.named_steps["classifier"].classes_
        )
    )

    results = {
        "pipeline": pipeline,
        "y_pred": y_pred,
    }
    return results


# ================================================================
# Stage 5 – VALIDATION / EVALUATION
# ================================================================
def evaluate(state, results, processed_data, config):
    # No separate evaluation stage required; metrics already printed in execute().
    return {"status": "not_applicable"}


# ================================================================
# Stage 6 – OUTPUT / PERSISTENCE
# ================================================================
def generate_output(state, results, evaluation, config):
    # Save the full preprocessing + model pipeline in pickle format.
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(results["pipeline"], model_file, protocol=pickle.HIGHEST_PROTOCOL)

    # Verify that the saved pickle can be loaded.
    with open(MODEL_PATH, "rb") as model_file:
        _ = pickle.load(model_file)  # verification; result not used further

    return {"model_path": MODEL_PATH}


# ================================================================
# Stage 7 – ORCHESTRATION
# ================================================================
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
