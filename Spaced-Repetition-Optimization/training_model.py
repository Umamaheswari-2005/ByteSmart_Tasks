
"""Train Spaced Repetition models using the features produced by dataset_processing.py."""
import argparse, json, pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
DATA_CANDIDATES = [PROJECT_ROOT / "data" / "training.csv",
                   PROJECT_ROOT / "data" / "data" / "training.csv"]
REGRESSOR_PATH = MODEL_DIR / "interval_regressor.pkl"
CLASSIFIER_PATH = MODEL_DIR / "recall_classifier.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# These columns are available in dataset_processing.py's TRAINING_COLUMNS.
LABEL = "recall_success"
TARGET = "days_since_last_review"
NUMERIC_FEATURES = [
    "days_since_last_review",
    "previous_assessment_score_pct",
    "items_attempted",
    "empirical_difficulty",
]
REGRESSOR_FEATURES = [
    "previous_assessment_score_pct", "items_attempted", "empirical_difficulty"
]
CLASSIFIER_FEATURES = NUMERIC_FEATURES


def save_pickle(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def find_data_path(user_path=None):
    if user_path:
        path = Path(user_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        return path
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("training.csv not found. Run dataset_processing.py first or pass --data-path.")


def load_training_data(path):
    df = pd.read_csv(path)
    required = set(REGRESSOR_FEATURES + [TARGET, LABEL])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"training.csv is missing required columns: {missing}")
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[LABEL] = pd.to_numeric(df[LABEL], errors="coerce")
    # First reviews have no previous interval, so they cannot train the interval regressor.
    df = df.dropna(subset=[LABEL]).copy()
    df[LABEL] = df[LABEL].astype(int)
    return df


def main():
    parser = argparse.ArgumentParser(description="Train interval and recall models.")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--refit-all", action="store_true",
                        help="Refit models on all eligible rows after evaluation.")
    args = parser.parse_args()

    path = find_data_path(args.data_path)
    df = load_training_data(path)
    reg_df = df.dropna(subset=[TARGET] + REGRESSOR_FEATURES).copy()
    if len(reg_df) < 10 or len(df) < 10:
        raise ValueError("Not enough valid rows to train; check training.csv.")

    reg_train, reg_test = train_test_split(reg_df, test_size=0.2, random_state=42)
    clf_train, clf_test = train_test_split(df, test_size=0.2, random_state=42)
    if clf_train[LABEL].nunique() < 2:
        raise ValueError("recall_success must contain both 0 and 1 in the training split.")

    reg = CatBoostRegressor(loss_function="MAE", iterations=500, depth=6,
                            learning_rate=0.05, random_seed=42, verbose=False,
                            allow_writing_files=False)
    reg.fit(reg_train[REGRESSOR_FEATURES], reg_train[TARGET])
    clf = CatBoostClassifier(loss_function="Logloss", iterations=500, depth=6,
                             learning_rate=0.05, random_seed=42, verbose=False,
                             allow_writing_files=False)
    clf.fit(clf_train[CLASSIFIER_FEATURES], clf_train[LABEL])

    reg_pred = np.clip(reg.predict(reg_test[REGRESSOR_FEATURES]), 1, 60)
    clf_prob = clf.predict_proba(clf_test[CLASSIFIER_FEATURES])[:, 1]
    clf_pred = (clf_prob >= 0.5).astype(int)
    auc = float(roc_auc_score(clf_test[LABEL], clf_prob)) if clf_test[LABEL].nunique() == 2 else None
    print(f"Rows loaded: {len(df):,}")
    print(f"Interval MAE (historical gap baseline): {mean_absolute_error(reg_test[TARGET], reg_pred):.3f} days")
    print(f"Recall accuracy: {accuracy_score(clf_test[LABEL], clf_pred) * 100:.1f}%")
    print(f"Recall AUC: {auc if auc is not None else 'n/a (test split has one class)'}")

    if args.refit_all:
        reg.fit(reg_df[REGRESSOR_FEATURES], reg_df[TARGET])
        clf.fit(df[CLASSIFIER_FEATURES], df[LABEL])

    save_pickle(reg, REGRESSOR_PATH)
    save_pickle(clf, CLASSIFIER_PATH)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_file": str(path), "data_rows": int(len(df)),
        "features": {"regressor": REGRESSOR_FEATURES, "classifier": CLASSIFIER_FEATURES},
        "target": TARGET, "label": LABEL, "model_format": "pickle",
        "note": "The available dataset contains the previous review gap, not the actual next-review interval. The regressor therefore learns historical gap as a proxy.",
        "holdout_metrics": {"interval_mae_days": round(float(mean_absolute_error(reg_test[TARGET], reg_pred)), 3),
                            "recall_accuracy_pct": round(float(accuracy_score(clf_test[LABEL], clf_pred) * 100), 1),
                            "recall_auc": auc}
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved models and metadata in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
