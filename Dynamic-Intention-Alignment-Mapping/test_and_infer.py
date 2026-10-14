#!/usr/bin/env python3
"""
DIAM test and inference
=======================
1. Loads model/model.pkl.
2. Evaluates it on the students that were held out during training.
3. Shows example predictions (actual vs predicted, with probabilities).
4. Optionally predicts for your own records from a JSON file.

Run (from anywhere):
    python3 model/test_and_infer.py
    python3 model/test_and_infer.py --samples 10
    python3 model/test_and_infer.py --json my_students.json

The JSON file can be one object, a list of objects, or {"records": [...]}.
Each object uses the feature names listed by:  python3 model/test_and_infer.py --features
Missing features are filled automatically (median for numbers, "None" for text).
"""
import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent

ACTIONS = {
    "Aligned": "Monitor: path supports the new intention",
    "At Risk": "Advisor check-in: review course plan against the new intention",
    "Misaligned": "Trigger advising: re-plan the path for the new intention",
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate model.pkl and run predictions.")
    p.add_argument("--model", default=str(BASE_DIR / "model" / "model.pkl"))
    p.add_argument("--data-dir", default=str(BASE_DIR / "data"))
    p.add_argument("--samples", type=int, default=5, help="Number of example predictions to show.")
    p.add_argument("--json", help="Path to a JSON file with your own records to predict.")
    p.add_argument("--features", action="store_true", help="Print the feature names the model expects and exit.")
    return p.parse_args()


def load_bundle(path: Path):
    if not path.exists():
        sys.exit(f"Model not found: {path}\nTrain it first:  python3 model/training_model.py")
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    trained = bundle.get("versions", {}).get("scikit_learn")
    if trained and trained != sklearn.__version__:
        print(f"WARNING: model trained with scikit-learn {trained}, you are running {sklearn.__version__}.\n"
              f"         Install the versions in requirements.txt to avoid problems.\n")
    return bundle


def to_frame(records, bundle):
    """Build a DataFrame with exactly the model's feature columns."""
    cols = bundle["feature_columns"]
    frame = pd.DataFrame(records).reindex(columns=cols)
    for c in bundle["numeric_columns"]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    for c in bundle["categorical_columns"]:
        frame[c] = frame[c].replace("", np.nan)
    return frame


def predict(bundle, frame):
    pipe = bundle["pipeline"]
    proba = pipe.predict_proba(frame)
    labels = pipe.classes_[proba.argmax(axis=1)]
    return labels, proba


def main():
    args = parse_args()
    bundle = load_bundle(Path(args.model))
    classes = bundle["classes"]

    print(f"Model: {bundle['model_name']}   trained {bundle['trained_at']}   "
          f"(refit on all students: {bundle['refit_all']})")

    if args.features:
        print("\nNumeric features:")
        for c in bundle["numeric_columns"]:
            lo, hi = bundle["numeric_ranges"][c]
            print(f"  {c:<34} range seen in training: {lo:g} to {hi:g}")
        print("\nCategorical features:")
        for c in bundle["categorical_columns"]:
            print(f"  {c:<34} {bundle['categorical_values'][c]}")
        return

    # ---------- 1. hold-out evaluation ----------
    table = Path(args.data_dir) / "diam_training_table.csv"
    if table.exists():
        df = pd.read_csv(table)
        test = df[df["student_id"].isin(set(bundle["test_student_ids"]))].reset_index(drop=True)
        X = to_frame(test.to_dict(orient="records"), bundle)
        y = test[bundle["target"]].astype(str)
        pred, proba = predict(bundle, X)

        print(f"\n=== Hold-out evaluation: {len(test):,} rows from {test['student_id'].nunique()} unseen students ===")
        if bundle["refit_all"]:
            print("NOTE: this model was refitted on all students, so these scores are optimistic.")
        print(f"Accuracy          : {accuracy_score(y, pred):.4f}")
        print(f"Balanced accuracy : {balanced_accuracy_score(y, pred):.4f}")
        print(f"Macro F1          : {f1_score(y, pred, average='macro'):.4f}")
        print(f"ROC AUC (OvR)     : {roc_auc_score(y, proba, multi_class='ovr', labels=classes):.4f}")
        print("\n" + classification_report(y, pred, digits=4))
        cm = pd.DataFrame(
            confusion_matrix(y, pred, labels=classes),
            index=[f"actual {c}" for c in classes],
            columns=[f"pred {c}" for c in classes],
        )
        print("Confusion matrix:\n" + cm.to_string())

        # ---------- 2. example predictions ----------
        n = max(0, min(args.samples, len(test)))
        if n:
            print(f"\n=== {n} example predictions ===")
            rng = np.random.default_rng(0)
            for i in rng.choice(len(test), size=n, replace=False):
                probs = ", ".join(f"{c} {p:.2f}" for c, p in zip(classes, proba[i]))
                mark = "OK " if pred[i] == y.iloc[i] else "MISS"
                print(f"[{mark}] {test.loc[i, 'student_id']} {test.loc[i, 'term']}  "
                      f"actual={y.iloc[i]:<10} predicted={pred[i]:<10} ({probs})")
    else:
        print(f"\nTraining table not found at {table}; skipping hold-out evaluation.")

    # ---------- 3. predictions for your own records ----------
    if args.json:
        payload = json.loads(Path(args.json).read_text())
        if isinstance(payload, dict) and "records" in payload:
            payload = payload["records"]
        if isinstance(payload, dict):
            payload = [payload]

        frame = to_frame(payload, bundle)
        labels, proba = predict(bundle, frame)
        print(f"\n=== Predictions for {args.json} ===")
        for i, (label, p) in enumerate(zip(labels, proba)):
            probs = ", ".join(f"{c} {v:.2f}" for c, v in zip(classes, p))
            print(f"record {i}: {label}  ({probs})\n          action: {ACTIONS.get(label, '')}")


if __name__ == "__main__":
    main()
