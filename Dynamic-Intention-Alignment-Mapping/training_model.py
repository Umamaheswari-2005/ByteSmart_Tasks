#!/usr/bin/env python3
"""
DIAM model training
===================
Predicts next_term_alignment_status (Aligned / At Risk / Misaligned): will the
student's current path stay aligned with their (new) intention next term?

What this script does
    1. Makes sure the data exists (runs data/synthetic_dataset.py and
       data/dataset_processing.py automatically if the CSVs are missing).
    2. Splits the data BY STUDENT (a student is never in both train and test).
    3. Trains and compares 3 models with 5-fold grouped cross-validation:
           - Logistic Regression
           - Random Forest
           - Gradient Boosting (HistGradientBoostingClassifier)
    4. Picks the best model by cross-validated macro F1 (fair for the
       minority classes At Risk and Misaligned).
    5. Saves the best model as model/model.pkl and the comparison, report,
       confusion matrix, chart and feature importance in outputs/.

Run (from anywhere):
    python3 model/training_model.py
    python3 model/training_model.py --refit-all     # final model fitted on all students

Note: by default model.pkl is fitted on the TRAIN split only, so that
test_and_infer.py can give an honest hold-out evaluation. Use --refit-all for
a final production model (its hold-out scores are then optimistic).
"""
import argparse
import json
import pickle
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
TARGET = "next_term_alignment_status"

# Identifiers, the two targets, and a text column that is not used.
# (remaining_program_credits already carries the "distance to goal" information.)
DROP_COLUMNS = [
    "student_id",
    "term",
    "next_term_alignment_score",
    "next_term_alignment_status",
    "projected_graduation_term",
]

RAW_FILES = [
    "student_profile.csv",
    "intention_history.csv",
    "course_catalog.csv",
    "program_requirements.csv",
    "course_enrollment.csv",
    "academic_progress.csv",
    "advising_history.csv",
    "hidden_scenarios.csv",
]


def parse_args():
    p = argparse.ArgumentParser(description="Train and compare 3 DIAM models, save the best one.")
    p.add_argument("--data-dir", default=str(BASE_DIR / "data"))
    p.add_argument("--model-dir", default=str(BASE_DIR / "model"))
    p.add_argument("--output-dir", default=str(BASE_DIR / "outputs"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.20, help="Share of STUDENTS held out for testing.")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--refit-all", action="store_true", help="Refit the best model on all students before saving.")
    return p.parse_args()


def run_script(script: Path, *args):
    print(f"  running {script.name} ...")
    subprocess.run([sys.executable, str(script), *map(str, args)], check=True)


def ensure_data(data_dir: Path) -> Path:
    """Generate the raw CSVs and the training table if they are not there yet."""
    table = data_dir / "diam_training_table.csv"
    if table.exists():
        return table

    print("Training table not found. Building the dataset first.")
    if any(not (data_dir / f).exists() for f in RAW_FILES):
        run_script(data_dir / "synthetic_dataset.py", "--output-dir", data_dir)
    run_script(data_dir / "dataset_processing.py", "--data-dir", data_dir)
    return table


def load_data(table: Path):
    df = pd.read_csv(table)
    groups = df["student_id"].to_numpy()
    y = df[TARGET].astype(str)
    X = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns]).copy()

    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    return X, y, groups, numeric, categorical, df


def make_preprocessor(numeric, categorical):
    # Only built-in scikit-learn parts are used, so model.pkl loads anywhere
    # without needing this training script on the import path.
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="None")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )


def build_models(seed):
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=3000, C=1.0, class_weight="balanced", random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=300,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            class_weight="balanced",
            random_state=seed,
        ),
    }


def make_pipeline(model, numeric, categorical):
    return Pipeline([("prep", make_preprocessor(numeric, categorical)), ("model", model)])


def evaluate(pipe, X, y):
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)
    classes = list(pipe.classes_)
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "weighted_f1": f1_score(y, pred, average="weighted"),
        "roc_auc_ovr": roc_auc_score(y, proba, multi_class="ovr", labels=classes),
    }, pred


def to_python(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python(v) for v in value]
    return value


def save_charts(comparison, cm, classes, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping charts.")
        return

    # 1. model comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(comparison))
    w = 0.38
    ax.bar(x - w / 2, comparison["cv_macro_f1_mean"], w, yerr=comparison["cv_macro_f1_std"], label="CV macro F1", capsize=3)
    ax.bar(x + w / 2, comparison["test_macro_f1"], w, label="Test macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["model"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1")
    ax.set_title("DIAM model comparison")
    ax.legend()
    for i, v in enumerate(comparison["test_macro_f1"]):
        ax.text(i + w / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=150)
    plt.close(fig)

    # 2. confusion matrix of the best model
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Best model: confusion matrix (test set)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "best_model_confusion_matrix.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    data_dir, model_dir, out_dir = Path(args.data_dir), Path(args.model_dir), Path(args.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = ensure_data(data_dir)
    X, y, groups, numeric, categorical, df = load_data(table)
    classes = sorted(y.unique())

    print(f"\nData: {len(X):,} rows, {len(np.unique(groups)):,} students, "
          f"{len(numeric)} numeric + {len(categorical)} categorical features")
    print("Target distribution (%):", (y.value_counts(normalize=True) * 100).round(1).to_dict())

    # ---- split by student ----
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    g_train = groups[train_idx]
    print(f"Train: {len(X_train):,} rows / {len(np.unique(g_train)):,} students   "
          f"Test: {len(X_test):,} rows / {len(np.unique(groups[test_idx])):,} students (no student overlap)")

    # ---- train and compare 3 models ----
    cv = GroupKFold(n_splits=args.cv_folds)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
        "roc_auc_ovr": "roc_auc_ovr",
    }

    rows, fitted = [], {}
    print(f"\nComparing 3 models ({args.cv_folds}-fold grouped cross-validation on the train split)")
    for name, model in build_models(args.seed).items():
        pipe = make_pipeline(model, numeric, categorical)

        t0 = time.time()
        cvres = cross_validate(pipe, X_train, y_train, groups=g_train, cv=cv, scoring=scoring, n_jobs=1)
        pipe.fit(X_train, y_train)
        elapsed = time.time() - t0

        test_metrics, _ = evaluate(pipe, X_test, y_test)
        fitted[name] = pipe
        rows.append(
            {
                "model": name,
                "cv_accuracy_mean": cvres["test_accuracy"].mean(),
                "cv_macro_f1_mean": cvres["test_macro_f1"].mean(),
                "cv_macro_f1_std": cvres["test_macro_f1"].std(),
                "cv_balanced_accuracy_mean": cvres["test_balanced_accuracy"].mean(),
                "cv_roc_auc_ovr_mean": cvres["test_roc_auc_ovr"].mean(),
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_weighted_f1": test_metrics["weighted_f1"],
                "test_roc_auc_ovr": test_metrics["roc_auc_ovr"],
                "train_seconds": round(elapsed, 1),
            }
        )
        print(f"  {name:<20} CV macro F1 {rows[-1]['cv_macro_f1_mean']:.4f} (+/- {rows[-1]['cv_macro_f1_std']:.4f})   "
              f"test macro F1 {rows[-1]['test_macro_f1']:.4f}   test acc {rows[-1]['test_accuracy']:.4f}   [{elapsed:.1f}s]")

    comparison = pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False).reset_index(drop=True)
    comparison.insert(0, "rank", comparison.index + 1)
    best_name = comparison.loc[0, "model"]
    best_pipe = fitted[best_name]

    print("\n" + comparison.round(4).to_string(index=False))
    print(f"\nBest model (highest cross-validated macro F1): {best_name}")

    # ---- detailed evaluation of the best model ----
    best_metrics, pred = evaluate(best_pipe, X_test, y_test)
    report_text = classification_report(y_test, pred, digits=4)
    cm = confusion_matrix(y_test, pred, labels=classes)
    print("\nBest model on the held-out students:\n" + report_text)

    # ---- permutation importance on the test set ----
    imp = permutation_importance(
        best_pipe, X_test, y_test, scoring="f1_macro", n_repeats=5, random_state=args.seed, n_jobs=-1
    )
    importance = (
        pd.DataFrame({"feature": X.columns, "importance_mean": imp.importances_mean, "importance_std": imp.importances_std})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    print("Top 10 features (permutation importance, drop in macro F1):")
    print(importance.head(10).round(4).to_string(index=False))

    # ---- optional refit on all students ----
    final_pipe = best_pipe
    if args.refit_all:
        final_pipe = make_pipeline(build_models(args.seed)[best_name], numeric, categorical).fit(X, y)
        print("\nFinal model refitted on ALL students (--refit-all).")

    # ---- save outputs ----
    comparison.to_csv(out_dir / "model_comparison.csv", index=False)
    importance.to_csv(out_dir / "feature_importance.csv", index=False)
    pd.DataFrame(cm, index=[f"actual_{c}" for c in classes], columns=[f"pred_{c}" for c in classes]).to_csv(
        out_dir / "best_model_confusion_matrix.csv"
    )
    (out_dir / "best_model_classification_report.txt").write_text(
        f"Best model: {best_name}\nHeld-out students: {len(np.unique(groups[test_idx]))}\n\n{report_text}"
    )
    save_charts(comparison, cm, classes, out_dir)

    sample_row = X_test.iloc[0].to_dict()
    sample_row = {k: (None if pd.isna(v) else v) for k, v in sample_row.items()}

    bundle = {
        "model_name": best_name,
        "pipeline": final_pipe,
        "target": TARGET,
        "classes": [str(c) for c in final_pipe.classes_],
        "feature_columns": list(X.columns),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "categorical_values": {c: sorted(X[c].dropna().astype(str).unique().tolist()) for c in categorical},
        "numeric_ranges": {c: [float(X[c].min()), float(X[c].max())] for c in numeric},
        "sample_input": to_python(sample_row),
        "test_metrics": to_python(best_metrics),
        "comparison": to_python(comparison.round(6).to_dict(orient="records")),
        "test_student_ids": sorted(set(groups[test_idx].tolist())),
        "refit_all": bool(args.refit_all),
        "training_rows": int(len(X_train)),
        "random_state": args.seed,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    (out_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "best_model": best_name,
                "test_metrics": to_python(best_metrics),
                "versions": bundle["versions"],
                "trained_at": bundle["trained_at"],
                "refit_all": bundle["refit_all"],
            },
            indent=2,
        )
    )

    print(f"\nSaved best model ({best_name}) -> {model_dir / 'model.pkl'}")
    print(f"Saved comparison, report, charts and importance -> {out_dir}")


if __name__ == "__main__":
    main()
