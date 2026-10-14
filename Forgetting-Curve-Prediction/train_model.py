"""
train_model.py
==============
FILE 2 OF 3  -  Model training for Forgetting Curve Prediction.

What this file does
-------------------
1. Reads the CSV from generate_dataset.py and keeps ONLY the 13 documented input features.
2. Compares two candidate models using GROUP cross-validation (students never appear in
   both the training and the validation part of a fold):
       A) Logistic Regression            -> simple, transparent
       B) Gradient Boosting (calibrated) -> flexible, monotone in time
3. Picks the model using a CAUTIOUS rule (see CAUTIOUS SELECTION RULE below).
4. Refits the chosen model on all training students, checks it on held-out validation
   students, and saves everything as a PICKLE file (forgetting_model.pkl).
   (pickle is used, NOT joblib.)

Run:
    python train_model.py
    python train_model.py --data synthetic_forgetting_dataset.csv --model_out forgetting_model.pkl

CAUTIOUS SELECTION RULE (what "more cautious" means here)
---------------------------------------------------------
- Models are judged on LOG LOSS (a proper score that punishes over-confident
  probabilities), plus Brier score and calibration error, not on accuracy alone.
- The simpler Logistic Regression wins unless the gradient-boosting model is clearly
  better: at least 1% lower log loss AND not noticeably worse calibrated.
- Gradient boosting is forced to be MONOTONE: a longer gap can never raise predicted recall.
- Probabilities of gradient boosting are calibrated (sigmoid, the more conservative method).
"""

# ----------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------
import argparse                                   # command line options
import json                                       # save the human-readable training report
import pickle                                     # save the model as .pkl (NOT joblib)
from datetime import datetime, timezone           # timestamp for the model metadata

import numpy as np                                # numeric arrays
import pandas as pd                               # tables
import sklearn                                    # only for the version number
from sklearn.calibration import CalibratedClassifierCV              # probability calibration
from sklearn.compose import ColumnTransformer                       # different handling per column
from sklearn.ensemble import HistGradientBoostingClassifier         # fast gradient boosting
from sklearn.linear_model import LogisticRegression                 # simple linear classifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold                      # CV that keeps students together
from sklearn.pipeline import Pipeline                               # chain preprocessing + model
from sklearn.preprocessing import OneHotEncoder, StandardScaler    # encoders / scalers

# Shared definitions (feature names, formulas) come from file 1 so everything stays consistent.
from generate_dataset import (CATEGORICAL_FEATURES, CONCEPT_TYPES, FEATURE_COLUMNS,
                              NUMERIC_FEATURES, RETENTION_REF, TARGET_COLUMN)

RANDOM_STATE = 42                                 # fixed seed so results are repeatable
MIN_RELATIVE_GAIN = 0.01                          # boosting must beat logistic by >= 1% log loss
MAX_ECE_WORSENING = 0.01                          # ...and be at most 0.01 worse in calibration error


# ----------------------------------------------------------------------------------
# METRIC HELPERS
# ----------------------------------------------------------------------------------
def expected_calibration_error(y_true, y_prob, n_bins=10):
    """ECE: average gap between predicted probability and actual recall rate, over probability bins."""
    y_true = np.asarray(y_true)                                      # make sure we have numpy arrays
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)                        # bin borders 0.0, 0.1, ... 1.0
    bin_ids = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)  # which bin each row falls in
    ece = 0.0
    for b in range(n_bins):                                          # visit each bin
        mask = bin_ids == b                                          # rows in this bin
        if mask.any():                                               # skip empty bins
            gap = abs(y_true[mask].mean() - y_prob[mask].mean())     # |actual rate - mean prediction|
            ece += mask.mean() * gap                                 # weight by share of rows
    return float(ece)


def score_predictions(y_true, y_prob):
    """Returns all metrics in one dictionary."""
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)                         # avoid log(0)
    return {
        "log_loss": float(log_loss(y_true, y_prob)),                 # lower is better
        "brier": float(brier_score_loss(y_true, y_prob)),            # lower is better
        "auc": float(roc_auc_score(y_true, y_prob)),                 # higher is better
        "ece": expected_calibration_error(y_true, y_prob),           # lower is better
    }


# ----------------------------------------------------------------------------------
# CANDIDATE MODELS
# ----------------------------------------------------------------------------------
def make_logistic():
    """Candidate A: scaled numeric inputs + one-hot concept_type -> L2-regularised logistic regression."""
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),                 # put numbers on the same scale
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
    clf = LogisticRegression(C=0.5, max_iter=2000)                   # C=0.5: fairly strong L2 regularisation
    return Pipeline([("prep", prep), ("clf", clf)])


def make_boosting_base():
    """Candidate B (inner part): one-hot concept_type + monotone gradient boosting."""
    prep = ColumnTransformer(
        [("num", "passthrough", NUMERIC_FEATURES),                   # trees do not need scaling
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False,                             # keep plain column names
    ).set_output(transform="pandas")                                 # so monotonic_cst can use names
    clf = HistGradientBoostingClassifier(
        max_depth=4,                                                 # shallow trees -> less over-fitting
        learning_rate=0.05,                                          # small steps
        max_iter=300,                                                # upper limit of trees
        min_samples_leaf=100,                                        # every leaf needs plenty of rows
        l2_regularization=1.0,                                       # shrink leaf values
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,  # stop when no longer improving
        monotonic_cst={"time_delta": -1, "log_time_delta": -1,       # longer gap  -> recall never rises
                       "retrievability": 1},                         # higher R    -> recall never falls
        random_state=RANDOM_STATE,
    )
    return Pipeline([("prep", prep), ("clf", clf)])


def fit_candidate(name, X, y, groups):
    """Builds and fits one candidate on (X, y). Boosting is wrapped in group-aware sigmoid calibration."""
    if name == "logistic":
        model = make_logistic()
        model.fit(X, y)                                              # plain fit
        return model
    # Boosting: calibrate with 3 inner folds that also keep students together.
    inner_cv = list(GroupKFold(n_splits=3).split(X, y, groups))      # list of (train_idx, test_idx)
    model = CalibratedClassifierCV(make_boosting_base(), method="sigmoid", cv=inner_cv)
    model.fit(X, y)
    return model


def positive_probability(model, X):
    """Returns P(recalled = 1) for every row."""
    return model.predict_proba(X)[:, 1]                              # column 1 = class "1"


# ----------------------------------------------------------------------------------
# MAIN TRAINING FLOW
# ----------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train the forgetting-curve model and save it as .pkl")
    parser.add_argument("--data", default="synthetic_forgetting_dataset.csv", help="CSV from generate_dataset.py")
    parser.add_argument("--model_out", default="forgetting_model.pkl", help="where to save the pickle file")
    parser.add_argument("--report_out", default="training_report.json", help="where to save the metrics report")
    parser.add_argument("--cv_folds", type=int, default=5, help="group cross-validation folds")
    args = parser.parse_args()

    # ---- 1. Load data and separate the splits made by student -----------------------------------
    df = pd.read_csv(args.data)                                      # read the CSV
    train_df = df[df["split"] == "train"].reset_index(drop=True)     # training students
    valid_df = df[df["split"] == "valid"].reset_index(drop=True)     # held-out validation students
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN].values   # ONLY the 13 features
    groups = train_df["student_id"].values                           # group label = student
    print(f"Train rows: {len(train_df):,}  |  Valid rows: {len(valid_df):,}")

    # ---- 2. Group cross-validation of both candidates -------------------------------------------
    candidates = ["logistic", "boosting"]
    cv_scores = {name: [] for name in candidates}                    # list of per-fold metric dicts
    outer = GroupKFold(n_splits=args.cv_folds)                       # students stay together
    for fold, (tr, va) in enumerate(outer.split(X_train, y_train, groups), start=1):
        for name in candidates:
            model = fit_candidate(name, X_train.iloc[tr], y_train[tr], groups[tr])   # fit on fold-train
            prob = positive_probability(model, X_train.iloc[va])                     # predict fold-valid
            cv_scores[name].append(score_predictions(y_train[va], prob))             # store metrics
        print(f"  CV fold {fold}/{args.cv_folds} done")

    cv_mean = {name: {m: float(np.mean([f[m] for f in folds])) for m in folds[0]}
               for name, folds in cv_scores.items()}                 # average metrics over folds
    print("\nGroup-CV mean metrics:")
    for name, metrics in cv_mean.items():
        print(f"  {name:9s}", {k: round(v, 4) for k, v in metrics.items()})

    # ---- 3. CAUTIOUS selection rule -------------------------------------------------------------
    lr, gb = cv_mean["logistic"], cv_mean["boosting"]
    relative_gain = (lr["log_loss"] - gb["log_loss"]) / lr["log_loss"]     # how much better is boosting?
    ece_ok = gb["ece"] <= lr["ece"] + MAX_ECE_WORSENING                    # is boosting still well calibrated?
    chosen = "boosting" if (relative_gain >= MIN_RELATIVE_GAIN and ece_ok) else "logistic"
    print(f"\nBoosting log-loss gain vs logistic: {relative_gain * 100:.2f}% "
          f"(needs >= {MIN_RELATIVE_GAIN * 100:.0f}% and calibration OK: {ece_ok})")
    print(f"Cautious choice -> {chosen}")

    # ---- 4. Refit the chosen model on ALL training students -------------------------------------
    final_model = fit_candidate(chosen, X_train, y_train, groups)

    # ---- 5. Check on held-out validation students -----------------------------------------------
    valid_prob = positive_probability(final_model, valid_df[FEATURE_COLUMNS])
    valid_metrics = score_predictions(valid_df[TARGET_COLUMN].values, valid_prob)
    baseline_r = score_predictions(valid_df[TARGET_COLUMN].values, valid_df["retrievability"].values)
    print("\nValidation metrics (unseen students):", {k: round(v, 4) for k, v in valid_metrics.items()})
    print("Baseline using retrievability only:   ", {k: round(v, 4) for k, v in baseline_r.items()})

    # ---- 6. Package everything needed for inference and save with PICKLE ------------------------
    complexity_table = (train_df.groupby("concept_id")["concept_complexity"].first().to_dict())  # lookup table
    bundle = {
        "model": final_model,                                        # fitted sklearn object
        "model_name": chosen,
        "feature_columns": FEATURE_COLUMNS,                          # exact input order
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "concept_types": CONCEPT_TYPES,
        "retention_reference": RETENTION_REF,
        "concept_complexity_table": complexity_table,                # concept_id -> complexity (train only)
        "cv_metrics": cv_mean,
        "validation_metrics": valid_metrics,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,                      # unpickle with the same version
    }
    with open(args.model_out, "wb") as f:                            # "wb" = write binary
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)     # <-- pickle, not joblib
    print(f"\nModel saved to {args.model_out}  (pickle format)")

    report = {k: v for k, v in bundle.items() if k not in ("model", "concept_complexity_table")}
    report["selection"] = {"relative_log_loss_gain": relative_gain, "ece_ok": ece_ok, "chosen": chosen}
    with open(args.report_out, "w") as f:                            # human-readable summary
        json.dump(report, f, indent=2)
    print(f"Report saved to {args.report_out}")


if __name__ == "__main__":
    main()
