"""
test_and_infer.py
=================
FILE 3 OF 3  -  Testing and inference for Forgetting Curve Prediction.

What this file does
-------------------
MODE "evaluate"  : loads forgetting_model.pkl and tests it on the held-out TEST students
                   (students the model never saw), reporting metrics, a calibration table,
                   a monotonicity check and a review-interval check against the hidden truth.
MODE "predict"   : gives, for ONE learner/concept record, the recall probability and a
                   CAUTIOUS recommended next-review time.
MODE "both"      : evaluate, then run a small built-in demo (default).

Run:
    python test_and_infer.py
    python test_and_infer.py --mode evaluate
    python test_and_infer.py --mode predict --record '{"time_delta": 7, "history_correct": 3,
        "history_wrong": 1, "success_streak": 2, "last_rating": 3, "response_duration": 5.0,
        "concept_type": "formula", "concept_complexity": 0.2, "student_avg_accuracy": 0.85}'

SECURITY NOTE: pickle files can run code when loaded. Only load .pkl files you created yourself.
"""

# ----------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------
import argparse                                   # command line options
import json                                       # read the --record JSON text
import math                                       # logarithms for the true interval
import pickle                                     # load the .pkl model (NOT joblib)

import numpy as np                                # numeric arrays
import pandas as pd                               # tables
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# Shared formulas and names from file 1 (same maths as training).
from generate_dataset import RETENTION_REF, add_engineered_features
# Metric helper from file 2.
from train_model import expected_calibration_error

# Columns a caller must supply for a prediction (everything else is derived by formulas).
RAW_REQUIRED = ["time_delta", "history_correct", "history_wrong", "success_streak",
                "last_rating", "response_duration", "concept_type", "student_avg_accuracy"]

MIN_INTERVAL_DAYS = 1.0 / 24.0                    # never schedule sooner than 1 hour
MAX_INTERVAL_DAYS = 180.0                         # search horizon = longest gap seen in training (no extrapolation)
GRID = np.geomspace(MIN_INTERVAL_DAYS, MAX_INTERVAL_DAYS, 120)   # candidate gaps, log-spaced (days)


# ----------------------------------------------------------------------------------
# LOADING AND INPUT PREPARATION
# ----------------------------------------------------------------------------------
def load_bundle(path):
    """Loads the pickle bundle created by train_model.py."""
    with open(path, "rb") as f:                   # "rb" = read binary
        return pickle.load(f)                     # <-- pickle, not joblib


def prepare_inputs(bundle, raw):
    """
    Takes a table of raw columns, fills concept_complexity if only concept_id is given,
    then adds the derived features with the shared formulas.
    """
    df = raw.copy()                                                          # do not touch the caller's data
    if "concept_complexity" not in df.columns or df["concept_complexity"].isna().any():
        table = bundle["concept_complexity_table"]                           # concept_id -> complexity
        fallback = float(np.mean(list(table.values())))                      # unseen concept -> average
        if "concept_id" in df.columns:
            looked_up = df["concept_id"].map(table).fillna(fallback)
        else:
            looked_up = pd.Series(fallback, index=df.index)
        df["concept_complexity"] = df.get("concept_complexity", looked_up).fillna(looked_up)
    missing = [c for c in RAW_REQUIRED if c not in df.columns]               # check required inputs
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    unknown = set(df["concept_type"]) - set(bundle["concept_types"])         # check category names
    if unknown:
        raise ValueError(f"Unknown concept_type {unknown}; allowed: {bundle['concept_types']}")
    return add_engineered_features(df)                                       # scaled, log, stability, R


def predict_probability(bundle, prepared):
    """P(recalled = 1) for each prepared row."""
    return bundle["model"].predict_proba(prepared[bundle["feature_columns"]])[:, 1]


# ----------------------------------------------------------------------------------
# REVIEW INTERVAL (the second prediction target)
# ----------------------------------------------------------------------------------
def probability_curves(bundle, rows):
    """
    For every row, predicts recall probability at every gap in GRID (history held fixed).
    Returns (raw_curves, enforced_curves), both shaped (n_rows, n_grid).
    'enforced' applies a running minimum so recall can never rise with a longer gap (cautious).
    """
    n, g = len(rows), len(GRID)
    repeated = rows.loc[rows.index.repeat(g)].reset_index(drop=True)         # each row repeated g times
    repeated["time_delta"] = np.tile(GRID, n)                                # try every candidate gap
    repeated = add_engineered_features(repeated)                             # recompute time-based features
    raw = predict_probability(bundle, repeated).reshape(n, g)                # one curve per row
    return raw, np.minimum.accumulate(raw, axis=1)


def intervals_from_curves(curves, target):
    """Finds, per row, when the curve first drops below `target` (linear interpolation in log-time)."""
    n = curves.shape[0]
    below = curves < target                                                  # True where recall < target
    has_cross = below.any(axis=1)                                            # does it ever drop below?
    first = np.where(has_cross, below.argmax(axis=1), -1)                    # index of first drop, -1 if never
    result = np.full(n, GRID[-1])                                            # never drops -> horizon (180 days)
    result[first == 0] = GRID[0]                                             # already below at 1 hour
    mid = first > 0                                                          # normal case: crossing inside grid
    idx = first[mid]
    p_lo = curves[mid, idx - 1]                                              # probability just before the drop
    p_hi = curves[mid, idx]                                                  # probability just after the drop
    frac = (p_lo - target) / (p_lo - p_hi)                                   # how far between the two points
    log_t = np.log(GRID[idx - 1]) + frac * (np.log(GRID[idx]) - np.log(GRID[idx - 1]))
    result[mid] = np.exp(log_t)
    return result


def recommend_review_interval(bundle, record, target_retention=0.90, safety_factor=0.85):
    """
    CAUTIOUS recommendation for ONE record (dict).
      model_interval = time when predicted recall falls to target_retention
      safe_interval  = model_interval * safety_factor  (review a bit EARLY rather than late)
    """
    prepared = prepare_inputs(bundle, pd.DataFrame([record]))
    _, curves = probability_curves(bundle, prepared)
    model_interval = float(intervals_from_curves(curves, target_retention)[0])
    safe_interval = max(MIN_INTERVAL_DAYS, model_interval * safety_factor)
    return model_interval, safe_interval


# ----------------------------------------------------------------------------------
# EVALUATION ON HELD-OUT TEST STUDENTS
# ----------------------------------------------------------------------------------
def evaluate(bundle, data_path, target_retention, safety_factor, n_interval_rows):
    df = pd.read_csv(data_path)
    test = df[df["split"] == "test"].reset_index(drop=True)                  # students never used in training
    y = test["recalled"].values
    prob = np.clip(predict_probability(bundle, test), 1e-6, 1 - 1e-6)
    print(f"Test rows: {len(test):,}  (model: {bundle['model_name']})\n")

    # ---- A) Probability quality vs simple baselines
    rows = {
        "model": prob,
        "retrievability only": np.clip(test["retrievability"].values, 1e-6, 1 - 1e-6),
        "constant (train mean)": np.full(len(test), y.mean()),
    }
    print("A) Probability quality (lower log loss / Brier / ECE is better, higher AUC is better)")
    print(f"   {'predictor':24s}{'log_loss':>10s}{'brier':>9s}{'auc':>8s}{'ece':>8s}")
    for name, p in rows.items():
        auc = roc_auc_score(y, p) if np.ptp(p) > 0 else 0.5                  # constant predictor has AUC 0.5
        print(f"   {name:24s}{log_loss(y, p):10.4f}{brier_score_loss(y, p):9.4f}{auc:8.3f}"
              f"{expected_calibration_error(y, p):8.4f}")

    # ---- B) Calibration table: predicted vs actual recall by probability band
    print("\nB) Calibration (predicted vs actual recall)")
    bands = pd.cut(prob, bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], include_lowest=True)
    table = pd.DataFrame({"band": bands, "pred": prob, "actual": y}).groupby("band", observed=True).agg(
        rows=("actual", "size"), mean_predicted=("pred", "mean"), actual_recall=("actual", "mean"))
    print(table.round(3).to_string())

    # ---- C) Sanity check: recall should not rise when the gap gets longer
    sample = test.sample(min(500, len(test)), random_state=1)
    raw, _ = probability_curves(bundle, sample)
    violations = float((np.diff(raw, axis=1) > 1e-9).any(axis=1).mean())
    print(f"\nC) Rows whose raw curve ever RISES with a longer gap: {violations * 100:.1f}% "
          f"(the running-minimum guard removes this at scheduling time)")

    # ---- D) Review interval vs hidden ground truth (possible because the data is synthetic)
    sample = test.sample(min(n_interval_rows, len(test)), random_state=2)
    _, curves = probability_curves(bundle, sample)
    model_int = intervals_from_curves(curves, target_retention)
    safe_int = np.maximum(MIN_INTERVAL_DAYS, model_int * safety_factor)
    true_int = np.clip(sample["gt_true_stability"].values * math.log(target_retention) / math.log(RETENTION_REF),
                       MIN_INTERVAL_DAYS, MAX_INTERVAL_DAYS)                 # same clipping as the search grid
    print(f"\nD) Review interval at target retention {target_retention:.2f} vs hidden truth "
          f"({len(sample)} rows)")
    for label, pred in (("model interval", model_int), (f"safe interval (x{safety_factor})", safe_int)):
        ratio = pred / true_int                                              # <1 means reviewed EARLIER than needed
        print(f"   {label:24s} median ratio to truth: {np.median(ratio):5.2f} | "
              f"early or on time: {np.mean(ratio <= 1.0) * 100:5.1f}% | "
              f"more than 25% late: {np.mean(ratio > 1.25) * 100:5.1f}% | "
              f"within 2x: {np.mean((ratio >= 0.5) & (ratio <= 2.0)) * 100:5.1f}%")
    print("   (A ratio below 1 means the learner is asked to review BEFORE the truth says they are at the threshold:"
          " the cautious direction.)")


# ----------------------------------------------------------------------------------
# DEMO AND COMMAND LINE ENTRY POINT
# ----------------------------------------------------------------------------------
def format_days(days):
    """Readable text: hours below 1 day, days otherwise; flags the search-horizon cap."""
    if days >= MAX_INTERVAL_DAYS * 0.999:
        return f">= {MAX_INTERVAL_DAYS:.0f} days (search horizon, no extrapolation beyond training range)"
    if days < 1.0:
        return f"{days * 24:.0f} hours"
    return f"{days:.1f} days"


def print_prediction(bundle, record, target, safety):
    prepared = prepare_inputs(bundle, pd.DataFrame([record]))
    p = float(predict_probability(bundle, prepared)[0])
    model_int, safe_int = recommend_review_interval(bundle, record, target, safety)
    print(f"  Recall probability after {record['time_delta']} days : {p:.2f}")
    print(f"  Estimated stability (baseline)           : {float(prepared['estimated_stability'].iloc[0]):.1f} days")
    print(f"  Model interval to reach {target:.2f} recall     : {format_days(model_int)}")
    print(f"  CAUTIOUS next review (x{safety})           : {format_days(safe_int)}")


def main():
    parser = argparse.ArgumentParser(description="Test the saved model and run inference.")
    parser.add_argument("--model", default="forgetting_model.pkl", help="pickle file from train_model.py")
    parser.add_argument("--data", default="synthetic_forgetting_dataset.csv", help="CSV from generate_dataset.py")
    parser.add_argument("--mode", choices=["evaluate", "predict", "both"], default="both")
    parser.add_argument("--record", type=str, default=None, help="JSON text with one learner/concept record")
    parser.add_argument("--target_retention", type=float, default=0.90, help="review when recall falls to this")
    parser.add_argument("--safety_factor", type=float, default=0.85, help="multiply interval by this (<1 = earlier)")
    parser.add_argument("--n_interval_rows", type=int, default=3000, help="test rows used for the interval check")
    args = parser.parse_args()

    bundle = load_bundle(args.model)
    print(f"Loaded {args.model}: model={bundle['model_name']}, trained with scikit-learn "
          f"{bundle['sklearn_version']}\n")

    if args.mode in ("evaluate", "both"):
        evaluate(bundle, args.data, args.target_retention, args.safety_factor, args.n_interval_rows)

    if args.mode == "predict" and args.record:                               # user-supplied record
        print("Prediction for your record:")
        print_prediction(bundle, json.loads(args.record), args.target_retention, args.safety_factor)
    elif args.mode in ("predict", "both"):                                   # built-in demo learners
        demos = {
            "New learner, hard concept": dict(time_delta=2, history_correct=0, history_wrong=1,
                success_streak=0, last_rating=1, response_duration=12.0, concept_type="conceptual",
                concept_complexity=0.25, student_avg_accuracy=0.70),
            "Average learner": dict(time_delta=7, history_correct=3, history_wrong=1,
                success_streak=2, last_rating=3, response_duration=5.0, concept_type="formula",
                concept_complexity=0.17, student_avg_accuracy=0.84),
            "Strong learner, well known concept": dict(time_delta=20, history_correct=8, history_wrong=0,
                success_streak=8, last_rating=4, response_duration=2.5, concept_type="vocabulary",
                concept_complexity=0.12, student_avg_accuracy=0.93),
        }
        print("\nDemo predictions:")
        for title, rec in demos.items():
            print(f"\n- {title}")
            print_prediction(bundle, rec, args.target_retention, args.safety_factor)


if __name__ == "__main__":
    main()
