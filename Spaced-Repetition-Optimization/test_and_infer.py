#!/usr/bin/env python3
"""Test the trained models with 4 hand-written inputs (no dataset needed), or infer one review from --input.

Usage (from the project root, after training_model.py):
    python model/test_and_infer.py                      # runs the 4 built-in test inputs
    python model/test_and_infer.py --input review.json  # one custom review
    python model/test_and_infer.py --example            # prints an example review JSON
    python model/test_and_infer.py --target-recall 0.6 --output result.json
"""
import argparse, json, pickle
from pathlib import Path

import numpy as np
import pandas as pd

from training_model import (
    CLASSIFIER_FEATURES, METADATA_PATH, REGRESSOR_FEATURES,
    REGRESSOR_PATH, CLASSIFIER_PATH
)

EXAMPLE_INPUT = {
    "review_timestamp": "2026-10-09 18:00:00",
    "days_since_last_review": 5,
    "previous_assessment_score_pct": 78,
    "items_attempted": 10,
    "empirical_difficulty": 0.42
}

# Four hand-written reviews that cover different student / topic situations.
TEST_CASES = [
    {"name": "1. Strong student, easy topic",
     "input": {"review_timestamp": "2026-10-09 18:00:00", "days_since_last_review": 3,
               "previous_assessment_score_pct": 92, "items_attempted": 10, "empirical_difficulty": 0.20}},
    {"name": "2. Average student, medium topic",
     "input": {"review_timestamp": "2026-10-10 19:30:00", "days_since_last_review": 5,
               "previous_assessment_score_pct": 70, "items_attempted": 8, "empirical_difficulty": 0.45}},
    {"name": "3. Struggling student, hard topic",
     "input": {"review_timestamp": "2026-10-11 20:15:00", "days_since_last_review": 7,
               "previous_assessment_score_pct": 45, "items_attempted": 12, "empirical_difficulty": 0.75}},
    {"name": "4. Strong student, hard topic",
     "input": {"review_timestamp": "2026-10-12 17:45:00", "days_since_last_review": 10,
               "previous_assessment_score_pct": 85, "items_attempted": 6, "empirical_difficulty": 0.80}},
]


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)          # only load .pkl files that you trained yourself


def load_models():
    for path in (REGRESSOR_PATH, CLASSIFIER_PATH, METADATA_PATH):
        if not path.exists():
            raise FileNotFoundError(f"{path.name} not found. Run training_model.py first.")
    reg, clf = load_pickle(REGRESSOR_PATH), load_pickle(CLASSIFIER_PATH)
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return reg, clf, meta


def make_features(state, gap=None):
    if not state.get("review_timestamp"):
        raise ValueError("review_timestamp is required, e.g. '2026-10-09 18:00:00'.")
    row = {
        "days_since_last_review": state.get("days_since_last_review", 1) if gap is None else gap,
        "previous_assessment_score_pct": state.get("previous_assessment_score_pct"),
        "items_attempted": state.get("items_attempted"),
        "empirical_difficulty": state.get("empirical_difficulty"),
    }
    missing = [k for k, v in row.items() if v is None]
    if missing:
        raise ValueError(f"Missing required feature values: {missing}")
    return pd.DataFrame([row])[CLASSIFIER_FEATURES], pd.DataFrame([row])[REGRESSOR_FEATURES]


def recommend_next_review(state, target_recall=0.5):
    reg, clf, _ = load_models()
    _, reg_x = make_features(state)
    predicted_gap = int(np.clip(round(float(reg.predict(reg_x)[0])), 1, 60))
    ts = pd.Timestamp(state["review_timestamp"])
    # Test possible gaps up to the model's predicted historical interval.
    candidates = list(range(1, predicted_gap + 1))
    rows = []
    for gap in candidates:
        clf_x, _ = make_features(state, gap=gap)
        rows.append(clf_x.iloc[0].to_dict())
    probs = clf.predict_proba(pd.DataFrame(rows)[CLASSIFIER_FEATURES])[:, 1]
    valid = [i for i, p in enumerate(probs) if p >= target_recall]
    idx = valid[-1] if valid else 0
    days, probability = candidates[idx], float(probs[idx])
    due = (ts + pd.Timedelta(days=days)).normalize()
    return {
        "headline": f"Review this topic again in {days} day(s), on {due:%Y-%m-%d}.",
        "recommendation": {
            "review_in_days": days,
            "review_on": due.strftime("%Y-%m-%d"),
            "expected_chance_of_recall": f"{probability:.0%}",
            "recall_level": "High" if probability >= 0.75 else "Moderate" if probability >= 0.5 else "Low"
        },
        "how_this_was_decided": {
            "model_suggestion_days": predicted_gap,
            "target_recall": f"{target_recall:.0%}",
            "note": "The available CSV contains days_since_last_review (a past gap), not a true next-review interval. Treat the interval output as a baseline estimate."
        }
    }


def run_test_cases(target_recall):
    cases, summary = [], []
    for case in TEST_CASES:
        result = recommend_next_review(case["input"], target_recall)
        cases.append({"case": case["name"], "input": case["input"], "result": result})
        rec = result["recommendation"]
        summary.append(f"{case['name']}: review in {rec['review_in_days']} day(s), "
                       f"{rec['expected_chance_of_recall']} chance of recall ({rec['recall_level']})")
    return {"test_inputs_used": len(cases), "summary": summary, "test_cases": cases}


def main():
    parser = argparse.ArgumentParser(description="Test models with 4 built-in inputs or generate one recommendation.")
    parser.add_argument("--input", help="JSON file containing one review and its feature values")
    parser.add_argument("--target-recall", type=float, default=0.5)
    parser.add_argument("--output", help="Optional output JSON path")
    parser.add_argument("--example", action="store_true", help="print an example --input JSON and exit")
    args = parser.parse_args()

    if args.example:
        print(json.dumps(EXAMPLE_INPUT, indent=2))
        return

    if args.input:
        state = json.loads(Path(args.input).read_text(encoding="utf-8"))
        result = recommend_next_review(state, args.target_recall)
    else:
        result = run_test_cases(args.target_recall)

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2))
        raise SystemExit(1)
