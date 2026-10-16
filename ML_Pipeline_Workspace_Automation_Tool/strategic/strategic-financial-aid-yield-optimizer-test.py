# ============================================================
# Converted from Google Colab / Jupyter Notebook via LLM
# Original file : strategic-financial-aid-yield-optimizer-test.ipynb
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
# Install required packages (converted from notebook !pip install)
subprocess.run(
    ["uv", "pip", "install", "--python", sys.executable, "-q", "xgboost"],
    check=True,
)
# ---------------------------------------------------------------

# ============================= Imports ==============================
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import contextlib
import io

# ============================= Constants ============================
TIER_MAPPING = {
    "Community College": 0,
    "Regional Comprehensive": 1,
    "R2 Doctoral": 2,
    "R1 Research": 3,
    "Liberal Arts College": 4,
}

# placeholder for objects that will be populated at runtime
pipeline = None          # type: ignore
FEATURE_COLUMNS = []    # type: ignore

# ============================= Helpers ==============================
def preprocess_applicant_for_model(raw_df):
    """
    Applies the same preprocessing used to create the model-ready training data.

    The pretrained pipeline was trained on already-processed columns, so raw
    categorical values cannot be passed directly to it.
    """
    model_df = raw_df.copy()

    # Same institutional-tier mapping used before model training.
    if "institutional_tier" in model_df.columns:
        model_df["institutional_tier"] = model_df["institutional_tier"].map(TIER_MAPPING)

    # Recreate the SAME urban_centric_locale one-hot columns used during training.
    # We use the trained feature names so a single test row does not accidentally
    # change which category is dropped by pd.get_dummies(drop_first=True).
    if "urban_centric_locale" in model_df.columns:
        locale = model_df["urban_centric_locale"].copy()
        model_df = model_df.drop(columns=["urban_centric_locale"])

        locale_columns = [
            c for c in FEATURE_COLUMNS
            if c.startswith("urban_centric_locale_")
        ]

        for col in locale_columns:
            category = col.replace("urban_centric_locale_", "", 1)
            model_df[col] = (locale == category).astype(int)

    # Same transformation used in build_model_ready_dataset().
    if "net_price_to_income_ratio" in model_df.columns:
        model_df["net_price_to_income_ratio"] = np.log1p(
            model_df["net_price_to_income_ratio"]
        )

    # Align exactly with the columns used when the model was trained.
    # Missing columns are filled with 0; extra raw columns are removed.
    model_df = model_df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    return model_df


def make_applicant(**overrides):
    """
    Builds one raw applicant record from a typical-applicant baseline,
    applies the same preprocessing as the training data, and returns the
    model-ready single-row DataFrame.
    """
    baseline = {
        "urban_centric_locale": "Suburb",
        "institutional_tier": "Regional Comprehensive",
        "is_in_state": 1,
        "student_aid_index": 8000.0,
        "adjusted_gross_income": 70000.0,
        "first_gen": 0,
        "pell_eligible": 0,
        "hs_gpa": 3.3,
        "sat_act_percentile": 60.0,
        "cost_of_attendance": 31000.0,
        "miles_from_campus": 60.0,
        "fafsa_month_sin": 0.5,
        "fafsa_month_cos": 0.866,
        "demonstrated_interest": 55.0,
        "merit_scholarship_amt": 4000.0,
        "need_grant_amt": 3000.0,
        "net_price": 24000.0,
        "net_price_to_income_ratio": 0.34,
        "financial_aid_discount_rate": 0.23,
        "unmet_financial_need_gap": 2000.0,
        "engagement_velocity": 0.10,
    }

    baseline.update(overrides)

    raw_df = pd.DataFrame([baseline])
    return preprocess_applicant_for_model(raw_df)


def predict_one(applicant_df):
    """Returns (predicted_label, enrollment_probability) for one model-ready row."""
    label = pipeline.predict(applicant_df)[0]
    proba = pipeline.predict_proba(applicant_df)[0, 1]
    return int(label), float(proba)


# Test result collector (global, but populated inside `execute`)
test_results = []


def run_test(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    test_results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


# ============================= Boilerplate ==========================
# STAGE 0 - ARGUMENTS / CONFIGURATION
def parse_arguments():
    parser = argparse.ArgumentParser(description="Pipeline implementation")
    args, _ = parser.parse_known_args()
    return vars(args)


# STAGE 1 - INPUT
def load_data(config):
    """
    Discover the pretrained model file (expected .pkl), load it, and expose
    pipeline and feature information as module‑level globals.
    """
    # Discover model file
    pkl_files = list(Path.cwd().glob("*.pkl"))
    if len(pkl_files) == 0:
        raise FileNotFoundError("No .pkl model file found in the current directory.")
    if len(pkl_files) > 1:
        raise RuntimeError(
            f"Multiple .pkl files found: {[p.name for p in pkl_files]}. "
            "Unable to infer which one to load."
        )
    model_path = pkl_files[0]

    # Load the pipeline
    global pipeline, FEATURE_COLUMNS
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    # Populate feature list
    FEATURE_COLUMNS = list(pipeline.feature_names_in_)

    # Replicate notebook side‑effects
    pd.set_option("display.max_columns", None)
    print("Model loaded.")
    print("Pipeline steps:", [name for name, _ in pipeline.steps])
    print("Expected input columns:", list(pipeline.feature_names_in_))

    # No raw data to return in this context
    return None


# STAGE 2 - PREPROCESSING / FEATURE ENGINEERING
def preprocess_data(raw_data, config):
    # No preprocessing of a dataset is required for this script.
    return {}


# STAGE 3 - INITIALIZATION / SETUP
def initialize(config, processed_data):
    """
    Populate a state dictionary with objects that later stages may need.
    """
    state = {
        "pipeline": pipeline,
        "FEATURE_COLUMNS": FEATURE_COLUMNS,
    }
    return state


# STAGE 4 - CORE EXECUTION
def execute(state, processed_data, config):
    """
    Run the suite of inference tests and collect results.
    """
    # Ensure the global collector is empty at start
    global test_results
    test_results = []

    # -----------------------------------------------------------
    # Cell 8 – Test 1
    case_1 = make_applicant(
        student_aid_index=-800.0,
        adjusted_gross_income=28000.0,
        first_gen=1,
        pell_eligible=1,
        net_price=6000.0,
        net_price_to_income_ratio=0.21,
        financial_aid_discount_rate=0.80,
        unmet_financial_need_gap=0.0,
        demonstrated_interest=75.0,
    )
    label_1, proba_1 = predict_one(case_1)
    print(f"Predicted label: {label_1}, Enrollment probability: {proba_1:.3f}")

    run_test(
        "Test 1: High-need, well-funded applicant predicted enrollment-likely",
        proba_1 > 0.5,
        detail=f"probability={proba_1:.3f}",
    )

    # -----------------------------------------------------------
    # Cell 9 – Test 2
    case_2 = make_applicant(
        is_in_state=0,
        student_aid_index=45000.0,
        adjusted_gross_income=210000.0,
        net_price=52000.0,
        net_price_to_income_ratio=1.9,
        financial_aid_discount_rate=0.05,
        miles_from_campus=1400.0,
        demonstrated_interest=20.0,
        unmet_financial_need_gap=0.0,
    )
    label_2, proba_2 = predict_one(case_2)
    print(f"Predicted label: {label_2}, Enrollment probability: {proba_2:.3f}")

    run_test(
        "Test 2: High-price, low-engagement, distant applicant predicted enrollment-unlikely",
        proba_2 < 0.5,
        detail=f"probability={proba_2:.3f}",
    )

    # -----------------------------------------------------------
    # Cell 10 – Test 3 (price‑sensitivity curve)
    aid_levels = [0, 5000, 10000, 15000, 20000, 25000]
    coa = 40000.0
    probabilities = []

    for aid in aid_levels:
        net_price = coa - aid
        applicant = make_applicant(
            cost_of_attendance=coa,
            merit_scholarship_amt=aid * 0.6,
            need_grant_amt=aid * 0.4,
            net_price=net_price,
            net_price_to_income_ratio=net_price / 70000.0,
            financial_aid_discount_rate=aid / coa,
            unmet_financial_need_gap=max(0.0, coa - 8000.0 - aid),
        )
        _, proba = predict_one(applicant)
        probabilities.append(proba)
        print(f"Aid=${aid:>6,.0f}  ->  Net price=${net_price:>7,.0f}  ->  P(enroll)={proba:.3f}")

    is_monotonic_nondecreasing = all(
        probabilities[i] <= probabilities[i + 1] + 0.03
        for i in range(len(probabilities) - 1)
    )

    run_test(
        "Test 3: Enrollment probability is non-decreasing as aid increases (price-sensitivity curve)",
        is_monotonic_nondecreasing,
        detail=str([round(p, 3) for p in probabilities]),
    )

    # -----------------------------------------------------------
    # Cell 11 – Test 4 (handling NaN)
    case_4 = make_applicant(sat_act_percentile=np.nan)

    try:
        label_4, proba_4 = predict_one(case_4)
        ran_without_error = True
    except Exception as e:
        ran_without_error = False
        print("Inference raised an error:", e)

    if ran_without_error:
        print(f"Predicted label: {label_4}, Enrollment probability: {proba_4:.3f}")

    run_test(
        "Test 4: Inference succeeds with a missing (NaN) SAT/ACT percentile",
        ran_without_error,
    )

    # -----------------------------------------------------------
    # Cell 12 – Test 5 (batch inference)
    batch = pd.concat([case_1, case_2, make_applicant()], ignore_index=True)
    batch_preds = pipeline.predict(batch)
    batch_probas = pipeline.predict_proba(batch)[:, 1]

    print("Batch predictions:", batch_preds)
    print("Batch probabilities:", np.round(batch_probas, 3))

    run_test(
        "Test 5: Batch inference returns one prediction per input row",
        len(batch_preds) == len(batch) and len(batch_probas) == len(batch),
    )

    # -----------------------------------------------------------
    results = {
        "test_results": test_results,
        "batch_preds": batch_preds,
        "batch_probas": batch_probas,
    }
    return results


# STAGE 5 - VALIDATION / EVALUATION
def evaluate(state, results, processed_data, config):
    """
    Build a summary DataFrame from the collected test results.
    """
    summary_df = pd.DataFrame(results["test_results"], columns=["Test", "Result", "Detail"])
    n_failed = (summary_df["Result"] == "FAIL").sum()
    evaluation = {
        "summary_df": summary_df,
        "n_failed": n_failed,
    }
    return evaluation


# STAGE 6 - OUTPUT / PERSISTENCE
def generate_output(state, results, evaluation, config):
    """
    Print the test summary and run a final example prediction.
    """
    # Original Cell 13 output
    summary_df = evaluation["summary_df"]
    print(summary_df.to_string(index=False))

    n_failed = evaluation["n_failed"]
    if n_failed == 0:
        print("\nAll tests passed.")
    else:
        print(f"\n{n_failed} test(s) failed — review before using this model downstream.")

    # Original Cell 14 – final example prediction
    new_applicant = {
        "urban_centric_locale": "City",
        "institutional_tier": "Liberal Arts College",
        "is_in_state": 0,
        "student_aid_index": 3200.0,
        "adjusted_gross_income": 61000.0,
        "first_gen": 1,
        "pell_eligible": 1,
        "hs_gpa": 3.85,
        "sat_act_percentile": 88.0,
        "cost_of_attendance": 63000.0,
        "miles_from_campus": 320.0,
        "fafsa_month_sin": 0.966,
        "fafsa_month_cos": 0.259,
        "demonstrated_interest": 68.0,
        "merit_scholarship_amt": 22000.0,
        "need_grant_amt": 18000.0,
        "net_price": 23000.0,
        "net_price_to_income_ratio": 0.377,
        "financial_aid_discount_rate": 0.635,
        "unmet_financial_need_gap": 0.0,
        "engagement_velocity": 0.18,
    }

    new_applicant = preprocess_applicant_for_model(
        pd.DataFrame([new_applicant])
    )

    predicted_label, enrollment_probability = predict_one(new_applicant)

    print(f"Predicted outcome : {'Enrolled' if predicted_label == 1 else 'Declined'}")
    print(f"Enrollment probability: {enrollment_probability:.1%}")

    return {}


# ============================= Import‑Safety Call ==========================
# Populate module‑level state for external reuse of `predict_one`.
with contextlib.redirect_stdout(io.StringIO()), \
     contextlib.redirect_stderr(io.StringIO()):
    load_data({})   # config is empty because the original notebook had no CLI args


# ============================= Orchestration ==========================
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
