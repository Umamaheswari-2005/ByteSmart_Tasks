# ============================================================
# Converted from Google Colab / Jupyter Notebook via LLM
# Original file : Alumni-Donor-Propensity-Forecaster-test.ipynb
# LLM model     : openai/gpt-oss-120b
# Pipeline Step 1: Colab to Local
# ============================================================

"""
Standardized Pipeline Implementation
"""
import argparse
import os
import pickle
import numpy as np
import pandas as pd
import sklearn

# ================================================================
# Original notebook cell 1
# ================================================================
print("Libraries imported successfully!")
print("scikit-learn version:", sklearn.__version__)

# ================================================================
# Original notebook cell 2
# ================================================================
print("Environment check completed.")

# ================================================================
# Constants and helper functions (original notebook cells)
# ================================================================
REFERENCE_YEAR = 2026


def engineer_features(frame, reference_year=REFERENCE_YEAR):
    """Apply the same feature engineering used during training."""
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


# ================================================================
# STAGE 0 - ARGUMENTS / CONFIGURATION
# ================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Pipeline implementation")
    args, _ = parser.parse_known_args()
    return vars(args)


# ================================================================
# STAGE 1 - INPUT
# ================================================================
def load_data(config):
    """
    Load the trained model from a local .pkl file.
    The original notebook used `files.upload()`; here we discover the file locally.
    """
    expected_model_name = "Alumni-Donor-Propensity-Forecaster-model_pipeline.pkl"
    print(f"Looking for model file: {expected_model_name}")

    # Discover matching files in the current working directory
    matches = [
        f for f in os.listdir(".") if f.lower() == expected_model_name.lower()
    ]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Model file {expected_model_name} not found in the current directory."
        )
    if len(matches) > 1:
        raise RuntimeError(f"Multiple model files found: {matches}")

    model_file = matches[0]

    with open(model_file, "rb") as file:
        model = pickle.load(file)

    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise TypeError(
            "The loaded object is not a prediction pipeline with predict_proba()."
        )

    print("Model loaded successfully!")
    print("Model file:", model_file)
    print(
        "File size:",
        round(os.path.getsize(model_file) / 1024, 2),
        "KB",
    )
    print("Model type:", type(model))

    # Return both the model and its filename for later use
    return {"model": model, "model_file": model_file}


# ================================================================
# STAGE 2 - PREPROCESSING / FEATURE ENGINEERING
# ================================================================
def preprocess_data(raw_data, config):
    """
    In this pipeline the raw_data is the dictionary returned by load_data.
    No additional preprocessing is required at this stage.
    """
    return raw_data  # simply forward the dictionary containing the model


# ================================================================
# STAGE 3 - INITIALIZATION / SETUP
# ================================================================
def initialize(config, processed_data):
    """
    Prepare the runtime state. Here we keep a reference to the model and its filename.
    """
    state = {
        "model": processed_data["model"],
        "model_file": processed_data["model_file"],
    }
    return state


# ================================================================
# STAGE 4 - CORE EXECUTION
# ================================================================
def execute(state, processed_data, config):
    """
    Perform a single-sample prediction, compute the propensity score,
    and run the batch of test cases.
    """
    model = state["model"]

    # ---- Single sample prediction (original cells 5‑7) ----
    sample_alumni = {
        "graduation_year": 2015,
        "age": 32,
        "events_attended": 5,
        "emails_received": 30,
        "emails_opened": 21,
        "newsletter_clicks": 8,
        "volunteer_events": 2,
        "previous_donations": 3,
        "total_donation_amount": 3500,
        "donation_frequency": 2,
        "days_since_last_donation": 200,
        "career_level": "Senior",
        "industry": "Technology",
        "days_since_last_interaction": 30,
    }
    input_data = pd.DataFrame([sample_alumni])
    input_data = engineer_features(input_data)

    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0, 1])
    propensity_score = probability * 100

    if propensity_score >= 80:
        category, interpretation = "High", "Likely to Donate"
    elif propensity_score >= 50:
        category, interpretation = "Medium", "Moderate Donation Likelihood"
    else:
        category, interpretation = "Low", "Lower Donation Likelihood"

    sample_result = {
        "prediction": prediction,
        "probability": probability,
        "propensity_score": propensity_score,
        "category": category,
        "interpretation": interpretation,
        "input_features": input_data,
    }

    # ---- Batch test cases (original cell 10) ----
    test_cases = [
        {
            "name": "Highly Engaged Alumni",
            "graduation_year": 2020,
            "age": 30,
            "events_attended": 8,
            "emails_received": 40,
            "emails_opened": 35,
            "newsletter_clicks": 12,
            "volunteer_events": 5,
            "previous_donations": 5,
            "total_donation_amount": 5000,
            "donation_frequency": 4,
            "days_since_last_donation": 30,
            "career_level": "Executive",
            "industry": "Technology",
            "days_since_last_interaction": 10,
        },
        {
            "name": "Moderately Engaged Alumni",
            "graduation_year": 2015,
            "age": 35,
            "events_attended": 4,
            "emails_received": 25,
            "emails_opened": 12,
            "newsletter_clicks": 5,
            "volunteer_events": 2,
            "previous_donations": 2,
            "total_donation_amount": 1000,
            "donation_frequency": 1,
            "days_since_last_donation": 200,
            "career_level": "Senior",
            "industry": "Finance",
            "days_since_last_interaction": 100,
        },
        {
            "name": "Low Engagement Alumni",
            "graduation_year": 2005,
            "age": 45,
            "events_attended": 0,
            "emails_received": 20,
            "emails_opened": 2,
            "newsletter_clicks": 0,
            "volunteer_events": 0,
            "previous_donations": 0,
            "total_donation_amount": 0,
            "donation_frequency": 0,
            "days_since_last_donation": 1000,
            "career_level": "Mid",
            "industry": "Manufacturing",
            "days_since_last_interaction": 450,
        },
    ]

    batch_results = []
    for case in test_cases:
        case_name = case["name"]
        raw_case = pd.DataFrame(
            [{k: v for k, v in case.items() if k != "name"}]
        )
        case_features = engineer_features(raw_case)
        prob = float(model.predict_proba(case_features)[0, 1]) * 100

        if prob >= 80:
            cat, interp = "High", "Likely to Donate"
        elif prob >= 50:
            cat, interp = "Medium", "Moderate Donation Likelihood"
        else:
            cat, interp = "Low", "Lower Donation Likelihood"

        batch_results.append(
            {
                "Alumni": case_name,
                "Propensity Score": f"{prob:.2f}%",
                "Category": cat,
                "Interpretation": interp,
            }
        )

    results = {
        "sample": sample_result,
        "batch": batch_results,
    }
    return results


# ================================================================
# STAGE 5 - VALIDATION / EVALUATION
# ================================================================
def evaluate(state, results, processed_data, config):
    """
    No explicit evaluation metrics are defined in the original notebook.
    """
    return {"status": "not_applicable"}


# ================================================================
# STAGE 6 - OUTPUT / PERSISTENCE
# ================================================================
def generate_output(state, results, evaluation, config):
    """
    Replicate the original notebook's print‑based reporting.
    """
    # ---- Sample prediction output (original cells 7‑9) ----
    sample = results["sample"]
    print("Predicted Class:", sample["prediction"])
    print(
        "Interpretation:",
        "Donated" if sample["prediction"] == 1 else "Did not donate",
    )
    print("\n========== DONOR PROPENSITY RESULT ==========")
    print(f"Donation Probability : {sample['probability']:.4f}")
    print(
        f"Propensity Score     : {sample['propensity_score']:.2f}%"
    )
    print(f"Category             : {sample['category']}")
    print(f"Interpretation       : {sample['interpretation']}")
    print("==============================================")
    print(
        "\nNote: This propensity score is an estimated likelihood, not a guarantee."
    )
    print("Propensity Score:", round(sample["propensity_score"], 2), "%")
    print("Category:", sample["category"])
    print("Interpretation:", sample["interpretation"])

    # ---- Batch test case table (original cell 10) ----
    batch_df = pd.DataFrame(results["batch"])
    print("\nBatch Test Cases Results:")
    print(batch_df)

    # ---- Final model test summary (original cell 11) ----
    print("\n--- FINAL MODEL TEST SUMMARY ---")
    print("Model type:", type(state["model"]).__name__)
    print("Task: Binary Classification")
    print("Prediction target: donated_next_12_months")
    print("Output: Donation probability / propensity score")
    print("Categories: High, Medium, Low")
    print("Model source:", state["model_file"])
    print(
        "\nThis script tests the saved model; training is performed in the training notebook."
    )
    return {"status": "completed"}


# ================================================================
# STAGE 7 - ORCHESTRATION
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
