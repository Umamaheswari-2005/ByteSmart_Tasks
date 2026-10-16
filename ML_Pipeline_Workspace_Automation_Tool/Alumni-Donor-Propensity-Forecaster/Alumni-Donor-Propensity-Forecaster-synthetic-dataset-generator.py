# ============================================================
# Converted from Google Colab / Jupyter Notebook via LLM
# Original file : Alumni-Donor-Propensity-Forecaster-synthetic-dataset-generator.ipynb
# LLM model     : openai/gpt-oss-120b
# Pipeline Step 1: Colab to Local
# ============================================================

"""
Standardized Pipeline Implementation
"""
import argparse
import subprocess
import sys

# ----------------------------------------------------------------------
# Cell 1: Import the required libraries
import pandas as pd
import numpy as np
# ----------------------------------------------------------------------
# Cell 2: Generate a consistent synthetic alumni dataset
def generate_alumni_dataset(n=5000, random_state=42, reference_year=2026):
    rng = np.random.default_rng(random_state)

    graduation_year = rng.integers(1980, 2025, n)
    # Approximate age as years since graduation plus a plausible graduation age.
    age = np.clip(
        (reference_year - graduation_year) + rng.integers(21, 29, n),
        22, 80
    )

    events_attended = rng.integers(0, 11, n)
    emails_received = rng.integers(5, 51, n)
    # An alumnus cannot open more emails than they received.
    emails_opened = np.array([rng.integers(0, received + 1) for received in emails_received])
    newsletter_clicks = rng.integers(0, 16, n)
    volunteer_events = rng.integers(0, 6, n)
    previous_donations = rng.integers(0, 6, n)
    # Keep donation amount consistent with donation history.
    total_donation_amount = np.where(
        previous_donations > 0,
        rng.integers(100, 10001, n),
        0
    )
    # Here donation_frequency is treated as a count and cannot exceed prior donations.
    donation_frequency = np.array([
        rng.integers(0, count + 1) for count in previous_donations
    ])
    # Use the maximum recency window for alumni who have never donated.
    days_since_last_donation = rng.integers(0, 1001, n)
    days_since_last_donation[previous_donations == 0] = 1000
    days_since_last_interaction = rng.integers(0, 501, n)

    data = pd.DataFrame({
        "alumni_id": np.arange(10001, 10001 + n),
        "graduation_year": graduation_year,
        "age": age,
        "events_attended": events_attended,
        "emails_received": emails_received,
        "emails_opened": emails_opened,
        "newsletter_clicks": newsletter_clicks,
        "volunteer_events": volunteer_events,
        "previous_donations": previous_donations,
        "total_donation_amount": total_donation_amount,
        "donation_frequency": donation_frequency,
        "days_since_last_donation": days_since_last_donation,
        "career_level": rng.choice(["Entry", "Mid", "Senior", "Executive"], n),
        "industry": rng.choice(["Technology", "Finance", "Healthcare", "Education", "Manufacturing", "Other"], n),
        "days_since_last_interaction": days_since_last_interaction,
    })

    # Synthetic target: higher engagement and previous giving increase propensity;
    # the offset/scale avoid making almost every record a positive class.
    score = (
        0.35 * data["events_attended"]
        + 0.04 * data["emails_opened"]
        + 0.08 * data["newsletter_clicks"]
        + 0.40 * data["volunteer_events"]
        + 0.80 * data["previous_donations"]
        + 0.00001 * data["total_donation_amount"]
        + 0.20 * data["donation_frequency"]
        - 0.002 * data["days_since_last_interaction"]
    )
    probability = 1 / (1 + np.exp(-((score - 6.0) / 2.5)))
    data["donated_next_12_months"] = (rng.random(n) < probability).astype(int)

    # Fail fast if a synthetic-data rule is accidentally broken later.
    assert (data["emails_opened"] <= data["emails_received"]).all()
    assert (data["donation_frequency"] <= data["previous_donations"]).all()
    assert (data.loc[data["previous_donations"] == 0, "total_donation_amount"] == 0).all()
    return data

# ----------------------------------------------------------------------
# STAGE 0 - ARGUMENTS / CONFIGURATION
def parse_arguments():
    parser = argparse.ArgumentParser(description="Pipeline implementation")
    args, _ = parser.parse_known_args()
    return vars(args)

# ----------------------------------------------------------------------
# STAGE 1 - INPUT
def load_data(config):
    # Cell 3: Generate the synthetic dataset with 5,000 records
    data = generate_alumni_dataset(n=5000, random_state=42)
    print("Dataset generated successfully!")

    # Cell 4: Validate dataset shape and columns, and display first 5 rows
    print("Dataset Shape:", data.shape)
    print("\nColumns in Dataset:")
    print(list(data.columns))
    print(data.head())  # display() converted to print()

    # Cell 5: Validate missing values, duplicates, business rules, and target distribution
    print("Missing Values:")
    print(data.isnull().sum())

    print("\nDuplicate Rows:", data.duplicated().sum())
    print("Emails opened greater than received:", (data["emails_opened"] > data["emails_received"]).sum())
    print("Donation frequency greater than previous donations:", (data["donation_frequency"] > data["previous_donations"]).sum())
    print("Donation amount present when previous donations are zero:", ((data["previous_donations"] == 0) & (data["total_donation_amount"] != 0)).sum())

    print("\nTarget Class Distribution:")
    print(data["donated_next_12_months"].value_counts().sort_index())
    print("Target Class Proportions:")
    print(data["donated_next_12_months"].value_counts(normalize=True).sort_index().round(3))

    return data

# ----------------------------------------------------------------------
# STAGE 2 - PREPROCESSING / FEATURE ENGINEERING
def preprocess_data(raw_data, config):
    # No preprocessing needed for this synthetic dataset.
    processed_data = {"data": raw_data}
    return processed_data

# ----------------------------------------------------------------------
# STAGE 3 - INITIALIZATION / SETUP
def initialize(config, processed_data):
    # Propagate needed artifacts to state.
    state = {"data": processed_data["data"]}
    return state

# ----------------------------------------------------------------------
# STAGE 4 - CORE EXECUTION
def execute(state, processed_data, config):
    # No core execution beyond data generation; return empty results.
    results = {}
    return results

# ----------------------------------------------------------------------
# STAGE 5 - VALIDATION / EVALUATION
def evaluate(state, results, processed_data, config):
    # No evaluation logic defined in the original notebook.
    return {"status": "not_applicable"}

# ----------------------------------------------------------------------
# STAGE 6 - OUTPUT / PERSISTENCE
def generate_output(state, results, evaluation, config):
    # Cell 6: Save the generated dataset as a CSV file
    output_path = "Alumni-Donor-Propensity-Forecaster-synthetic-dataset.csv"
    state["data"].to_csv(output_path, index=False)
    print(f"Dataset saved successfully to: {output_path}")

    # Cell 7: Download the saved CSV file directly to your local computer
    # The original Colab download command is omitted for local execution.
    # files.download("Alumni-Donor-Propensity-Forecaster-synthetic-dataset.csv")
    return {"output_path": output_path}

# ----------------------------------------------------------------------
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
