# ============================================================
# Converted from Google Colab / Jupyter Notebook via LLM
# Original file : strategic-financial-aid-yield-optimizer-dataset-generator.ipynb
# LLM model     : openai/gpt-oss-120b
# Pipeline Step 1: Colab to Local
# ============================================================

"""
Standardized Pipeline Implementation
"""
import argparse
import numpy as np
import pandas as pd

# ================================================================
# Original notebook cell 2
# NumPy is used for random-number generation and numerical calculations.
# Pandas is used to create, organize, and manipulate the applicant dataset.
# ================================================================

# ================================================================
# Original notebook cell 3
# Constants and configuration values
# ================================================================
STATE_WEIGHTS = {
    "CA": 0.118, "TX": 0.089, "FL": 0.065, "NY": 0.058, "PA": 0.038,
    "IL": 0.037, "OH": 0.035, "GA": 0.032, "NC": 0.031, "MI": 0.030,
    "NJ": 0.027, "VA": 0.026, "WA": 0.023, "AZ": 0.022, "MA": 0.021,
    "TN": 0.021, "IN": 0.020, "MO": 0.018, "MD": 0.018, "WI": 0.018,
    "CO": 0.017, "MN": 0.017, "SC": 0.016, "AL": 0.015, "LA": 0.014,
    "KY": 0.013, "OR": 0.013, "OK": 0.012, "CT": 0.011, "UT": 0.010,
    "IA": 0.010, "NV": 0.009, "AR": 0.009, "MS": 0.009, "KS": 0.009,
    "NM": 0.006, "NE": 0.006, "WV": 0.005, "ID": 0.005, "HI": 0.004,
    "NH": 0.004, "ME": 0.004, "MT": 0.003, "RI": 0.003, "DE": 0.003,
    "SD": 0.003, "ND": 0.002, "AK": 0.002, "VT": 0.002, "WY": 0.002,
}
STATES = list(STATE_WEIGHTS.keys())
STATE_P = np.array(list(STATE_WEIGHTS.values()))
STATE_P = STATE_P / STATE_P.sum()

HOME_STATE = "OH"

LOCALE_CATEGORIES = ["City", "Suburb", "Town", "Rural"]
LOCALE_P = [0.30, 0.40, 0.15, 0.15]

TIER_CATEGORIES = [
    "R1 Research",
    "R2 Doctoral",
    "Regional Comprehensive",
    "Liberal Arts College",
    "Community College",
]
TIER_P = [0.18, 0.14, 0.34, 0.14, 0.20]

TIER_BASE_COA = {
    "R1 Research": 58000,
    "R2 Doctoral": 42000,
    "Regional Comprehensive": 31000,
    "Liberal Arts College": 63000,
    "Community College": 19000,
}
TIER_COA_SD = {
    "R1 Research": 9000,
    "R2 Doctoral": 6000,
    "Regional Comprehensive": 4500,
    "Liberal Arts College": 10000,
    "Community College": 2500,
}
OUT_OF_STATE_PREMIUM = 14000
RNG_MIN_COA = 8000
MAX_AID_SHARE_OF_COA = 0.95
NPIR_CAP = 3.0
SAT_PLAUSIBLE_FLOOR = 1.0

# ================================================================
# Original notebook cell 4
# Helper sigmoid function
# ================================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# ================================================================
# Original notebook cell 5
# Dataset generation function
# ================================================================
def generate_dataset(n=10000, seed=42):
    rng = np.random.default_rng(seed)

    # Geography
    state_residency = rng.choice(STATES, size=n, p=STATE_P)
    urban_centric_locale = rng.choice(LOCALE_CATEGORIES, size=n, p=LOCALE_P)
    is_in_state = (state_residency == HOME_STATE)

    base_miles = rng.lognormal(mean=4.2, sigma=1.05, size=n)
    miles_from_campus = np.where(
        is_in_state,
        np.clip(base_miles * 0.55, 2, 600),
        np.clip(base_miles * 1.9, 15, 3000),
    ).round(1)

    # Household financial profile
    adjusted_gross_income = rng.lognormal(mean=11.05, sigma=0.72, size=n)
    adjusted_gross_income = np.clip(adjusted_gross_income, 0, 650_000).round(0)

    family_size_factor = rng.normal(loc=1.0, scale=0.18, size=n)
    family_size_factor = np.clip(family_size_factor, 0.55, 1.6)

    sai_raw = (adjusted_gross_income * 0.22) / family_size_factor
    sai_noise = rng.normal(loc=0, scale=3500, size=n)
    student_aid_index = np.clip(sai_raw + sai_noise - 1800, -1500, 220_000).round(0)

    # Financial‑need indicators
    pell_threshold = np.quantile(student_aid_index, 0.34)
    pell_eligible = (student_aid_index <= pell_threshold).astype(int)

    first_gen_logit = -0.55 + (-1.15e-5) * (adjusted_gross_income - 60000)
    first_gen_p = np.clip(sigmoid(first_gen_logit), 0.05, 0.85)
    first_gen = rng.binomial(1, first_gen_p)

    # Academic profile
    gpa_beta = rng.beta(a=6.2, b=2.3, size=n)
    gpa_adj = (0.06 * (1 - first_gen) + 0.00000025 * (adjusted_gross_income - 60000))
    hs_gpa = np.clip(gpa_beta * 4.0 + gpa_adj, 1.5, 4.0).round(2)

    sat_core = 55 + (hs_gpa - 3.0) * 38 + rng.normal(0, 13, size=n)
    sat_core_internal = np.clip(sat_core, 1, 99)

    sat_act_percentile = np.where(sat_core < SAT_PLAUSIBLE_FLOOR, np.nan, sat_core)
    sat_act_percentile = np.clip(sat_act_percentile, 1, 99)
    sat_act_percentile = np.round(sat_act_percentile, 0)

    # Institutional assignment and cost
    institutional_tier = rng.choice(TIER_CATEGORIES, size=n, p=TIER_P)

    base_coa = np.array([TIER_BASE_COA[t] for t in institutional_tier])
    coa_sd = np.array([TIER_COA_SD[t] for t in institutional_tier])
    coa_noise = rng.normal(0, 1, size=n) * coa_sd

    oos_premium = np.where(
        (~is_in_state) & (institutional_tier != "Community College"),
        OUT_OF_STATE_PREMIUM,
        0,
    )
    cost_of_attendance = np.clip(base_coa + coa_noise + oos_premium, RNG_MIN_COA, None).round(0)

    # Behavioral and engagement profile
    month_weights = np.array([0.20, 0.16, 0.12, 0.09, 0.07, 0.05,
                              0.04, 0.04, 0.05, 0.07, 0.05, 0.06])
    month_weights = month_weights / month_weights.sum()

    fafsa_submitted_month = rng.choice(np.arange(1, 13), size=n, p=month_weights)
    fafsa_month_sin = np.sin(2 * np.pi * fafsa_submitted_month / 12).round(4)
    fafsa_month_cos = np.cos(2 * np.pi * fafsa_submitted_month / 12).round(4)

    engagement_base = rng.normal(loc=52, scale=18, size=n)
    engagement_base += (13 - fafsa_submitted_month).clip(0, None) * 0.9
    engagement_base += first_gen * 4
    demonstrated_interest = np.clip(engagement_base, 0, 100).round(1)

    # Financial‑aid awarding logic
    sat_component = sat_core_internal / 100.0
    academic_strength = (hs_gpa / 4.0) * 0.5 + sat_component * 0.5

    tier_merit_generosity = np.array([
        {
            "R1 Research": 0.9,
            "R2 Doctoral": 1.15,
            "Regional Comprehensive": 1.35,
            "Liberal Arts College": 1.25,
            "Community College": 0.4,
        }[t] for t in institutional_tier
    ])

    merit_scholarship_amt = np.clip(
        academic_strength * 0.42 * cost_of_attendance * tier_merit_generosity
        + rng.normal(0, 1500, size=n),
        0,
        cost_of_attendance * 0.9,
    ).round(0)

    need_gap_proxy = np.clip(cost_of_attendance - student_aid_index, 0, None)
    need_generosity = np.where(pell_eligible == 1, 0.55, 0.28)

    need_grant_amt = np.clip(
        need_gap_proxy * need_generosity * rng.uniform(0.5, 1.0, size=n),
        0,
        cost_of_attendance * 0.85,
    ).round(0)

    total_aid_package = np.clip(
        merit_scholarship_amt + need_grant_amt,
        0,
        cost_of_attendance * MAX_AID_SHARE_OF_COA,
    ).round(0)

    net_price = np.clip(cost_of_attendance - total_aid_package, 0, None).round(0)

    npir = (cost_of_attendance - total_aid_package) / np.clip(adjusted_gross_income, 1000, None)
    npir = np.clip(npir, 0, 5).round(4)

    fadr = (total_aid_package / cost_of_attendance).round(4)

    ufng = cost_of_attendance - (student_aid_index + total_aid_package)
    ufng = np.clip(ufng, 0, None).round(0)

    days_to_decision = rng.integers(30, 150, size=n)
    touchpoints = np.clip(rng.poisson(lam=demonstrated_interest / 8.0, size=n), 0, None)
    engagement_velocity = (touchpoints / days_to_decision).round(4)

    tier_prestige = np.array([
        {
            "R1 Research": 0.85,
            "R2 Doctoral": 0.55,
            "Regional Comprehensive": 0.25,
            "Liberal Arts College": 0.65,
            "Community College": 0.05,
        }[t] for t in institutional_tier
    ])

    overqualification = np.clip(academic_strength - tier_prestige, -1, 1)

    logit = (
        1.55
        - 3.1 * npir
        - 0.0000045 * ufng
        + 0.021 * (demonstrated_interest - 50)
        - 0.0016 * miles_from_campus
        - 1.35 * overqualification
        + 0.55 * fadr
        + 0.15 * pell_eligible * (need_grant_amt > 0)
        + rng.normal(0, 0.55, size=n)
    )
    enroll_prob = sigmoid(logit)
    enrolled = rng.binomial(1, enroll_prob)

    df = pd.DataFrame({
        "state_residency": state_residency,
        "is_in_state": is_in_state.astype(int),
        "urban_centric_locale": urban_centric_locale,
        "student_aid_index": student_aid_index,
        "adjusted_gross_income": adjusted_gross_income,
        "first_gen": first_gen,
        "pell_eligible": pell_eligible,
        "hs_gpa": hs_gpa,
        "sat_act_percentile": sat_act_percentile,
        "institutional_tier": institutional_tier,
        "cost_of_attendance": cost_of_attendance,
        "miles_from_campus": miles_from_campus,
        "fafsa_submitted_month": fafsa_submitted_month,
        "fafsa_month_sin": fafsa_month_sin,
        "fafsa_month_cos": fafsa_month_cos,
        "demonstrated_interest": demonstrated_interest,
        "merit_scholarship_amt": merit_scholarship_amt,
        "need_grant_amt": need_grant_amt,
        "total_aid_package": total_aid_package,
        "net_price": net_price,
        "enrolled": enrolled,
        "net_price_to_income_ratio": npir,
        "financial_aid_discount_rate": fadr,
        "unmet_financial_need_gap": ufng,
        "engagement_velocity": engagement_velocity,
    })
    return df

# ================================================================
# Original notebook cell 6
# Model‑ready dataset builder
# ================================================================
def build_model_ready_dataset(df):
    model_df = df.copy()

    cols_to_drop = ["total_aid_package", "state_residency", "fafsa_submitted_month"]
    model_df = model_df.drop(columns=[col for col in cols_to_drop if col in model_df.columns])

    tier_mapping = {
        "Community College": 0,
        "Regional Comprehensive": 1,
        "R2 Doctoral": 2,
        "R1 Research": 3,
        "Liberal Arts College": 4,
    }
    if "institutional_tier" in model_df.columns:
        model_df["institutional_tier"] = model_df["institutional_tier"].map(tier_mapping)

    if "urban_centric_locale" in model_df.columns:
        model_df = pd.get_dummies(model_df, columns=["urban_centric_locale"], drop_first=True, dtype=int)

    if "net_price_to_income_ratio" in model_df.columns:
        model_df["net_price_to_income_ratio"] = np.log1p(model_df["net_price_to_income_ratio"])

    # Note: NaNs in sat_act_percentile are left as‑is intentionally.
    return model_df

# ================================================================
# Original notebook cell 7
# Generation parameters
# ================================================================
N = 6000
SEED = 42

# ================================================================
# Original notebook cell 8
# Output file paths
# ================================================================
RAW_OUTPUT_PATH = "financial_aid_yield_dataset.csv"
MODEL_OUTPUT_PATH = "financial_aid_yield_dataset_model_ready.csv"

# ================================================================
# Boilerplate – Stage 0: Argument parsing
# ================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Pipeline implementation")
    args, _ = parser.parse_known_args()
    return vars(args)

# ================================================================
# Stage 1 – Load data
# ================================================================
def load_data(config):
    # Generate the synthetic financial‑aid yield dataset.
    df = generate_dataset(n=N, seed=SEED)
    return df

# ================================================================
# Stage 2 – Preprocess data
# ================================================================
def preprocess_data(raw_data, config):
    # Build a model‑ready version of the dataset.
    model_df = build_model_ready_dataset(raw_data)
    # Thread both raw and model‑ready data forward.
    processed_data = {
        "raw_data": raw_data,
        "model_df": model_df,
    }
    return processed_data

# ================================================================
# Stage 3 – Initialization / Setup
# ================================================================
def initialize(config, processed_data):
    # In this pipeline no additional state is required beyond the
    # processed artifacts; simply forward them.
    state = dict(processed_data)
    return state

# ================================================================
# Stage 4 – Core execution
# ================================================================
def execute(state, processed_data, config):
    # No further computation is needed; return an empty results dict.
    results = {}
    return results

# ================================================================
# Stage 5 – Evaluation
# ================================================================
def evaluate(state, results, processed_data, config):
    # No evaluation metrics are defined for this data‑generation script.
    return {"status": "not_applicable"}

# ================================================================
# Stage 6 – Output / Persistence
# ================================================================
def generate_output(state, results, evaluation, config):
    raw_data = state["raw_data"]
    model_df = state["model_df"]

    # Save CSV files.
    raw_data.to_csv(RAW_OUTPUT_PATH, index=False)
    model_df.to_csv(MODEL_OUTPUT_PATH, index=False)

    # Confirmation messages (preserved from the original notebook).
    print("Dataset generated successfully!")
    print("Raw dataset saved at:", RAW_OUTPUT_PATH)
    print("Model-ready dataset saved at:", MODEL_OUTPUT_PATH)

    # Display first five records.
    print("\nFirst five raw records:")
    print(raw_data.head())

    print("\nFirst five model ready records:")
    print(model_df.head())

    # Display shapes.
    print("Raw dataset shape:", raw_data.shape)
    print("Model-ready dataset shape:", model_df.shape)

    # Display column names.
    print("\nDataset columns:")
    print(raw_data.columns.tolist())

    print("\nDataset columns (model-ready):")
    print(model_df.columns.tolist())

    # Display basic statistics.
    print("\nDataset summary:")
    print(raw_data.describe())

    print("\nDataset summary (model-ready):")
    print(model_df.describe())

    # Important dataset statistics.
    print(f"\nEnrollment rate: {raw_data['enrolled'].mean():.1%}")
    print(f"Pell-eligible rate: {raw_data['pell_eligible'].mean():.1%}")
    print(f"First-generation rate: {raw_data['first_gen'].mean():.1%}")
    print(f"Median AGI: ${raw_data['adjusted_gross_income'].median():,.0f}")
    print(f"Median net price: ${raw_data['net_price'].median():,.0f}")
    print(f"Median NPIR: {raw_data['net_price_to_income_ratio'].median():.2f}")
    print(f"Missing sat_act_percentile (test-optional non-submitters): {raw_data['sat_act_percentile'].isna().mean():.1%}")
    print(f"Min unmet_financial_need_gap: {raw_data['unmet_financial_need_gap'].min()} (should be >= 0)")
    print(f"Max financial_aid_discount_rate: {raw_data['financial_aid_discount_rate'].max():.4f} (should be <= {MAX_AID_SHARE_OF_COA})")

    # Dataset info (replicates df.info()).
    print('\nDataset Info : \n')
    raw_data.info()
    print('\nDataset Info (model-ready) : \n')
    model_df.info()

    # The original notebook used `files.download` to trigger a download in Colab.
    # This step is omitted in the pure‑Python script.
    return {"status": "output_generated"}

# ================================================================
# Stage 7 – Orchestration
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
