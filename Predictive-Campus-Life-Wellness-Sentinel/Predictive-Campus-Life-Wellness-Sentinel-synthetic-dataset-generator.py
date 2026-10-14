# ============================================================
# Converted from Google Colab / Jupyter Notebook via LLM
# Original file : Predictive-Campus-Life-Wellness-Sentinel-synthetic-dataset-generator.ipynb
# LLM model     : openai/gpt-oss-120b
# Pipeline Step 1: Colab to Local
# ============================================================

"""
Standardized Pipeline Implementation
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ================================================================
# Original notebook cell 2
# ================================================================
SEED = 42
N_STUDENTS = 1000
N_WEEKS = 12
OUTPUT_PATH = Path("Predictive-Campus-Life-Wellness-Sentinel-dataset.csv")
rng = np.random.default_rng(SEED)

# ================================================================
# Original notebook cell 3
# ================================================================
US_STATES = [
    "California", "Texas", "Florida", "New York", "Pennsylvania",
    "Illinois", "Ohio", "Georgia", "North Carolina", "Michigan",
    "Washington", "Arizona", "Massachusetts", "Virginia", "Colorado",
    "Oregon", "Minnesota", "Tennessee", "Maryland", "Wisconsin"
]
INSTITUTION_TYPES = [
    "Public University", "Private University", "Community College"
]
DEPARTMENTS = [
    "Computer Science", "Information Technology", "Data Science",
    "Artificial Intelligence", "Electrical Engineering",
    "Mechanical Engineering", "Civil Engineering",
    "Business Administration", "Psychology", "Biology", "Nursing"
]

def generate_student_profiles(n_students):
    records = []
    for i in range(1, n_students + 1):
        records.append({
            "Student_ID": f"US{i:05d}",
            "State": rng.choice(US_STATES),
            "Institution_Type": rng.choice(INSTITUTION_TYPES, p=[0.55, 0.25, 0.20]),
            "Department": rng.choice(DEPARTMENTS),
            "Graduation_Year": int(rng.choice([2026, 2027, 2028, 2029, 2030])),
            "Cumulative_GPA": round(float(np.clip(rng.normal(3.0, 0.45), 1.5, 4.0)), 2),
            "Engagement_Level": float(np.clip(rng.normal(0.60, 0.18), 0.15, 0.95)),
            "Activity_Tendency": float(np.clip(rng.normal(0.55, 0.20), 0.10, 0.95)),
            "Achievement_Tendency": float(np.clip(rng.normal(0.35, 0.20), 0.05, 0.90))
        })
    return pd.DataFrame(records)

# ================================================================
# Original notebook cell 4
# ================================================================
def generate_weekly_record(student, week, engagement):
    lms_login_count = int(np.clip(rng.poisson(3 + engagement * 15), 0, 25))
    lms_active_days = int(np.clip(round(rng.normal(1 + engagement * 6, 0.8)), 0, 7))

    attendance = round(float(np.clip(55 + engagement * 40 + rng.normal(0, 5), 40, 100)), 2)
    academic_average = round(float(np.clip(55 + student["Cumulative_GPA"] * 8 + engagement * 12 + rng.normal(0, 5), 40, 100)), 2)
    assignment_rate = round(float(np.clip(45 + engagement * 55 + rng.normal(0, 6), 30, 100)), 2)
    library_count = int(np.clip(rng.poisson(max(engagement * 5, 0.05)), 0, 10))

    achievement_probability = float(np.clip(
        student["Achievement_Tendency"] * student["Activity_Tendency"] * engagement * 0.15,
        0, 0.20
    ))
    achievement_count = int(rng.binomial(1, achievement_probability))

    cultural_count = int(np.clip(rng.poisson(max(student["Activity_Tendency"] * engagement * 2.5, 0.02)), 0, 5))
    club_sports_count = int(np.clip(rng.poisson(max(student["Activity_Tendency"] * engagement * 2.0, 0.02)), 0, 5))
    mentor_probability = float(np.clip(0.05 + engagement * 0.20, 0, 0.30))
    mentor_count = int(rng.binomial(1, mentor_probability))

    return {
        "Student_ID": student["Student_ID"],
        "Week": week,
        "State": student["State"],
        "Institution_Type": student["Institution_Type"],
        "Department": student["Department"],
        "Graduation_Year": student["Graduation_Year"],
        "Achievement_Award_Count": achievement_count,
        "LMS_Login_Count": lms_login_count,
        "LMS_Active_Days": lms_active_days,
        "Attendance_Percentage": attendance,
        "Library_Login_Count": library_count,
        "Academic_Assessment_Average": academic_average,
        "Assignment_Submission_Rate": assignment_rate,
        "Cultural_Event_Participation_Count": cultural_count,
        "Club_Sports_Participation_Count": club_sports_count,
        "Academic_Advisor_Mentor_Meeting_Count": mentor_count,
        "Cumulative_GPA": student["Cumulative_GPA"]
    }

def generate_dataset(profiles, n_weeks):
    records = []
    for _, student in profiles.iterrows():
        engagement = student["Engagement_Level"]
        trend_type = rng.choice(["improving", "stable", "declining"], p=[0.20, 0.55, 0.25])
        if trend_type == "improving":
            weekly_trend = rng.uniform(0.005, 0.025)
        elif trend_type == "declining":
            weekly_trend = rng.uniform(-0.030, -0.008)
        else:
            weekly_trend = rng.uniform(-0.005, 0.005)

        for week in range(1, n_weeks + 1):
            engagement = float(np.clip(engagement + weekly_trend + rng.normal(0, 0.025), 0.05, 0.98))
            records.append(generate_weekly_record(student, week, engagement))
    return pd.DataFrame(records)

# ================================================================
# Original notebook cell 5
# ================================================================
def _add_target_and_validate(df):
    engagement_columns = [
        "LMS_Login_Count", "LMS_Active_Days", "Attendance_Percentage",
        "Library_Login_Count", "Assignment_Submission_Rate",
        "Cultural_Event_Participation_Count", "Club_Sports_Participation_Count",
        "Academic_Advisor_Mentor_Meeting_Count"
    ]
    engagement_score = sum(df[column].rank(pct=True) for column in engagement_columns) / len(engagement_columns)
    df["Risk_Level"] = pd.cut(
        engagement_score,
        bins=[-np.inf, 0.35, 0.65, np.inf],
        labels=["High", "Medium", "Low"]
    ).astype(str)

    assert df["Risk_Level"].notna().all()
    assert df["Cumulative_GPA"].between(1.5, 4.0).all()
    assert df["Attendance_Percentage"].between(40, 100).all()
    assert df["State"].isin(US_STATES).all()

    print("Target distribution:")
    print(df["Risk_Level"].value_counts())
    print("\nMissing values:", int(df.isnull().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))
    print("Dataset columns:", df.columns.tolist())
    return df

# ================================================================
# Original notebook cell 6
# ================================================================
def _save_dataset(df):
    df.to_csv(OUTPUT_PATH, index=False)
    print("US synthetic campus wellness dataset saved to:", OUTPUT_PATH)

# ================================================================
# Boilerplate functions
# ================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Pipeline implementation")
    args, _ = parser.parse_known_args()
    return vars(args)

def load_data(config):
    # Libraries imported messages (originally after imports)
    print("Libraries imported successfully.")
    print(f"Students: {N_STUDENTS}; monitoring period: {N_WEEKS} weeks")

    # Generate student profiles
    student_profiles = generate_student_profiles(N_STUDENTS)
    print("Student profiles generated:", student_profiles.shape)
    print(student_profiles.head())

    # Generate weekly dataset
    df = generate_dataset(student_profiles, N_WEEKS)
    print("Generated dataset shape:", df.shape)
    print(df.head())
    return df

def preprocess_data(raw_data, config):
    processed_df = _add_target_and_validate(raw_data)
    return processed_df

def initialize(config, processed_data):
    # No special state needed for this pipeline
    return {}

def execute(state, processed_data, config):
    # In this simple pipeline, execution just forwards the processed data
    results = processed_data
    return results

def evaluate(state, results, processed_data, config):
    # No evaluation metrics defined
    return {"status": "not_applicable"}

def generate_output(state, results, evaluation, config):
    _save_dataset(results)
    return {"output_path": str(OUTPUT_PATH)}

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
