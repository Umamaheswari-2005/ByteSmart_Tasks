# ============================================================
# Converted from Google Colab / Jupyter Notebook via LLM
# Original file : Predictive-Campus-Life-Wellness-Sentinel-test.ipynb
# LLM model     : openai/gpt-oss-120b
# Pipeline Step 1: Colab to Local
# ============================================================

"""
Standardized Pipeline Implementation
"""
import argparse
import subprocess
import sys

# ================================================================
# Original notebook cell 2
# ================================================================
import io
import pickle
import pandas as pd

def parse_arguments():
    """
    No command‑line arguments are defined in the original notebook.
    """
    parser = argparse.ArgumentParser(description="Pipeline implementation")
    args, _ = parser.parse_known_args()
    return vars(args)


def load_data(config):
    """
    Loads the pickled classification pipeline that was originally uploaded
    via `files.upload()` in the notebook.  The upload step is replaced by a
    local‑file discovery that looks for a single ``*.pkl`` file in the current
    working directory.
    """
    # ------------------------------------------------------------
    # Local file discovery (replacement for `files.upload()`)
    # ------------------------------------------------------------
    from pathlib import Path

    pkl_files = list(Path.cwd().glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(
            "No .pkl model file found in the current directory. "
            "Place the trained model file here before running the script."
        )
    if len(pkl_files) > 1:
        raise RuntimeError(
            f"Multiple .pkl files found: {[p.name for p in pkl_files]}. "
            "Exactly one model file must be present."
        )
    model_path = pkl_files[0]

    # ------------------------------------------------------------
    # Load the pipeline
    # ------------------------------------------------------------
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    if not hasattr(pipeline, "predict") or not hasattr(pipeline, "predict_proba"):
        raise TypeError(
            "The loaded pickle does not contain a compatible classification pipeline."
        )

    print("Pickle model loaded successfully:", model_path.name)
    print("Pipeline steps:", list(pipeline.named_steps.keys()))

    # Return a dict so that later stages can access the pipeline.
    return {"pipeline": pipeline}


# ================================================================
# Original notebook cell 3
# ================================================================
def preprocess_data(raw_data, config):
    """
    Constructs the unseen‑student DataFrame and aligns its columns with the
    feature names expected by the loaded pipeline.
    """
    # ---- Unseen student examples (original cell 3) ----
    unseen_students = pd.DataFrame(
        [
            {
                "State": "California",
                "Institution_Type": "Public University",
                "Department": "Computer Science",
                "Graduation_Year": 2029,
                "Achievement_Award_Count": 4,
                "LMS_Login_Count": 22,
                "LMS_Active_Days": 7,
                "Attendance_Percentage": 96.0,
                "Library_Login_Count": 9,
                "Academic_Assessment_Average": 91.0,
                "Assignment_Submission_Rate": 98.0,
                "Cultural_Event_Participation_Count": 4,
                "Club_Sports_Participation_Count": 3,
                "Academic_Advisor_Mentor_Meeting_Count": 2,
                "Cumulative_GPA": 3.8,
            },
            {
                "State": "Texas",
                "Institution_Type": "Community College",
                "Department": "Business Administration",
                "Graduation_Year": 2028,
                "Achievement_Award_Count": 1,
                "LMS_Login_Count": 12,
                "LMS_Active_Days": 4,
                "Attendance_Percentage": 78.0,
                "Library_Login_Count": 4,
                "Academic_Assessment_Average": 70.0,
                "Assignment_Submission_Rate": 76.0,
                "Cultural_Event_Participation_Count": 2,
                "Club_Sports_Participation_Count": 2,
                "Academic_Advisor_Mentor_Meeting_Count": 1,
                "Cumulative_GPA": 2.9,
            },
            {
                "State": "New York",
                "Institution_Type": "Private University",
                "Department": "Psychology",
                "Graduation_Year": 2027,
                "Achievement_Award_Count": 0,
                "LMS_Login_Count": 3,
                "LMS_Active_Days": 1,
                "Attendance_Percentage": 52.0,
                "Library_Login_Count": 1,
                "Academic_Assessment_Average": 46.0,
                "Assignment_Submission_Rate": 42.0,
                "Cultural_Event_Participation_Count": 0,
                "Club_Sports_Participation_Count": 0,
                "Academic_Advisor_Mentor_Meeting_Count": 0,
                "Cumulative_GPA": 1.8,
            },
            {
                "State": "Colorado",
                "Institution_Type": "Public University",
                "Department": "Biology",
                "Graduation_Year": 2030,
                "Achievement_Award_Count": 2,
                "LMS_Login_Count": 16,
                "LMS_Active_Days": 5,
                "Attendance_Percentage": 86.0,
                "Library_Login_Count": 6,
                "Academic_Assessment_Average": 79.0,
                "Assignment_Submission_Rate": 87.0,
                "Cultural_Event_Participation_Count": 3,
                "Club_Sports_Participation_Count": 2,
                "Academic_Advisor_Mentor_Meeting_Count": 1,
                "Cumulative_GPA": 3.3,
            },
        ]
    )

    # Align input columns with the model's recorded feature names, if available.
    pipeline = raw_data["pipeline"]
    expected_features = getattr(pipeline, "feature_names_in_", None)
    if expected_features is not None:
        missing = set(expected_features) - set(unseen_students.columns)
        extra = set(unseen_students.columns) - set(expected_features)
        if missing:
            raise ValueError(
                f"Unseen inputs are missing model features: {sorted(missing)}"
            )
        if extra:
            unseen_students = unseen_students[list(expected_features)]

    # Return everything that later stages may need.
    return {
        "unseen_students": unseen_students,
        "pipeline": pipeline,
    }


# ================================================================
# Original notebook cell 4
# ================================================================
def initialize(config, processed_data):
    """
    In this pipeline the only state required for execution is the loaded
    ``pipeline`` object.  It is passed forward via the ``state`` dictionary.
    """
    return {"pipeline": processed_data["pipeline"]}


def execute(state, processed_data, config):
    """
    Runs inference on the unseen student profiles.
    """
    pipeline = state["pipeline"]
    unseen_students = processed_data["unseen_students"]

    predictions = pipeline.predict(unseen_students)
    probabilities = pipeline.predict_proba(unseen_students)
    classes = pipeline.named_steps["classifier"].classes_

    results = unseen_students.copy()
    results["Predicted_Risk_Level"] = predictions
    results["Prediction_Confidence"] = probabilities.max(axis=1).round(4)

    for class_index, class_name in enumerate(classes):
        results[f"Probability_{class_name}"] = probabilities[:, class_index].round(4)

    return results


def evaluate(state, results, processed_data, config):
    """
    No explicit evaluation step is defined in the original notebook.
    """
    return {"status": "not_applicable"}


def generate_output(state, results, evaluation, config):
    """
    Mirrors the notebook's final output: prints the results DataFrame and a
    completion message.
    """
    # ``display`` is replaced by ``print`` as required for plain scripts.
    print(results)
    print("Inference completed for", len(results), "unseen student profiles.")
    return results


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
