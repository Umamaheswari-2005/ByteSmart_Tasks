# ============================================================
# Converted from Google Colab / Jupyter Notebook via LLM
# Original file : Predictive-Grant-Funding-Success-Classifier-test.ipynb
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
# Import Path to work with file and folder paths.
# It helps us safely locate the saved model file.
from pathlib import Path

# Import pickle to load the trained machine learning pipeline.
# The pipeline was previously saved as a .pkl file using Python pickle.
import pickle

# Import pandas for creating and handling tabular input data.
# New grant proposal details will be converted into a DataFrame
# before sending them to the model.
import pandas as pd

# ================================================================
# Original notebook cell 14 (imports for evaluation)
# ================================================================
# Import train_test_split to recreate the same training and testing split.
from sklearn.model_selection import train_test_split

# Import evaluation metrics to measure model performance.
from sklearn.metrics import (
    accuracy_score,     # Measures overall correct predictions.
    precision_score,    # Measures how many predicted funded cases were actually funded.
    recall_score,       # Measures how many actual funded cases were correctly identified.
    f1_score,           # Combines precision and recall into one metric.
    roc_auc_score       # Measures the model's ability to separate the two classes.
)

# ================================================================
# Constants and reusable objects (module‑level, no side‑effects)
# ================================================================
# Define the exact order of features expected by the trained model.
FEATURE_ORDER = [
    'proposal_length',
    'requested_budget',
    'methodology_type',
    'agency',
    'applicant_state',
    'strategic_alignment_score',
    'team_size',
    'avg_team_experience_years',
    'prior_grants_won',
    'avg_publication_count',
    'reviewer_focus_match',
]

# Test cases used for sanity‑check predictions.
test_cases = [
    {
        'name': 'Strong, well-aligned proposal',
        'input': {
            'proposal_length': 3200,
            'requested_budget': 180000,
            'methodology_type': 'mixed_methods',
            'agency': 'NSF',
            'applicant_state': 'MA',
            'strategic_alignment_score': 0.85,
            'team_size': 5,
            'avg_team_experience_years': 12.0,
            'prior_grants_won': 3,
            'avg_publication_count': 30,
            'reviewer_focus_match': 0.80,
        },
    },
    {
        'name': 'Weak, poorly-aligned proposal',
        'input': {
            'proposal_length': 1500,
            'requested_budget': 90000,
            'methodology_type': 'qualitative',
            'agency': 'EPA',
            'applicant_state': 'TX',
            'strategic_alignment_score': 0.20,
            'team_size': 1,
            'avg_team_experience_years': 2.0,
            'prior_grants_won': 0,
            'avg_publication_count': 3,
            'reviewer_focus_match': 0.25,
        },
    },
    {
        'name': 'Edge case: very large budget ask, small team',
        'input': {
            'proposal_length': 5000,
            'requested_budget': 700000,
            'methodology_type': 'computational',
            'agency': 'DOE',
            'applicant_state': 'CA',
            'strategic_alignment_score': 0.55,
            'team_size': 2,
            'avg_team_experience_years': 6.0,
            'prior_grants_won': 1,
            'avg_publication_count': 15,
            'reviewer_focus_match': 0.50,
        },
    },
]

# Reusable prediction helper (kept with its original name).
def predict_one(model, raw_features: dict) -> dict:
    """
    Run the trained pipeline on one grant proposal
    and return the predicted funding class
    and funding‑success probability.
    """
    row = pd.DataFrame([raw_features])[FEATURE_ORDER]

    pred_class = int(model.predict(row)[0])
    pred_proba = float(model.predict_proba(row)[0, 1])

    return {
        'funded_prediction': pred_class,
        'funding_probability': round(pred_proba, 4)
    }

# ================================================================
# Helper for local file discovery (replaces Colab uploads)
# ================================================================
def _discover_file(extension: str, description: str) -> Path:
    """
    Search the current working directory for a file with the given extension.
    If none are found, raise FileNotFoundError.
    If more than one is found, raise RuntimeError listing the matches.
    Returns the Path of the single matching file.
    """
    matches = list(Path('.').glob(f'*{extension}'))
    if not matches:
        raise FileNotFoundError(f'No {description} file found in the current directory.')
    if len(matches) > 1:
        raise RuntimeError(
            f'Multiple {description} files found: {[p.name for p in matches]}'
        )
    return matches[0]

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
    Discover and load the trained pickle model.
    Mirrors the original notebook's upload‑then‑load flow.
    """
    # Discover the .pkl model file.
    model_path = _discover_file('.pkl', '.pkl model')
    model_file_name = model_path.name

    # Original notebook printed this after a successful upload.
    print(f'Model file uploaded successfully: {model_file_name}')

    # Load the model.
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    print('Pickle model loaded successfully.')
    print(f'Model path: {model_path.resolve()}')

    # Return a dict that will be threaded forward.
    return {
        'model': model,
        'model_path': model_path,
        'model_file_name': model_file_name,
    }

# ================================================================
# STAGE 2 - PREPROCESSING / FEATURE ENGINEERING
# ================================================================
def preprocess_data(raw_data, config):
    """
    No preprocessing is required for this pipeline.
    Pass through the model and expose test cases and feature order.
    """
    processed_data = {
        'model': raw_data['model'],
        'model_path': raw_data['model_path'],
        'model_file_name': raw_data['model_file_name'],
        'test_cases': test_cases,
        'FEATURE_ORDER': FEATURE_ORDER,
    }
    return processed_data

# ================================================================
# STAGE 3 - INITIALIZATION / SETUP
# ================================================================
def initialize(config, processed_data):
    """
    Populate the mutable state dictionary.
    """
    state = {
        'model': processed_data['model'],
        'FEATURE_ORDER': processed_data['FEATURE_ORDER'],
    }
    return state

# ================================================================
# STAGE 4 - CORE EXECUTION
# ================================================================
def execute(state, processed_data, config):
    """
    Run sanity‑check predictions on the first two test cases,
    then generate batch predictions for all cases.
    """
    model = state['model']

    # ---- Sanity‑check predictions ----
    strong_result = predict_one(model, processed_data['test_cases'][0]['input'])
    weak_result = predict_one(model, processed_data['test_cases'][1]['input'])

    assert strong_result['funding_probability'] > weak_result['funding_probability'], (
        'Expected the strong, well-aligned proposal to score higher than the weak one.'
    )
    print('Sanity check passed: strong proposal scores higher than weak proposal.')
    print(
        f"Strong: {strong_result['funding_probability']}  |  "
        f"Weak: {weak_result['funding_probability']}"
    )

    # ---- Batch predictions for all test cases ----
    batch_df = pd.DataFrame(
        [case['input'] for case in processed_data['test_cases']]
    )[processed_data['FEATURE_ORDER']]

    batch_df['funded_prediction'] = model.predict(batch_df)

    batch_df['funding_probability'] = (
        model.predict_proba(
            batch_df.drop(columns=['funded_prediction'])
        )[:, 1]
        .round(4)
    )

    results = {
        'strong_result': strong_result,
        'weak_result': weak_result,
        'batch_predictions': batch_df,
    }
    return results

# ================================================================
# STAGE 5 - VALIDATION / EVALUATION
# ================================================================
def evaluate(state, results, processed_data, config):
    """
    Load an evaluation CSV dataset (replacing the Colab upload)
    and compute standard classification metrics.
    """
    # Discover the evaluation CSV file.
    eval_path = _discover_file('.csv', 'evaluation CSV')
    evaluation_dataset_file = eval_path.name
    print(f'Evaluation dataset uploaded: {evaluation_dataset_file}')

    # Load the dataset.
    df = pd.read_csv(eval_path)
    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)

    # Inform that the model is ready for evaluation (mirrors original notebook).
    print("Saved pickle model is ready for evaluation.")
    print("Model type:", type(state['model']))

    # Define features and target (identical to the original notebook).
    features = [
        "proposal_length",
        "requested_budget",
        "methodology_type",
        "agency",
        "applicant_state",
        "strategic_alignment_score",
        "team_size",
        "avg_team_experience_years",
        "prior_grants_won",
        "avg_publication_count",
        "reviewer_focus_match"
    ]
    target = "funded"

    X = df[features]
    y = df[target]

    # Recreate the same train/test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print("\nTraining samples:", len(X_train))
    print("Testing samples :", len(X_test))

    # Predict on the test split.
    y_test_pred = state['model'].predict(X_test)
    y_test_proba = state['model'].predict_proba(X_test)[:, 1]

    # Compute metrics.
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_test_proba)

    # Display results (exactly as in the notebook).
    print("\n" + "=" * 60)
    print("              TEST DATA PERFORMANCE")
    print("=" * 60)
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print("=" * 60)

    evaluation = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'dataset_path': eval_path,
    }
    return evaluation

# ================================================================
# STAGE 6 - OUTPUT / PERSISTENCE
# ================================================================
def generate_output(state, results, evaluation, config):
    """
    Assemble the final output dictionary.
    No additional printing is performed here.
    """
    output = {
        'results': results,
        'evaluation': evaluation,
        'model_path': state.get('model_path', None),
    }
    return output

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
