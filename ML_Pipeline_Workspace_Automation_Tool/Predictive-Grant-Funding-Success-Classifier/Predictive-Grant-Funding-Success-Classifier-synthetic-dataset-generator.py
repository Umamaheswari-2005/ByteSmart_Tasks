# ============================================================
# Converted from Google Colab / Jupyter Notebook via LLM
# Original file : Predictive-Grant-Funding-Success-Classifier-synthetic-dataset-generator.ipynb
# LLM model     : openai/gpt-oss-120b
# Pipeline Step 1: Colab to Local
# ============================================================

"""
Standardized Pipeline Implementation
"""
import argparse

# ================================================================
# Original notebook cell 2
# Import NumPy for numerical operations and array handling.
import numpy as np

# Import Pandas for creating, reading, and manipulating datasets.
import pandas as pd

# Import Path for handling file and folder paths safely.
from pathlib import Path
# ================================================================

# ================================================================
# Original notebook cell 4
# Define a function to generate a synthetic grant proposal dataset.
def generate_us_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    # Create a NumPy random number generator using the fixed seed.
    rng = np.random.default_rng(random_state)

    # Generate the length of each grant proposal in words.
    proposal_length = rng.integers(1000, 6000, n_samples)

    # Generate the requested grant budget.
    requested_budget = rng.integers(20_000, 750_000, n_samples)

    # Generate the research methodology type for each proposal.
    methodology_type = rng.choice(
        ['quantitative', 'qualitative', 'mixed_methods', 'computational'],
        size=n_samples,
        p=[0.35, 0.15, 0.30, 0.20]
    )

    # Generate US federal grant funding agencies for each proposal.
    agency = rng.choice(
        ['NSF', 'NIH', 'DOE', 'USDA', 'EPA'],
        size=n_samples,
        p=[0.28, 0.28, 0.16, 0.16, 0.12]
    )

    # Generate the US state where the applicant institution is located.
    applicant_state = rng.choice(
        ['CA', 'NY', 'TX', 'MA', 'IL', 'FL', 'PA', 'OH', 'WA', 'NC', 'CO', 'GA'],
        size=n_samples
    )

    # Generate the strategic alignment score between 0 and 1.
    strategic_alignment_score = rng.beta(2, 2, n_samples)

    # Generate the number of researchers in each proposal team.
    team_size = rng.integers(1, 12, n_samples)

    # Generate the average research experience of the team in years.
    avg_team_experience_years = rng.gamma(
        shape=4.0,
        scale=2.5,
        size=n_samples
    ).round(1)

    # Generate the number of previous grants won by the research team.
    prior_grants_won = rng.poisson(
        lam=1.5,
        size=n_samples
    )

    # Generate the average number of publications produced by the team.
    avg_publication_count = rng.gamma(
        shape=3.0,
        scale=8.0,
        size=n_samples
    ).round(0)

    # Generate the reviewer focus matching score between 0 and 1.
    reviewer_focus_match = rng.beta(
        2.5,
        2,
        n_samples
    )

    # Calculate a latent quality and funding-fit score.
    z = (
        0.9 * strategic_alignment_score
        + 0.8 * reviewer_focus_match
        + 0.05 * np.log1p(avg_publication_count)
        + 0.25 * np.tanh(prior_grants_won / 2)
        + 0.10 * np.tanh(avg_team_experience_years / 10)
        + 0.05 * np.tanh(team_size / 6)
        - 0.15 * (requested_budget / 750_000)
        - 1.2
        + rng.normal(0, 0.35, n_samples)
    )

    # Convert the latent score into a funding probability between 0 and 1.
    funded_prob = 1 / (1 + np.exp(-4 * z))

    # Generate the binary funding outcome using the calculated probability.
    funded = rng.binomial(
        1,
        funded_prob
    )

    # Combine all generated features and the target variable into a DataFrame.
    df = pd.DataFrame({
        'proposal_length': proposal_length,
        'requested_budget': requested_budget,
        'methodology_type': methodology_type,
        'agency': agency,
        'applicant_state': applicant_state,
        'strategic_alignment_score': strategic_alignment_score.round(3),
        'team_size': team_size,
        'avg_team_experience_years': avg_team_experience_years,
        'prior_grants_won': prior_grants_won,
        'avg_publication_count': avg_publication_count,
        'reviewer_focus_match': reviewer_focus_match.round(3),
        'funded': funded,
    })

    return df
# ================================================================

# ================================================================
# Original notebook cell 5
# Generate the synthetic grant funding dataset with 2,000 records.
# random_state=42 ensures the same dataset is generated every time.
# (Handled inside load_data stage.)
# ================================================================

# ================================================================
# Original notebook cell 6
# (Printing and inspection steps – moved to generate_output stage.)
# ================================================================

# ================================================================
# Original notebook cell 7
# (Dataset saving – moved to generate_output stage.)
# ================================================================

# STAGE 0 - ARGUMENTS / CONFIGURATION
def parse_arguments():
    parser = argparse.ArgumentParser(description="Pipeline implementation")
    args, _ = parser.parse_known_args()
    return vars(args)

# STAGE 1 - INPUT
def load_data(config):
    """Generate the synthetic dataset."""
    df = generate_us_dataset(n_samples=2000, random_state=42)
    return df

# STAGE 2 - PREPROCESSING / FEATURE ENGINEERING
def preprocess_data(raw_data, config):
    """Pass through the raw DataFrame."""
    processed_data = {"df": raw_data}
    return processed_data

# STAGE 3 - INITIALIZATION / SETUP
def initialize(config, processed_data):
    """Make the DataFrame available to later stages."""
    state = {"df": processed_data["df"]}
    return state

# STAGE 4 - CORE EXECUTION
def execute(state, processed_data, config):
    """No core execution needed for this generator."""
    results = {}
    return results

# STAGE 5 - VALIDATION / EVALUATION
def evaluate(state, results, processed_data, config):
    """No evaluation needed for this generator."""
    evaluation = {}
    return evaluation

# STAGE 6 - OUTPUT / PERSISTENCE
def generate_output(state, results, evaluation, config):
    """Print diagnostics and save the synthetic dataset."""
    df = state["df"]

    # Display the number of rows and columns in the dataset.
    print(df.shape)

    # Class balance (funded):
    print('Class balance (funded):')
    print(df['funded'].value_counts(normalize=True).round(3))
    print()  # blank line

    # Missing values per column:
    print('Missing values per column:')
    print(df.isna().sum())
    print()  # blank line

    # Save dataset to CSV so downstream notebooks can load it independently
    OUTPUT_PATH = Path('Predictive-Grant-Funding-Success-Classifier-synthetic-dataset.csv')
    df.to_csv(OUTPUT_PATH, index=False)
    print(f'Dataset saved to: {OUTPUT_PATH.resolve()}')

    # No explicit output object required by the original notebook
    return {}

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
