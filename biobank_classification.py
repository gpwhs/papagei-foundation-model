import os
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from biobank_experiment_utils import (
    load_yaml_config,
    get_embedding_df,
    preprocess_data,
    check_for_imbalance,
)
from biobank_reporting_utils import plot_calibration_curves, create_summary
from biobank_classification_functions import setup_experiments, run_experiment


def main():
    parser = argparse.ArgumentParser(description="Load Configuration from YAML file")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="xgb_config_ht.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_yaml_config(args.config)
    model = config["model"]
    outcome = config["outcome"]
    results_dir = config["results_directory"]
    handle_imbalance = config["handle_imbalance"]
    apply_pca = config["apply_pca"]
    print(f"Running experiments with model {model} for outcome {outcome}")

    # Create a nested directory structure,taking into account PCA flags
    if apply_pca:
        outcome_dir = f"{results_dir}/{outcome}_PCA"
        model_dir = f"{outcome_dir}/{model}_PCA"
    else:
        outcome_dir = f"{results_dir}/{outcome}"
        model_dir = f"{outcome_dir}/{model}"
    os.makedirs(model_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    data_path = os.getenv("BIOBANK_DATA_PATH")
    if not data_path:
        raise ValueError("BIOBANK_DATA_PATH environment variable is not set")

    df = pd.read_parquet(
        f"{data_path}/250k_waves_conditions_pyppg_first_cleaned.parquet"
    )
    if handle_imbalance:
        print("Class weighting enabled for handling imbalanced data")
    else:
        check_for_imbalance(df[outcome], outcome)
    embedding_df = get_embedding_df(df, outcome, apply_pca)

    all_features, target = preprocess_data(df, outcome, embedding_df)

    # Create train/test splits
    X_train, X_test, y_train, y_test = train_test_split(
        all_features, target, test_size=0.2, random_state=42, stratify=target
    )

    # Setup experiments using the PCA-transformed embedding feature names
    embedding_columns = [col for col in embedding_df.columns if col != outcome]

    experiments = setup_experiments(embedding_columns, outcome)

    # Run experiments
    results = {}
    model_predictions = {}

    for exp_key, exp_config in tqdm(experiments.items()):
        results[exp_key], model_predictions[exp_key] = run_experiment(
            experiment_config=exp_config,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model_type=model,
            outcome=outcome,
            output_dir=outcome_dir,  # Pass outcome directory as the output_dir
            handle_imbalance=handle_imbalance,  # Pass the class weighting flag
        )
    # Save results
    with open(f"{model_dir}/experiment_results.json", "w") as f:
        import json

        # Use model_dump() instead of dict() for Pydantic v2 compatibility
        try:
            # For Pydantic v2
            f.write(
                json.dumps({k: v.model_dump() for k, v in results.items()}, indent=2)
            )
        except AttributeError:
            # Fallback for Pydantic v1
            f.write(json.dumps({k: v.dict() for k, v in results.items()}, indent=2))

    # Create summary of results
    create_summary(results, outcome_dir, model)  # Pass outcome directory as results_dir

    # Generate calibration curves and metrics
    calibration_metrics = plot_calibration_curves(
        y_test=y_test,
        model_predictions=model_predictions,
        model_dir=model_dir,
    )

    print(f"All experiments completed successfully! Results saved to {model_dir}/")

    # Print calibration metrics
    print("\nCalibration Metrics (Brier Scores, lower is better):")
    for model_key, brier_score in calibration_metrics.items():
        print(f"{model_key}: {brier_score:.4f}")


if __name__ == "__main__":
    main()
