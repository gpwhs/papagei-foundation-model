#!/usr/bin/env python3
"""
Main script for running Biobank Survival Analysis with improved feature selection.
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
import yaml
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import utilities

# Import survival-specific modules
from biobank_experiment_constants import FULL_PYPPG_FEATURES
from biobank_survival_preprocessing import (
    prepare_survival_data,
    save_processed_survival_data,
    load_processed_survival_data,
)
from biobank_survival_analysis import (
    setup_survival_experiments,
    run_survival_experiment,
    create_survival_analysis_summary,
    plot_kaplan_meier,
)


def main():
    parser = argparse.ArgumentParser(description="Run Biobank Survival Analysis")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="configs/survival_config.yaml",
        help="Path to the survival analysis configuration file",
    )
    parser.add_argument(
        "--preprocessed-data",
        type=str,
        default=None,
        help="Path to preprocessed survival data (skip preprocessing if provided)",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing even if --preprocessed-data is not provided",
    )

    args = parser.parse_args()

    # Load configuration from yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    outcome = config["outcome"]
    outcome_time = config["outcome_time"]
    results_dir = config["results_directory"]
    penalizer = config.get("penalizer", 0.1)
    l1_ratio = config.get("l1_ratio", 0.0)
    feature_selection = config.get("feature_selection", True)

    print(f"Running survival analysis for outcome: {outcome}")
    print(f"Using time-to-event column: {outcome_time}")

    # Create directory structure
    outcome_dir = f"{results_dir}/survival_{outcome}"
    os.makedirs(outcome_dir, exist_ok=True)

    # Either load preprocessed data or preprocess from scratch
    if args.preprocessed_data and os.path.exists(args.preprocessed_data):
        print(f"Loading preprocessed data from {args.preprocessed_data}")
        data_dict = load_processed_survival_data(args.preprocessed_data)

        # Extract required components
        X_train = data_dict["X_train"]
        X_test = data_dict["X_test"]
        event_train = data_dict["event_train"]
        event_test = data_dict["event_test"]
        time_train = data_dict["time_train"]
        time_test = data_dict["time_test"]

        # Get feature names for experiment setup
        if "feature_names" in data_dict:
            feature_names = data_dict["feature_names"]
        else:
            feature_names = X_train.columns.tolist()
    elif not args.skip_preprocessing:
        # Load data from scratch
        print("Loading and preprocessing data...")
        data_path = os.getenv("BIOBANK_DATA_PATH")
        if not data_path:
            raise ValueError("BIOBANK_DATA_PATH environment variable is not set")

        df = pd.read_parquet(f"{data_path}/first_visit_survival_data_pyppg.parquet")
        # remove entries if outcome_time is below 90
        df = df[df[outcome_time] >= 90]
        df.reset_index(drop=True, inplace=True)
        # list columns that have missing values

        # Prepare data for survival analysis
        data_dict = prepare_survival_data(
            df=df,
            outcome=outcome,
            time_column=outcome_time,
            test_size=0.2,
            random_state=42,
            stratify=True,
        )

        # Save processed data for future use
        save_processed_survival_data(data_dict, f"{outcome_dir}/preprocessed_data")

        # Extract required components
        X_train = data_dict["X_train"]
        X_test = data_dict["X_test"]
        event_train = data_dict["event_train"]
        event_test = data_dict["event_test"]
        time_train = data_dict["time_train"]
        time_test = data_dict["time_test"]
        feature_names = data_dict["feature_names"]
    else:
        raise ValueError(
            "Either provide --preprocessed-data or disable --skip-preprocessing"
        )

    # Generate a Kaplan-Meier plot for the overall dataset
    print("Generating Kaplan-Meier curve for the overall dataset...")
    plot_kaplan_meier(
        time=pd.concat([time_train, time_test]),
        event=pd.concat([event_train, event_test]),
        outcome_name=outcome,
        title="Overall Kaplan-Meier Survival Curve",
        output_path=f"{outcome_dir}/overall_km_curve.png",
    )

    # Determine feature groups for experiments
    print("Setting up experiments...")

    # Extract PCA features if they exist
    pca_features = [col for col in X_train.columns if col.startswith("pca_")]
    if pca_features:
        embedding_columns = pca_features
    else:
        # Try other common embedding column naming patterns
        embedding_columns = [col for col in X_train.columns if col.startswith("emb_")]

    # If no embeddings found, use all features except traditional ones
    if not embedding_columns:
        print("No embedding features found, using all features as embeddings")
        traditional_features = ["age", "sex", "BMI"]
        embedding_columns = [
            col for col in X_train.columns if col not in traditional_features
        ]
    pyppg_columns = [col for col in X_train.columns if col in FULL_PYPPG_FEATURES]

    # Setup experiments using the appropriate feature names
    experiments = setup_survival_experiments(pyppg_columns, embedding_columns)

    # Run survival experiments
    print("Running survival analysis experiments...")
    results = {}
    model_predictions = {}

    for exp_key, exp_config in tqdm(experiments.items(), desc="Running experiments"):
        try:
            results[exp_key], model_predictions[exp_key] = run_survival_experiment(
                experiment_config=exp_config,
                X_train=X_train,
                X_test=X_test,
                time_train=time_train,
                time_test=time_test,
                event_train=event_train,
                event_test=event_test,
                outcome=outcome,
                output_dir=outcome_dir,
                penalizer=penalizer,
                l1_ratio=l1_ratio,
            )
        except Exception as e:
            print(f"Error running experiment {exp_key}: {e}")
            # Continue with other experiments even if one fails
            continue

    # Create summary of results
    if results:
        print("Creating summary of results...")
        create_survival_analysis_summary(results, outcome_dir)

        # Save the dataset split for future reference
        import joblib

        test_data = {
            "X_test": X_test,
            "time_test": time_test,
            "event_test": event_test,
        }
        joblib.dump(test_data, f"{outcome_dir}/test_data.joblib")

        print(
            f"All experiments completed successfully! Results saved to {outcome_dir}/"
        )
    else:
        print("No experiments completed successfully.")


if __name__ == "__main__":
    main()
