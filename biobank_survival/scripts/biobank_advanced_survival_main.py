#!/usr/bin/env python3
"""
Main script for running Advanced Biobank Survival Analysis models.

This script implements the advanced survival analysis models including:
- Accelerated Failure Time (AFT) models
- Random Survival Forest
- DeepSurv neural network
"""

import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from biobank_experiment_constants import FULL_PYPPG_FEATURES
from biobank_experiment_utils import load_yaml_config

# Import survival-specific modules
from biobank_survival_preprocessing import (
    prepare_survival_data,
    load_processed_survival_data,
)
from biobank_survival_analysis import (
    setup_survival_experiments,
    plot_kaplan_meier,
)
from biobank_advanced_survival import (
    train_random_survival_forest,
    SKSURV_AVAILABLE,
    DEEPSURV_AVAILABLE,
)
from biobank_survival_comparison import (
    create_model_comparison_plots,
    compare_multiple_models,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run Advanced Biobank Survival Analysis"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="../config/advanced_survival_config.yaml",
        help="Path to the advanced survival analysis configuration file",
    )
    parser.add_argument(
        "--preprocessed-data",
        type=str,
        default=None,
        help="Path to preprocessed survival data (skip preprocessing if provided)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["rsf"],
        default=["rsf"],
        help="List of advanced survival models to run",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_yaml_config(args.config)
    outcome = config["outcome"]
    outcome_time = config["outcome_time"]
    results_dir = config["results_directory"]
    experiment_name = config.get("experiment_name", "advanced_survival")

    rsf_n_estimators = config.get("rsf_n_estimators", 100)
    rsf_max_depth = config.get("rsf_max_depth", None)
    rsf_min_samples_split = config.get("rsf_min_samples_split", 10)
    rsf_min_samples_leaf = config.get("rsf_min_samples_leaf", 5)

    print(f"Running advanced survival analysis for outcome: {outcome}")
    print(f"Using time-to-event column: {outcome_time}")
    print(f"Selected models: {args.models}")

    # Create directory structure
    outcome_dir = f"{results_dir}/{experiment_name}_{outcome}"
    os.makedirs(outcome_dir, exist_ok=True)

    # Either load preprocessed data or look in the standard location
    # if args.preprocessed_data and os.path.exists(args.preprocessed_data):
    #     print(f"Loading preprocessed data from {args.preprocessed_data}")
    #     data_dict = load_processed_survival_data(args.preprocessed_data)
    # else:
    #     # Look for data in the standard location from basic survival analysis
    #     standard_data_path = f"{results_dir}/survival_{outcome}/preprocessed_data"
    #     if os.path.exists(standard_data_path):
    #         print(f"Loading preprocessed data from {standard_data_path}")
    #         data_dict = load_processed_survival_data(standard_data_path)
    #     else:
    #         # Load data from scratch if no preprocessed data is found
    #         print("No preprocessed data found. Loading and preprocessing data...")
    data_path = os.getenv("BIOBANK_DATA_PATH")
    if not data_path:
        raise ValueError("BIOBANK_DATA_PATH environment variable is not set")

    df = pd.read_parquet(f"{data_path}/first_visit_survival_data_pyppg.parquet")
    df = df[df[outcome_time] >= 90]
    df.reset_index(drop=True, inplace=True)

    # Prepare data for survival analysis
    data_dict = prepare_survival_data(
        df=df,
        outcome=outcome,
        time_column=outcome_time,
        test_size=0.2,
        random_state=42,
        stratify=True,
    )

    # Extract required components
    X_train = data_dict["X_train"]
    X_test = data_dict["X_test"]
    event_train = data_dict["event_train"]
    event_test = data_dict["event_test"]
    time_train = data_dict["time_train"]
    time_test = data_dict["time_test"]

    # Generate a Kaplan-Meier plot for the overall dataset if it doesn't exist
    km_path = f"{outcome_dir}/overall_km_curve.png"
    if not os.path.exists(km_path):
        print("Generating Kaplan-Meier curve for the overall dataset...")
        plot_kaplan_meier(
            time=pd.concat([time_train, time_test]),
            event=pd.concat([event_train, event_test]),
            outcome_name=outcome,
            title="Overall Kaplan-Meier Survival Curve",
            output_path=km_path,
        )

    # Determine feature groups for experiments
    print("Setting up experiments...")
    pyppg_features = [col for col in X_train.columns if col in FULL_PYPPG_FEATURES]

    # Extract PCA features if they exist
    pca_features = [col for col in X_train.columns if col.startswith("pca_")]
    if pca_features:
        embedding_columns = pca_features
    else:
        # Try other common embedding column naming patterns
        embedding_columns = [col for col in X_train.columns if col.startswith("emb_")]
    print(embedding_columns)

    # Setup experiments using the appropriate feature names
    experiments = setup_survival_experiments(
        pyppg_features,
        embedding_columns,
    )

    # Dictionary to store results from all models
    all_results = {model_type: {} for model_type in args.models}
    all_model_objects = {}

    # Run advanced survival experiments
    for exp_key, exp_config in tqdm(experiments.items(), desc="Running experiments"):
        print(f"\n--- Processing {exp_config.name}: {exp_config.description} ---")

        # Create experiment directory
        exp_dir = f"{outcome_dir}/{exp_config.name}"
        os.makedirs(exp_dir, exist_ok=True)

        # Select features for this experiment
        X_train_exp = X_train[exp_config.feature_columns]
        X_test_exp = X_test[exp_config.feature_columns]

        # Run each selected model
        if "rsf" in args.models and SKSURV_AVAILABLE:
            print("\nTraining Random Survival Forest model...")
            try:
                rsf_results = train_random_survival_forest(
                    X_train=X_train_exp,
                    X_test=X_test_exp,
                    time_train=time_train,
                    time_test=time_test,
                    event_train=event_train,
                    event_test=event_test,
                    model_name=f"{exp_config.name}_RSF",
                    outcome=outcome,
                    output_dir=exp_dir,
                    n_estimators=rsf_n_estimators,
                    max_depth=rsf_max_depth,
                    min_samples_split=rsf_min_samples_split,
                    min_samples_leaf=rsf_min_samples_leaf,
                )
                print(f"Random Survival Forest C-index: {rsf_results.c_index:.4f}")
                all_results["rsf"][exp_key] = rsf_results

                # Store model for comparison
                import joblib

                rsf_model_path = f"{exp_dir}/{exp_config.name}_RSF_model.joblib"
                if os.path.exists(rsf_model_path):
                    all_model_objects[f"{exp_config.name}_RSF"] = joblib.load(
                        rsf_model_path
                    )

            except Exception as e:
                print(f"Error training Random Survival Forest model: {e}")
        elif "rsf" in args.models:
            print("Skipping Random Survival Forest - scikit-survival not available")

    # Create comparison plots across models and experiment types
    print("\nCreating model comparison visualizations...")

    # First, compare between feature sets (M0-M4) for each model type
    for model_type, results in all_results.items():
        if results:  # Only if we have results for this model type
            model_output_dir = f"{outcome_dir}/comparisons/{model_type}"
            os.makedirs(model_output_dir, exist_ok=True)

            # Create C-index comparison plot
            try:
                create_model_comparison_plots(
                    results_dir=outcome_dir,
                    models=[f"{exp}_{model_type}" for exp in results.keys()],
                    metric="c_index",
                    title=f"C-index Comparison for {model_type.upper()} Models",
                    output_name=f"{model_type}_c_index_comparison.png",
                )
            except Exception as e:
                print(f"Error creating comparison plot for {model_type}: {e}")

    # Then, for the best feature set (usually M2 or M4), compare between model types
    best_exp = "M2"  # Default to M2 (PaPaGei + Traditional)
    if "M4" in experiments:  # Prefer M4 if available (pyPPG + Traditional)
        best_exp = "M4"

    # Collect models for the best experiment
    best_models = {}
    for model_type, results in all_results.items():
        if best_exp in results:
            model_name = f"{best_exp}_{model_type}"
            model_dir = f"{outcome_dir}/{best_exp}"

            # Try to find and load the model
            try:
                import joblib

                if model_type == "aft":
                    model_path = f"{model_dir}/{model_name}_aft_model.joblib"
                elif model_type == "rsf":
                    model_path = f"{model_dir}/{model_name}_rsf_model.joblib"
                elif model_type == "deepsurv":
                    # DeepSurv models are saved differently, need to reconstruct
                    import torch
                    from biobank_advanced_survival import DeepSurv

                    model_path = f"{model_dir}/{model_name}_deepsurv_model.pt"
                    if os.path.exists(model_path):
                        # Create an empty model
                        model = DeepSurv(in_features=X_test.shape[1])
                        # Load the state dict
                        model.load_state_dict(
                            torch.load(model_path, map_location=torch.device("cpu"))
                        )
                        best_models[model_type] = model

                if os.path.exists(model_path) and model_type != "deepsurv":
                    best_models[model_type] = joblib.load(model_path)
            except Exception as e:
                print(f"Error loading {model_type} model for comparison: {e}")

    # Create comprehensive comparison between model types if we have enough models
    if len(best_models) > 1:
        try:
            comparison_dir = f"{outcome_dir}/comparisons/model_types"
            os.makedirs(comparison_dir, exist_ok=True)

            compare_result = compare_multiple_models(
                models=best_models,
                X_test=X_test,
                time_test=time_test,
                event_test=event_test,
                output_dir=comparison_dir,
                title=f"Comparison of Survival Model Types for {outcome}",
            )

            # Print the results
            print("\nModel Comparison Results:")
            print(compare_result)

        except Exception as e:
            print(f"Error creating comprehensive model comparison: {e}")

    print(
        f"\nAll advanced survival analysis experiments completed! Results saved to {outcome_dir}/"
    )


if __name__ == "__main__":
    main()
