#!/usr/bin/env python3
"""
Biobank Survival Insights Generator

This script generates clinical insights and visualizations from survival analysis results.
"""

import os
import argparse
import joblib
import pandas as pd
from pathlib import Path

from biobank_experiment_utils import (
    load_yaml_config,
    get_embedding_df,
    preprocess_classification_data,
)
from biobank_survival_insights import (
    generate_survival_report,
    create_individual_risk_calculator,
    generate_feature_impact_visualizations,
    time_to_event_distributions,
    generate_clinical_decision_curve,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate clinical insights from survival analysis"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing survival analysis results",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to the original data (to create new visualizations)",
    )
    parser.add_argument(
        "--outcome",
        type=str,
        required=True,
        help="Outcome column name",
    )
    parser.add_argument(
        "--time",
        type=str,
        required=True,
        help="Time-to-event column name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/survival_insights",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--risk-calculator",
        action="store_true",
        help="Generate a standalone risk calculator",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=15,
        help="Number of top features to include in visualizations",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate the survival report
    report_path = generate_survival_report(
        model_dir=args.results_dir,
        output_file=f"{args.output_dir}/survival_insights_report.md",
        top_features=args.top_features,
    )

    # Generate feature impact visualizations
    generate_feature_impact_visualizations(
        model_dir=args.results_dir,
        output_dir=f"{args.output_dir}/feature_impact",
        model_type="all",
        top_n=args.top_features,
    )

    # Check if we should generate a risk calculator
    if args.risk_calculator:
        # Find a Cox model
        cox_models = list(Path(args.results_dir).glob("**/*_classifier.joblib"))

        # Prefer M2 or M4 models (combined features)
        for model_path in cox_models:
            if "M2" in str(model_path) or "M4" in str(model_path):
                model_file = model_path
                break
        else:
            # If no M2/M4, use the first available model
            if cox_models:
                model_file = cox_models[0]
            else:
                model_file = None
                print("No Cox model found for risk calculator")

        if model_file:
            # Find corresponding scaler
            scaler_file = os.path.join(os.path.dirname(model_file), "scaler.joblib")

            if os.path.exists(scaler_file):
                create_individual_risk_calculator(
                    cox_model_path=str(model_file),
                    scaler_path=scaler_file,
                    output_file=f"{args.output_dir}/risk_calculator.py",
                )
            else:
                print(f"Scaler not found: {scaler_file}")

    # If data path is provided, create additional visualizations
    if args.data_path:
        try:
            # Load the data
            df = pd.read_parquet(args.data_path)

            # Generate time-to-event distribution visualizations
            time_to_event_distributions(
                df=df,
                outcome_col=args.outcome,
                time_col=args.time,
                output_dir=f"{args.output_dir}/time_distributions",
                stratify_by=["age", "sex", "BMI"],  # Common stratification variables
            )

            # Find a test set if available
            test_files = list(Path(args.results_dir).glob("**/test_data.joblib"))

            if test_files:
                test_data = joblib.load(test_files[0])

                # Get X_test, time_test, event_test
                if isinstance(test_data, dict) and "X_test" in test_data:
                    # Generate decision curve analysis
                    generate_clinical_decision_curve(
                        model_dir=args.results_dir,
                        X_test=test_data["X_test"],
                        time_test=test_data["time_test"],
                        event_test=test_data["event_test"],
                        output_dir=f"{args.output_dir}/decision_curves",
                        time_horizon=365,  # 1-year
                    )

                    # Also create 30-day and 5-year curves
                    generate_clinical_decision_curve(
                        model_dir=args.results_dir,
                        X_test=test_data["X_test"],
                        time_test=test_data["time_test"],
                        event_test=test_data["event_test"],
                        output_dir=f"{args.output_dir}/decision_curves",
                        time_horizon=30,  # 30-day
                    )

                    generate_clinical_decision_curve(
                        model_dir=args.results_dir,
                        X_test=test_data["X_test"],
                        time_test=test_data["time_test"],
                        event_test=test_data["event_test"],
                        output_dir=f"{args.output_dir}/decision_curves",
                        time_horizon=1825,  # 5-year
                    )

        except Exception as e:
            print(f"Error processing data: {e}")

    print(
        f"All survival insights generated successfully! Results saved to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
