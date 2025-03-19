#!/usr/bin/env python3
"""
Complete Biobank Survival Analysis Pipeline

This script runs the entire suite of survival analysis tools:
1. Basic Cox Proportional Hazards modeling
2. Advanced survival models (AFT, Random Survival Forest, DeepSurv)
3. Model comparisons and visualizations
"""

import os
import argparse
import subprocess
import yaml
import sys


def run_command(command, description):
    """Run a command and print its output"""
    print(f"\n{'=' * 80}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(command)}")
    print(f"{'=' * 80}\n")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    # Print output in real-time
    for line in process.stdout:
        print(line, end="")

    # Wait for the process to complete and get the return code
    process.wait()

    if process.returncode != 0:
        print(f"\nERROR: Command failed with return code {process.returncode}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete Biobank Survival Analysis pipeline"
    )
    parser.add_argument(
        "--outcome", type=str, required=True, help="Outcome column name"
    )
    parser.add_argument(
        "--time", type=str, required=True, help="Time-to-event column name"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/survival", help="Output directory"
    )
    parser.add_argument(
        "--skip-basic", action="store_true", help="Skip basic Cox PH analysis"
    )
    parser.add_argument(
        "--skip-advanced", action="store_true", help="Skip advanced models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["aft", "rsf", "deepsurv"],
        default=["aft", "rsf"],
        help="Advanced models to run",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Create config files
    basic_config = {
        "outcome": args.outcome,
        "outcome_time": args.time,
        "results_directory": args.output_dir,
    }

    with open(f"{args.output_dir}/survival_config.yaml", "w") as f:
        yaml.dump(basic_config, f)

    advanced_config = {
        "outcome": args.outcome,
        "outcome_time": args.time,
        "results_directory": args.output_dir,
        "experiment_name": "advanced_survival",
    }

    with open(f"{args.output_dir}/advanced_survival_config.yaml", "w") as f:
        yaml.dump(advanced_config, f)

    # 2. Run basic survival analysis
    if not args.skip_basic:
        success = run_command(
            [
                "python",
                "biobank_survival_main.py",
                "--config",
                f"{args.output_dir}/survival_config.yaml",
            ],
            "Basic Cox Proportional Hazards Analysis",
        )

        if not success:
            print("Basic survival analysis failed. Exiting.")
            return 1

    # 3. Run advanced survival models
    if not args.skip_advanced:
        success = run_command(
            [
                "python",
                "biobank_advanced_survival_main.py",
                "--config",
                f"{args.output_dir}/advanced_survival_config.yaml",
                "--models",
            ]
            + args.models,
            "Advanced Survival Models",
        )

        if not success:
            print("Advanced survival analysis failed. Exiting.")
            return 1

    # 4. Generate a combined report
    print("\n\nSurvival Analysis Pipeline Completed Successfully!")
    print(f"Results are available in: {args.output_dir}")

    # Provide a summary of what was run
    print("\nSummary of Analyses:")
    if not args.skip_basic:
        print("✓ Basic Cox Proportional Hazards Analysis")
    if not args.skip_advanced:
        print("✓ Advanced Survival Models:")
        for model in args.models:
            print(f"  - {model.upper()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
