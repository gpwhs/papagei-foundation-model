#!/usr/bin/env python3
"""
Script for running survival-specific feature selection and analyzing results.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from biobank_survival_feature_functions import (
    get_univariate_cox_pvalue,
    check_proportional_hazards_assumption,
    select_features_for_survival,
)


def plot_feature_significance(
    df: pd.DataFrame,
    time_series: pd.Series,
    event_series: pd.Series,
    output_dir: str,
    max_features: int = 50,
):
    """
    Generate plots showing feature significance based on univariate Cox regression.

    Args:
        df: Feature DataFrame
        time_series: Time-to-event Series
        event_series: Event indicator Series
        output_dir: Directory to save plots
        max_features: Maximum number of features to include in plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Calculate p-values for all features
    print("Calculating univariate Cox p-values for all features...")
    p_values = []

    for col in df.columns:
        p = get_univariate_cox_pvalue(df[col], time_series, event_series)
        p_values.append({"feature": col, "p_value": p, "significant": p < 0.05})

    # Create DataFrame of p-values
    p_df = pd.DataFrame(p_values)

    # Sort by p-value
    p_df = p_df.sort_values("p_value")

    # Save the full p-value table
    p_df.to_csv(f"{output_dir}/univariate_cox_pvalues.csv", index=False)

    # Create a bar plot of p-values for top features
    plt.figure(figsize=(12, max(8, min(0.2 * len(p_df), 20))))

    # Get top features
    top_df = p_df.head(max_features)

    # Plot bars with color indicating significance
    bars = plt.barh(
        range(len(top_df)),
        -np.log10(top_df["p_value"]),
        color=top_df["significant"].map({True: "darkblue", False: "lightblue"}),
    )

    # Add feature names
    plt.yticks(range(len(top_df)), top_df["feature"])

    # Add labels and title
    plt.xlabel("-log10(p-value)")
    plt.title("Feature Significance in Univariate Cox Regression")

    # Add significance threshold line at p=0.05
    plt.axvline(-np.log10(0.05), color="red", linestyle="--", label="p=0.05")

    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{output_dir}/top_features_significance.png", dpi=120)
    plt.close()

    # Create a histogram of p-values
    plt.figure(figsize=(10, 6))
    plt.hist(p_df["p_value"], bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("p-value")
    plt.ylabel("Frequency")
    plt.title("Distribution of p-values from Univariate Cox Regression")
    plt.axvline(0.05, color="red", linestyle="--", label="p=0.05")
    plt.grid(linestyle="--", alpha=0.3)
    plt.legend()

    # Save the plot
    plt.savefig(f"{output_dir}/pvalue_distribution.png")
    plt.close()

    # Create a pie chart of significant vs. non-significant features
    sig_count = sum(p_df["significant"])
    nonsig_count = len(p_df) - sig_count

    plt.figure(figsize=(8, 8))
    plt.pie(
        [sig_count, nonsig_count],
        labels=[
            f"Significant (p<0.05): {sig_count}",
            f"Non-significant: {nonsig_count}",
        ],
        colors=["darkblue", "lightblue"],
        autopct="%1.1f%%",
        startangle=90,
        explode=(0.1, 0),
    )
    plt.title("Proportion of Significant Features in Univariate Cox Regression")

    # Save the plot
    plt.savefig(f"{output_dir}/significant_features_proportion.png")
    plt.close()

    return p_df


def plot_correlation_analysis(
    df: pd.DataFrame,
    output_dir: str,
    corr_threshold: float = 0.8,
):
    """
    Generate correlation analysis plots.

    Args:
        df: Feature DataFrame
        output_dir: Directory to save plots
        corr_threshold: Correlation threshold to highlight
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Calculate correlation matrix
    print("Calculating correlation matrix...")
    corr = df.corr().abs()

    # Save the correlation matrix
    corr.to_csv(f"{output_dir}/correlation_matrix.csv")

    # If there are too many features, select a subset for visualization
    max_features = 50
    if corr.shape[0] > max_features:
        # Get the features with highest correlations
        upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        most_correlated = []

        for col in upper_tri.columns:
            max_corr = upper_tri[col].max()
            if not pd.isna(max_corr) and max_corr > corr_threshold:
                most_correlated.append(col)

                # Find the feature it's most correlated with
                corr_feature = upper_tri[col].idxmax()
                if corr_feature not in most_correlated:
                    most_correlated.append(corr_feature)

        # If we still have too many, take the top by largest correlation
        if len(most_correlated) > max_features:
            most_correlated = most_correlated[:max_features]

        # Create a smaller correlation matrix for visualization
        vis_corr = corr.loc[most_correlated, most_correlated]
    else:
        vis_corr = corr

    # Create a heatmap
    plt.figure(
        figsize=(max(10, vis_corr.shape[0] * 0.3), max(8, vis_corr.shape[0] * 0.3))
    )
    mask = np.triu(np.ones_like(vis_corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        vis_corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=0,
        center=0.5,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        annot=vis_corr.shape[0] < 20,  # Only show annotations if fewer than 20 features
    )

    plt.title("Feature Correlation Matrix")
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=120)
    plt.close()

    # Create a histogram of correlation values
    plt.figure(figsize=(10, 6))

    # Get the upper triangle values
    upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    corr_values = upper_tri.values.flatten()
    corr_values = corr_values[~np.isnan(corr_values)]

    plt.hist(corr_values, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Absolute Correlation")
    plt.ylabel("Frequency")
    plt.title("Distribution of Feature Correlations")
    plt.axvline(
        corr_threshold,
        color="red",
        linestyle="--",
        label=f"Threshold: {corr_threshold}",
    )
    plt.grid(linestyle="--", alpha=0.3)
    plt.legend()

    # Save the plot
    plt.savefig(f"{output_dir}/correlation_distribution.png")
    plt.close()

    # Count highly correlated pairs
    high_corr_count = sum(corr_values > corr_threshold)
    total_pairs = len(corr_values)

    print(
        f"Number of highly correlated feature pairs (|r| > {corr_threshold}): {high_corr_count}/{total_pairs} ({high_corr_count/total_pairs*100:.1f}%)"
    )

    return vis_corr


def plot_ph_assumption_results(
    df: pd.DataFrame,
    time_series: pd.Series,
    event_series: pd.Series,
    output_dir: str,
    alpha: float = 0.05,
):
    """
    Generate plots for proportional hazards assumption test results.

    Args:
        df: Feature DataFrame
        time_series: Time-to-event Series
        event_series: Event indicator Series
        output_dir: Directory to save plots
        alpha: Significance level for PH assumption test
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check proportional hazards assumption
    print("Checking proportional hazards assumption...")
    ph_results = check_proportional_hazards_assumption(
        df, time_series, event_series, alpha
    )

    # Save results to CSV
    ph_results.to_csv(f"{output_dir}/ph_assumption_test.csv", index=False)

    # Count violations
    violations = ph_results[ph_results["violates_assumption"]]
    n_violations = len(violations)

    print(
        f"Number of features violating PH assumption: {n_violations}/{len(ph_results)} ({n_violations/len(ph_results)*100:.1f}%)"
    )

    # Create a bar plot of -log10(p-values) for features violating the assumption
    if not violations.empty:
        plt.figure(figsize=(10, max(6, 0.3 * len(violations))))

        # Sort by p-value
        violations = violations.sort_values("p_value")

        # Plot bars
        plt.barh(
            range(len(violations)), -np.log10(violations["p_value"]), color="crimson"
        )

        # Add feature names
        plt.yticks(range(len(violations)), violations["feature"])

        # Add labels and title
        plt.xlabel("-log10(p-value)")
        plt.title("Features Violating Proportional Hazards Assumption")

        # Add significance threshold line
        plt.axvline(-np.log10(alpha), color="black", linestyle="--", label=f"p={alpha}")

        plt.grid(axis="x", linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Save the plot
        plt.savefig(f"{output_dir}/ph_violations.png", dpi=120)
        plt.close()

    # Create a pie chart of violations
    plt.figure(figsize=(8, 8))
    plt.pie(
        [n_violations, len(ph_results) - n_violations],
        labels=[
            f"Violations: {n_violations}",
            f"No violations: {len(ph_results) - n_violations}",
        ],
        colors=["crimson", "lightgreen"],
        autopct="%1.1f%%",
        startangle=90,
        explode=(0.1, 0),
    )
    plt.title("Proportion of Features Violating Proportional Hazards Assumption")

    # Save the plot
    plt.savefig(f"{output_dir}/ph_violations_proportion.png")
    plt.close()

    return ph_results


def main():
    parser = argparse.ArgumentParser(
        description="Run survival feature selection analysis"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to the data file (parquet or csv)"
    )
    parser.add_argument(
        "--outcome",
        type=str,
        required=True,
        help="Column name for the event indicator (1=event, 0=censored)",
    )
    parser.add_argument(
        "--time", type=str, required=True, help="Column name for the time-to-event"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/feature_selection",
        help="Directory to save results",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.8,
        help="Correlation threshold for feature selection",
    )
    parser.add_argument(
        "--vif-threshold",
        type=float,
        default=5.0,
        help="VIF threshold for feature selection",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Path to pre-computed embeddings file (numpy array)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data}")
    if args.data.endswith(".parquet"):
        df = pd.read_parquet(args.data)
    elif args.data.endswith(".csv"):
        df = pd.read_csv(args.data)
    else:
        raise ValueError("Data file must be parquet or csv")

    # Check if required columns exist
    if args.outcome not in df.columns:
        raise ValueError(f"Outcome column '{args.outcome}' not found in dataset")
    if args.time not in df.columns:
        raise ValueError(f"Time column '{args.time}' not found in dataset")

    # Check for missing values
    missing_outcome = df[args.outcome].isna().sum()
    missing_time = df[args.time].isna().sum()

    if missing_outcome > 0 or missing_time > 0:
        print(
            f"Warning: Found {missing_outcome} missing values in outcome and {missing_time} missing values in time"
        )
        print("Removing rows with missing values")
        df = df.dropna(subset=[args.outcome, args.time])

    # Extract time and event data
    time_series = df[args.time]
    event_series = df[args.outcome].astype(int)

    # Get the full feature set to analyze
    print("Preparing features for analysis...")

    # Define features to exclude
    exclude_cols = [args.outcome, args.time, "subject_id", "eid"]

    # Get features as all columns except excluded ones
    features = df.drop(columns=[col for col in exclude_cols if col in df.columns])

    # Summary of data
    print(f"\nData summary:")
    print(f"Total samples: {len(df)}")
    print(f"Event rate: {event_series.mean()*100:.1f}%")
    print(f"Total features: {features.shape[1]}")
    print(
        f"Time range: {time_series.min()} to {time_series.max()} (median: {time_series.median()})"
    )

    # Run univariate Cox regression analysis
    print("\nRunning univariate Cox regression analysis...")
    p_df = plot_feature_significance(
        features, time_series, event_series, f"{args.output_dir}/univariate"
    )

    # Run correlation analysis
    print("\nRunning correlation analysis...")
    corr_df = plot_correlation_analysis(
        features, f"{args.output_dir}/correlation", args.corr_threshold
    )

    # Check proportional hazards assumption
    print("\nChecking proportional hazards assumption...")
    ph_df = plot_ph_assumption_results(
        features, time_series, event_series, f"{args.output_dir}/ph_assumption"
    )

    # Run full feature selection
    print("\nRunning full feature selection process...")
    selected_features, selection_info = select_features_for_survival(
        features,
        time_series,
        event_series,
        corr_threshold=args.corr_threshold,
        vif_threshold=args.vif_threshold,
    )

    # Save selected features
    selected_features.to_csv(f"{args.output_dir}/selected_features.csv")

    # Save selection info
    with open(f"{args.output_dir}/selection_summary.txt", "w") as f:
        f.write("Feature Selection Summary\n")
        f.write("=======================\n\n")
        f.write(f"Original features: {selection_info['n_features_original']}\n")
        f.write(f"Selected features: {selection_info['n_features_selected']}\n\n")

        f.write(f"Correlation filtering (threshold: {args.corr_threshold}):\n")
        f.write(f"  Removed {len(selection_info['dropped_correlated'])} features\n\n")

        f.write(f"VIF filtering (threshold: {args.vif_threshold}):\n")
        f.write(f"  Removed {len(selection_info['dropped_vif'])} features\n\n")

        f.write("Dropped due to correlation:\n")
        for feature in selection_info["dropped_correlated"]:
            f.write(f"  - {feature}\n")

        f.write("\nDropped due to high VIF:\n")
        for feature in selection_info["dropped_vif"]:
            f.write(f"  - {feature}\n")

        f.write("\nViolates proportional hazards assumption:\n")
        for feature in selection_info["ph_assumption_violations"]:
            f.write(f"  - {feature}\n")

    print("\nFeature selection complete!")
    print(f"Original features: {selection_info['n_features_original']}")
    print(f"Selected features: {selection_info['n_features_selected']}")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
