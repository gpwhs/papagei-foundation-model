import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler
from biobank_experiment_constants import FULL_PYPPG_FEATURES
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from pydantic import BaseModel


class SurvivalResults(BaseModel):
    """Results from a survival analysis experiment."""

    model: str
    parameters: Dict[str, Any]
    c_index: float
    c_index_lower_ci: float
    c_index_upper_ci: float
    log_likelihood: float
    aic: float
    concordance_train: float
    concordance_test: float
    significant_features: List[str]
    p_values: Dict[str, float]
    hazard_ratios: Dict[str, float]
    hr_lower_ci: Dict[str, float]
    hr_upper_ci: Dict[str, float]
    training_time: float


class SurvivalExperimentConfig:
    """Configuration for a survival analysis experiment."""

    def __init__(
        self,
        name: str,
        description: str,
        feature_columns: List[str],
    ):
        self.name = name
        self.description = description
        self.feature_columns = feature_columns


def bootstrap_ci_c_index(
    y_true_time: np.ndarray,
    y_true_event: np.ndarray,
    y_pred_risk: np.ndarray,
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Calculate the concordance index and its confidence interval using bootstrapping.

    Args:
        y_true_time: Observed survival time
        y_true_event: Event indicator (1 if event occurred, 0 if censored)
        y_pred_risk: Predicted risk scores
        n_bootstraps: Number of bootstrap samples
        random_state: Random seed

    Returns:
        Tuple of (c-index, lower_bound, upper_bound)
    """
    np.random.seed(random_state)

    # Calculate c-index on the full dataset
    c_index = concordance_index(y_true_time, -y_pred_risk, y_true_event)

    # Bootstrap to get confidence interval
    bootstrap_indices = np.random.randint(
        0, len(y_true_time), (n_bootstraps, len(y_true_time))
    )
    bootstrap_c_indices = []

    for indices in bootstrap_indices:
        try:
            bootstrap_c_index = concordance_index(
                y_true_time[indices], -y_pred_risk[indices], y_true_event[indices]
            )
            bootstrap_c_indices.append(bootstrap_c_index)
        except Exception as e:
            print(f"Error calculating bootstrap c-index: {e}")
            # Skip if there's an error (e.g., no events in the bootstrap sample)
            continue

    # Calculate 95% confidence interval
    lower_bound = np.percentile(bootstrap_c_indices, 2.5)
    upper_bound = np.percentile(bootstrap_c_indices, 97.5)

    return c_index, lower_bound, upper_bound


def plot_kaplan_meier(
    time: pd.Series,
    event: pd.Series,
    groups: Optional[pd.Series] = None,
    group_names: Optional[List[str]] = None,
    outcome_name: str = "Outcome",
    title: str = "Kaplan-Meier Survival Curve",
    output_path: str = "km_curve.png",
) -> None:
    """
    Plot Kaplan-Meier survival curves, optionally stratified by groups.

    Args:
        time: Series of observed times
        event: Series of event indicators (1=event, 0=censored)
        groups: Optional series of group indicators for stratification
        group_names: Names of the groups for the legend
        outcome_name: Name of the outcome being analyzed
        title: Title of the plot
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()

    if groups is None:
        # Single curve for the entire dataset
        kmf.fit(time, event, label="All Patients")
        kmf.plot_survival_function(ci_show=True)
    else:
        # Create a color palette
        palette = sns.color_palette("Set1", n_colors=len(pd.unique(groups)))

        # Stratified curves
        for i, group in enumerate(sorted(pd.unique(groups))):
            mask = groups == group
            group_label = group_names[i] if group_names else f"Group {group}"
            kmf.fit(time[mask], event[mask], label=group_label)
            kmf.plot_survival_function(
                ci_show=True,
                color=palette[i],
                linewidth=2,
            )

        # Add log-rank test p-value
        if len(pd.unique(groups)) == 2:
            # Binary groups
            g1, g2 = sorted(pd.unique(groups))
            mask1, mask2 = (groups == g1), (groups == g2)

            results = logrank_test(
                time[mask1], time[mask2], event[mask1], event[mask2], alpha=0.95
            )

            plt.text(
                0.1,
                0.1,
                f"Log-rank test p-value: {results.p_value:.4f}",
                transform=plt.gca().transAxes,
                bbox=dict(facecolor="white", alpha=0.7),
            )

    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")
    plt.title(f"{title} - {outcome_name}")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_survival_curves_from_cox_model(
    model: CoxPHFitter,
    X_test: pd.DataFrame,
    feature_name: str,
    outcome_name: str,
    output_path: str,
    percentiles: List[float] = [0.25, 0.5, 0.75],
    feature_labels: Optional[List[str]] = None,
) -> None:
    """
    Plot survival curves for different values of a specific feature from a Cox model.
    The function varies the given feature while keeping all other features at their median values.

    Args:
        model: Fitted Cox Proportional Hazards model.
        X_test: DataFrame containing test features.
        feature_name: Name of the feature to vary.
        outcome_name: Name of the outcome.
        output_path: Path to save the plot.
        percentiles: Percentiles of the feature to use for plotting.
        feature_labels: Optional labels for the feature values.
    """
    plt.figure(figsize=(12, 8))

    # Calculate median values for all features from X_test
    medians = X_test.median()

    # Check if the feature exists
    if feature_name not in X_test.columns:
        print(f"Feature {feature_name} not found in the dataframe.")
        return

    # Get the feature values at the desired percentiles
    values = [X_test[feature_name].quantile(p) for p in percentiles]

    # Create labels if none provided
    if feature_labels is None:
        feature_labels = [
            f"{feature_name} = {value:.2f} ({int(p*100)}th percentile)"
            for p, value in zip(percentiles, values)
        ]

    # Choose a color palette for the curves
    palette = sns.color_palette("Set1", n_colors=len(values))

    # For each percentile value, generate and plot the survival curve
    for i, value in enumerate(values):
        # Create a new sample row with all features set to their median value
        sample = medians.copy().to_frame().T
        # Set the feature of interest to the desired percentile value
        sample[feature_name] = value

        # Predict the survival function using the Cox model
        surv_func = model.predict_survival_function(sample)

        # Plot the survival curve
        plt.step(
            surv_func.index,
            surv_func.iloc[:, 0],
            where="post",
            label=feature_labels[i],
            color=palette[i],
            linewidth=2,
        )

    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")
    plt.title(f"Survival Curves by {feature_name} - {outcome_name}")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_hazard_ratios(
    model: CoxPHFitter,
    output_path: str,
    n_features: int = 15,
    title: str = "Hazard Ratios with 95% CI",
) -> None:
    """
    Plot hazard ratios and their confidence intervals for the most significant features.

    Args:
        model: Fitted Cox Proportional Hazards model
        output_path: Path to save the plot
        n_features: Number of top features to include in the plot
        title: Title of the plot
    """
    summary = model.summary

    # Sort by p-value and select top features
    summary_sorted = summary.sort_values(by="p")
    summary_top = summary_sorted.head(n_features)

    # Create forest plot
    plt.figure(figsize=(10, n_features * 0.4 + 2))

    # Plot data
    plt.errorbar(
        summary_top["exp(coef)"],
        range(len(summary_top)),
        xerr=[
            summary_top["exp(coef)"] - summary_top["exp(coef) lower 95%"],
            summary_top["exp(coef) upper 95%"] - summary_top["exp(coef)"],
        ],
        fmt="o",
        capsize=5,
        color="navy",
        ecolor="gray",
        markersize=8,
    )

    # Add feature names
    plt.yticks(range(len(summary_top)), summary_top.index)

    # Add reference line at HR=1
    plt.axvline(x=1, color="red", linestyle="--", alpha=0.7)

    # Add p-value annotations
    for i, (_, row) in enumerate(summary_top.iterrows()):
        p_value = row["p"]
        if p_value < 0.001:
            p_text = "p < 0.001"
        else:
            p_text = f"p = {p_value:.3f}"

        plt.text(
            max(4, summary_top["exp(coef) upper 95%"].max() * 1.1),
            i,
            p_text,
            va="center",
            fontsize=9,
        )

    # Add HR values

    for i, (_, row) in enumerate(summary_top.iterrows()):
        hr = row["exp(coef)"]
        ci_lower = row["exp(coef) lower 95%"]
        ci_upper = row["exp(coef) upper 95%"]

        plt.text(
            0.2,
            i,
            f"HR: {hr:.2f} ({ci_lower:.2f}-{ci_upper:.2f})",
            va="center",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6),
        )

    plt.xlabel("Hazard Ratios (95% CI)")
    plt.xscale("log")
    plt.title(title)
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_survival_analysis_results(
    model_name: str,
    outcome: str,
    output_dir: str,
    results: SurvivalResults,
) -> None:
    """
    Save survival analysis results to files.

    Args:
        model_name: Name of the model
        outcome: Name of the outcome
        output_dir: Directory to save results
        results: SurvivalResults object containing the results
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save results to text file
    print(f"\nSaving results to {output_dir}/{model_name}_results.txt")
    with open(f"{output_dir}/{model_name}_results.txt", "w") as f:
        f.write(f"{model_name} - {outcome} Survival Analysis Results:\n")
        f.write(
            f"C-index: {results.c_index:.4f} (CI: {results.c_index_lower_ci:.4f}-{results.c_index_upper_ci:.4f})\n"
        )
        f.write(f"Log-likelihood: {results.log_likelihood:.4f}\n")
        f.write(f"AIC: {results.aic:.4f}\n")
        f.write(f"Concordance (train): {results.concordance_train:.4f}\n")
        f.write(f"Concordance (test): {results.concordance_test:.4f}\n")
        f.write(f"Training time: {results.training_time:.2f} seconds\n\n")

        f.write("Significant Features (p < 0.05):\n")
        for feature in results.significant_features:
            f.write(f"  {feature}: HR={results.hazard_ratios[feature]:.4f} ")
            f.write(
                f"(CI: {results.hr_lower_ci[feature]:.4f}-{results.hr_upper_ci[feature]:.4f}), "
            )
            f.write(f"p={results.p_values[feature]:.4f}\n")

    # Save hazard ratios to CSV
    hazard_ratios_df = pd.DataFrame(
        {
            "Feature": list(results.hazard_ratios.keys()),
            "Hazard_Ratio": list(results.hazard_ratios.values()),
            "HR_Lower_CI": list(results.hr_lower_ci.values()),
            "HR_Upper_CI": list(results.hr_upper_ci.values()),
            "P_Value": list(results.p_values.values()),
        }
    )

    hazard_ratios_df = hazard_ratios_df.sort_values("P_Value")
    hazard_ratios_df.to_csv(f"{output_dir}/{model_name}_hazard_ratios.csv", index=False)

    # Save model parameters
    with open(f"{output_dir}/{model_name}_parameters.txt", "w") as f:
        f.write("Model parameters:\n")
        for param, value in results.parameters.items():
            f.write(f"  {param}: {value}\n")


def train_and_evaluate_cox_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    time_train: pd.Series,
    time_test: pd.Series,
    event_train: pd.Series,
    event_test: pd.Series,
    model_name: str,
    outcome: str,
    output_dir: str,
    penalizer: float = 0.1,
    l1_ratio: float = 0.0,
    standardize_features: bool = True,
) -> SurvivalResults:
    """
    Train and evaluate a Cox Proportional Hazards model.

    Args:
        X_train: Training features
        X_test: Testing features
        time_train: Training survival times
        time_test: Testing survival times
        event_train: Training event indicators
        event_test: Testing event indicators
        model_name: Name of the model
        outcome: Name of the outcome
        output_dir: Directory to save results
        penalizer: Regularization strength (alpha)
        l1_ratio: L1 ratio for elastic net regularization (0=ridge, 1=lasso)
        standardize_features: Whether to standardize features

    Returns:
        SurvivalResults object containing model performance metrics
    """
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Prepare the training data with time and event info
    train_df = X_train.copy()
    train_df["time"] = time_train
    train_df["event"] = event_train

    # Prepare the test data with time and event info
    test_df = X_test.copy()
    test_df["time"] = time_test
    test_df["event"] = event_test

    # Standardize features if requested
    if standardize_features:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

        # Update the DataFrames with scaled features
        train_df = X_train_scaled.copy()
        train_df["time"] = time_train
        train_df["event"] = event_train

        test_df = X_test_scaled.copy()
        test_df["time"] = time_test
        test_df["event"] = event_test

    # Initialize and fit the Cox model
    start_time = time.time()
    cox_model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)

    try:
        cox_model.fit(train_df, duration_col="time", event_col="event")
        training_time = time.time() - start_time

        # Save the model summary
        cox_summary = cox_model.summary
        cox_summary.to_csv(f"{output_dir}/{model_name}_summary.csv")

        # Plot the hazard ratios
        plot_hazard_ratios(
            cox_model,
            output_path=f"{output_dir}/{model_name}_hazard_ratios.png",
            title=f"{model_name} - Hazard Ratios with 95% CI",
        )

        # Generate predictions
        # Note: For Cox models, predictions are risk scores where higher = higher risk
        train_predictions = cox_model.predict_partial_hazard(X_train)
        test_predictions = cox_model.predict_partial_hazard(X_test)

        # Calculate concordance index (C-index)
        c_index_train = concordance_index(time_train, -train_predictions, event_train)
        c_index_test = concordance_index(time_test, -test_predictions, event_test)

        # Calculate bootstrap confidence interval for C-index
        c_index, c_index_lower, c_index_upper = bootstrap_ci_c_index(
            time_test.values, event_test.values, test_predictions.values
        )
        print(
            f"{model_name} - C-index: {c_index:.4f} ({c_index_lower:.4f}-{c_index_upper:.4f})"
        )

        # Get model metrics
        log_likelihood = cox_model.log_likelihood_
        aic = cox_model.AIC_partial_

        # Extract hazard ratios and confidence intervals using the proper column names
        hazard_ratios = cox_summary["exp(coef)"].to_dict()
        hr_lower_ci = cox_summary["exp(coef) lower 95%"].to_dict()
        hr_upper_ci = cox_summary["exp(coef) upper 95%"].to_dict()
        p_values = cox_summary["p"].to_dict()

        # Get significant features (p < 0.05)
        significant_features = [
            feature for feature, p_value in p_values.items() if p_value < 0.05
        ]

        # Plot survival curves for important features
        if len(significant_features) > 0:
            for feature in significant_features[:3]:  # Plot top 3 significant features
                plot_survival_curves_from_cox_model(
                    cox_model,
                    X_test,
                    feature_name=feature,
                    outcome_name=outcome,
                    output_path=f"{output_dir}/{model_name}_{feature}_survival_curves.png",
                )

        # Create results object
        results = SurvivalResults(
            model="Cox Proportional Hazards",
            parameters={
                "penalizer": penalizer,
                "l1_ratio": l1_ratio,
                "standardize": standardize_features,
            },
            c_index=c_index,
            c_index_lower_ci=c_index_lower,
            c_index_upper_ci=c_index_upper,
            log_likelihood=log_likelihood,
            aic=aic,
            concordance_train=c_index_train,
            concordance_test=c_index_test,
            significant_features=significant_features,
            p_values=p_values,
            hazard_ratios=hazard_ratios,
            hr_lower_ci=hr_lower_ci,
            hr_upper_ci=hr_upper_ci,
            training_time=training_time,
        )

        # Save results to files
        save_survival_analysis_results(
            model_name=model_name,
            outcome=outcome,
            output_dir=output_dir,
            results=results,
        )

        return results

    except Exception as e:
        print(f"Error fitting Cox model: {e}")
        raise


def setup_survival_experiments(
    pyppg_features: List[str],
    embedding_columns: List[str],
) -> Dict[str, SurvivalExperimentConfig]:
    """
    Set up experiment configurations for survival analysis.

    Args:
        pyppg_features: List of pyPPG feature column names
        embedding_columns: List of embedding column names

    Returns:
        Dictionary mapping experiment keys to configurations
    """
    traditional_features = ["age", "sex", "BMI"]

    return {
        "M0": SurvivalExperimentConfig(
            name="M0_PaPaGei_Only",
            description="Only PaPaGei features",
            feature_columns=embedding_columns,
        ),
        "M1": SurvivalExperimentConfig(
            name="M1_Traditional_Only",
            description="Only metadata (age, sex, BMI)",
            feature_columns=traditional_features,
        ),
        "M2": SurvivalExperimentConfig(
            name="M2_PaPaGei_Traditional",
            description="Both PaPaGei features and metadata",
            feature_columns=embedding_columns + traditional_features,
        ),
        "M3": SurvivalExperimentConfig(
            name="M3_pyPPG_Only",
            description="pyPPG features",
            feature_columns=pyppg_features,
        ),
        "M4": SurvivalExperimentConfig(
            name="M4_pyPPG_Traditional",
            description="pyPPG features and metadata",
            feature_columns=pyppg_features + traditional_features,
        ),
    }


def run_survival_experiment(
    experiment_config: SurvivalExperimentConfig,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    time_train: pd.Series,
    time_test: pd.Series,
    event_train: pd.Series,
    event_test: pd.Series,
    outcome: str,
    output_dir: str,
    penalizer: float = 0.1,
    l1_ratio: float = 0.0,
) -> SurvivalResults:
    """
    Run a single survival analysis experiment.

    Args:
        experiment_config: Configuration for the experiment
        X_train, X_test: Feature DataFrames for train/test
        time_train, time_test: Survival times for train/test
        event_train, event_test: Event indicators for train/test
        outcome: Name of the outcome
        output_dir: Directory to save results
        penalizer: Regularization strength
        l1_ratio: L1 ratio for elastic net

    Returns:
        SurvivalResults object
    """
    print(
        f"\n--- Running Survival Experiment {experiment_config.name}: {experiment_config.description} ---"
    )

    # Select features for this experiment
    X_train_exp = X_train[experiment_config.feature_columns]
    X_test_exp = X_test[experiment_config.feature_columns]
    # need to sanitize column names for anything that makes filenames not work
    X_train_exp.columns = X_train_exp.columns.str.replace("/", "_")
    X_test_exp.columns = X_test_exp.columns.str.replace("/", "_")

    # Create experiment output directory
    exp_output_dir = f"{output_dir}/{experiment_config.name}"
    os.makedirs(exp_output_dir, exist_ok=True)

    # Train and evaluate Cox model
    results = train_and_evaluate_cox_model(
        X_train=X_train_exp,
        X_test=X_test_exp,
        time_train=time_train,
        time_test=time_test,
        event_train=event_train,
        event_test=event_test,
        model_name=experiment_config.name,
        outcome=outcome,
        output_dir=exp_output_dir,
        penalizer=penalizer,
        l1_ratio=l1_ratio,
    )

    return results


def create_survival_analysis_summary(
    results: Dict[str, SurvivalResults],
    output_dir: str,
) -> None:
    """
    Create a summary of survival analysis results across experiments.

    Args:
        results: Dictionary mapping experiment names to SurvivalResults
        output_dir: Directory to save summary
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create summary DataFrame
    summary = pd.DataFrame(
        {
            "Model": [
                "M0: PaPaGei Only",
                "M1: Traditional Factors",
                "M2: PaPaGei + Traditional",
                "M3: pyPPG Only",
                "M4: pyPPG + Traditional",
            ],
            "C-index": [
                f"{results['M0'].c_index:.4f} ({results['M0'].c_index_lower_ci:.4f}-{results['M0'].c_index_upper_ci:.4f})",
                f"{results['M1'].c_index:.4f} ({results['M1'].c_index_lower_ci:.4f}-{results['M1'].c_index_upper_ci:.4f})",
                f"{results['M2'].c_index:.4f} ({results['M2'].c_index_lower_ci:.4f}-{results['M2'].c_index_upper_ci:.4f})",
                f"{results['M3'].c_index:.4f} ({results['M3'].c_index_lower_ci:.4f}-{results['M3'].c_index_upper_ci:.4f})",
                f"{results['M4'].c_index:.4f} ({results['M4'].c_index_lower_ci:.4f}-{results['M4'].c_index_upper_ci:.4f})",
            ],
            "Log-likelihood": [
                f"{results['M0'].log_likelihood:.2f}",
                f"{results['M1'].log_likelihood:.2f}",
                f"{results['M2'].log_likelihood:.2f}",
                f"{results['M3'].log_likelihood:.2f}",
                f"{results['M4'].log_likelihood:.2f}",
            ],
            "AIC": [
                f"{results['M0'].aic:.2f}",
                f"{results['M1'].aic:.2f}",
                f"{results['M2'].aic:.2f}",
                f"{results['M3'].aic:.2f}",
                f"{results['M4'].aic:.2f}",
            ],
            "Training Time (s)": [
                f"{results['M0'].training_time:.2f}",
                f"{results['M1'].training_time:.2f}",
                f"{results['M2'].training_time:.2f}",
                f"{results['M3'].training_time:.2f}",
                f"{results['M4'].training_time:.2f}",
            ],
            "Significant Features": [
                f"{len(results['M0'].significant_features)}",
                f"{len(results['M1'].significant_features)}",
                f"{len(results['M2'].significant_features)}",
                f"{len(results['M3'].significant_features)}",
                f"{len(results['M4'].significant_features)}",
            ],
        }
    )

    # Save summary to CSV
    summary.to_csv(f"{output_dir}/survival_analysis_summary.csv", index=False)
    print("\nSurvival Analysis Summary:")
    print(summary)

    # Create bar chart comparing C-indices
    plt.figure(figsize=(12, 6))

    # Extract C-indices and confidence intervals
    models = ["M0", "M1", "M2", "M3", "M4"]
    c_indices = [results[m].c_index for m in models]
    c_index_errors = [
        [results[m].c_index - results[m].c_index_lower_ci for m in models],
        [results[m].c_index_upper_ci - results[m].c_index for m in models],
    ]

    # Plot bar chart
    bars = plt.bar(
        range(len(models)),
        c_indices,
        yerr=c_index_errors,
        capsize=5,
        color="royalblue",
    )

    # Add value labels above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Add labels and title
    plt.xlabel("Model")
    plt.ylabel("C-index (Concordance)")
    plt.title("Comparison of C-indices Across Models")
    plt.xticks(
        range(len(models)),
        [
            "M0: PaPaGei Only",
            "M1: Traditional",
            "M2: PaPaGei+Trad",
            "M3: pyPPG Only",
            "M4: pyPPG+Trad",
        ],
    )
    plt.ylim(0.5, 1.0)  # C-index ranges from 0.5 (random) to 1.0 (perfect)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{output_dir}/c_index_comparison.png")
    plt.close()

    # Create AIC comparison
    plt.figure(figsize=(12, 6))
    aic_values = [results[m].aic for m in models]
    bars = plt.bar(
        range(len(models)),
        aic_values,
        color="lightcoral",
    )

    # Add value labels above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 20,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Add labels and title
    plt.xlabel("Model")
    plt.ylabel("AIC (lower is better)")
    plt.title("Comparison of AIC Across Models")
    plt.xticks(
        range(len(models)),
        [
            "M0: PaPaGei Only",
            "M1: Traditional",
            "M2: PaPaGei+Trad",
            "M3: pyPPG Only",
            "M4: pyPPG+Trad",
        ],
    )
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{output_dir}/aic_comparison.png")
    plt.close()


def preprocess_data(
    df: pd.DataFrame,
    outcome: str,
    embedding_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess data for modeling.

    Args:
        df: Source DataFrame
        outcome: Target variable name
        embedding_df: DataFrame with pre-computed embeddings

    Returns:
        Tuple of (features_df, target_series)
    """
    # Extract traditional features
    embedding_df.to_csv("embedding_df.csv")
    traditional_features = ["age", "sex", "BMI"]
    traditional_df = df[traditional_features]
    target = df[outcome]
    if os.path.exists(f"final_pyppg_feature_columns_{outcome}.txt"):
        print(f"Loading from final_pyppg_feature_columns_{outcome}.txt")
        with open(f"final_pyppg_feature_columns_{outcome}.txt", "r") as f:
            pyppg_df_final_columns = f.read().splitlines()
        pyppg_df_final = df[pyppg_df_final_columns]
    else:
        pyppg_df = df[FULL_PYPPG_FEATURES]
        pyppg_df_reduced, dropped_corr = remove_highly_correlated_features(
            pyppg_df, target, corr_threshold=0.9
        )
        print(f"Dropped due to high correlation from pyPPG features: {dropped_corr}")
        pyppg_df_final, _ = remove_high_vif_features(
            pyppg_df_reduced, vif_threshold=5.0
        )
        print(f"Length of pyppg_df_final: {len(pyppg_df_final)}")
        # write final feature columns to file
        with open(f"final_pyppg_feature_columns_{outcome}.txt", "w") as f:
            for col in pyppg_df_final.columns:
                f.write(f"{col}\n")

    # Combine all features
    all_features = pd.concat([embedding_df, traditional_df, pyppg_df_final], axis=1)

    # Drop the outcome column if present
    if outcome in all_features.columns:
        all_features = all_features.drop(columns=[outcome])

    return all_features, target
