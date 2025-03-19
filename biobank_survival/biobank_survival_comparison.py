import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from lifelines.statistics import logrank_test
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold


def create_model_comparison_plots(
    results_dir: str,
    models: List[str],
    metric: str = "c_index",
    title: str = "Model Comparison",
    output_name: str = "model_comparison.png",
):
    """
    Create comparison plots for different survival models.

    Args:
        results_dir: Directory containing results
        models: List of model directories to compare
        metric: Metric to compare ('c_index', 'aic', etc.)
        title: Title for the plot
        output_name: Filename for the output plot
    """
    # Collect metrics from each model
    metrics = []
    model_names = []
    ci_lower = []
    ci_upper = []

    for model in models:
        # Read the results.txt file
        results_file = f"{results_dir}/{model}/{model}_results.txt"
        if not os.path.exists(results_file):
            print(f"Warning: Results file not found for model {model}")
            continue

        with open(results_file, "r") as f:
            lines = f.readlines()

        # Extract the metric of interest
        if metric == "c_index":
            # Find line with C-index
            for line in lines:
                if "C-index:" in line:
                    # Extract value and CI
                    parts = line.split("C-index:")[1].strip().split()
                    c_index = float(parts[0])
                    ci = parts[1].strip("()").split("-")
                    metrics.append(c_index)
                    ci_lower.append(float(ci[0]))
                    ci_upper.append(float(ci[1]))
                    break
        else:
            # Implementation for other metrics...
            raise NotImplementedError(f"Metric {metric} not implemented for comparison")

        model_names.append(model)

    # Create the comparison plot
    plt.figure(figsize=(12, 6))

    # Convert to numpy arrays for easier manipulation
    metrics = np.array(metrics)
    ci_lower = np.array(ci_lower)
    ci_upper = np.array(ci_upper)

    # Calculate error bars
    yerr = np.vstack([metrics - ci_lower, ci_upper - metrics])

    # Create bar chart
    bars = plt.bar(
        range(len(model_names)),
        metrics,
        yerr=yerr,
        capsize=5,
        color="steelblue",
    )

    # Add value labels
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

    # Set labels and title
    plt.xlabel("Model")
    plt.ylabel(metric.upper())
    plt.title(title)
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{results_dir}/{output_name}")
    plt.close()


def calculate_integrated_brier_score(
    time_test: pd.Series,
    event_test: pd.Series,
    survival_func_pred: pd.DataFrame,
    times: Optional[np.ndarray] = None,
    max_time: Optional[float] = None,
) -> float:
    """
    Calculate the Integrated Brier Score (IBS) for a survival model.

    The Brier score at time t measures the mean squared error between
    the predicted survival probability at time t and the true status.

    Args:
        time_test: True event times
        event_test: True event indicators (1=event, 0=censored)
        survival_func_pred: Predicted survival functions (DataFrame with samples as columns)
        times: Array of time points to evaluate, if None, uses survival_func_pred index
        max_time: Maximum time point to include (recommended to truncate where censoring gets heavy)

    Returns:
        Integrated Brier Score
    """
    from sksurv.metrics import integrated_brier_score
    from sksurv.util import Surv

    # Convert to array format required by scikit-survival
    y_test = Surv.from_arrays(event_test.astype(bool), time_test)

    # Get time points
    if times is None:
        times = survival_func_pred.index.values

    # Truncate at max_time if specified
    if max_time is not None:
        times = times[times <= max_time]

    # Calculate IBS
    ibs = integrated_brier_score(y_test, survival_func_pred.loc[times].values.T, times)

    return ibs


def perform_cross_validation(
    X: pd.DataFrame,
    time: pd.Series,
    event: pd.Series,
    model_func: callable,
    model_params: Dict[str, Any],
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Perform cross-validation for a survival model.

    Args:
        X: Feature DataFrame
        time: Time column
        event: Event indicator column
        model_func: Function to train the model
        model_params: Parameters for the model function
        n_splits: Number of CV splits
        random_state: Random seed

    Returns:
        Dictionary of cross-validation results
    """
    # Create CV splitter that preserves the event distribution
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Store results
    cv_results = {
        "c_index": [],
        "training_time": [],
    }

    # Perform CV
    for i, (train_idx, test_idx) in enumerate(cv.split(X, event)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        time_train, time_test = time.iloc[train_idx], time.iloc[test_idx]
        event_train, event_test = event.iloc[train_idx], event.iloc[test_idx]

        print(f"\nCV Fold {i+1}/{n_splits}")

        # Train model
        results = model_func(
            X_train=X_train,
            X_test=X_test,
            time_train=time_train,
            time_test=time_test,
            event_train=event_train,
            event_test=event_test,
            model_name=f"CV_fold_{i+1}",
            **model_params,
        )

        # Store results
        cv_results["c_index"].append(results.c_index)
        cv_results["training_time"].append(results.training_time)

        print(f"Fold {i+1} C-index: {results.c_index:.4f}")

    # Calculate summary statistics
    for metric in cv_results:
        values = cv_results[metric]
        cv_results[f"{metric}_mean"] = np.mean(values)
        cv_results[f"{metric}_std"] = np.std(values)
        cv_results[f"{metric}_median"] = np.median(values)
        cv_results[f"{metric}_min"] = np.min(values)
        cv_results[f"{metric}_max"] = np.max(values)

    return cv_results


def create_survival_curve_heatmap(
    survival_funcs: pd.DataFrame,
    title: str = "Population Survival Curves",
    output_path: str = "survival_heatmap.png",
):
    """
    Create a heatmap visualization of multiple survival curves.

    Args:
        survival_funcs: DataFrame with survival probabilities
                       (times as index, samples as columns)
        title: Title for the plot
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))

    # Create the heatmap
    sns.heatmap(
        survival_funcs.T,  # Transpose to have times on x-axis
        cmap="viridis",
        cbar_kws={"label": "Survival Probability"},
    )

    # Get x-axis tick positions (time points)
    times = survival_funcs.index.values
    n_ticks = min(10, len(times))  # Limit to 10 ticks for readability
    tick_positions = np.linspace(0, len(times) - 1, n_ticks, dtype=int)
    tick_labels = [f"{times[pos]:.0f}" for pos in tick_positions]

    # Set labels and title
    plt.xlabel("Time (days)")
    plt.ylabel("Patients")
    plt.title(title)
    plt.xticks(tick_positions, tick_labels)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def compare_survival_probabilities_by_feature(
    X: pd.DataFrame,
    time: pd.Series,
    event: pd.Series,
    feature: str,
    output_dir: str,
    n_quantiles: int = 4,
    max_time: Optional[float] = None,
):
    """
    Compare survival probabilities stratified by a feature using Kaplan-Meier curves.

    Args:
        X: Feature DataFrame
        time: Time column
        event: Event indicator column
        feature: Feature to stratify by
        output_dir: Directory to save output
        n_quantiles: Number of quantiles to divide the feature into
        max_time: Maximum time to plot
    """
    from lifelines import KaplanMeierFitter

    if feature not in X.columns:
        print(f"Feature '{feature}' not found in dataset")
        return

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create quantiles for the feature
    if n_quantiles <= 1:
        print("n_quantiles must be at least 2")
        return

    # For binary features, use the binary values
    if X[feature].nunique() <= 2:
        groups = X[feature]
        group_names = [f"{feature}={val}" for val in sorted(X[feature].unique())]
    else:
        # For continuous features, create quantiles
        quantiles = np.linspace(0, 1, n_quantiles + 1)[1:-1]  # excluding 0 and 1
        cut_points = (
            [X[feature].min()]
            + [X[feature].quantile(q) for q in quantiles]
            + [X[feature].max()]
        )

        # Create groups based on quantiles
        groups = pd.cut(
            X[feature],
            bins=cut_points,
            labels=[f"Q{i+1}" for i in range(n_quantiles)],
            include_lowest=True,
        )

        # Create group names with the range of values
        group_names = []
        for i in range(n_quantiles):
            lower = cut_points[i]
            upper = cut_points[i + 1]
            group_names.append(f"{feature} [{lower:.2f}-{upper:.2f}]")

    # Create the KM plot
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()

    # Create a color palette
    palette = sns.color_palette("Set1", n_colors=n_quantiles)

    # Plot each group
    p_values = []
    for i, group_value in enumerate(sorted(groups.unique())):
        mask = groups == group_value
        if mask.sum() > 0:  # Ensure we have samples
            kmf.fit(
                time[mask],
                event[mask],
                label=(
                    group_names[i] if i < len(group_names) else f"Group {group_value}"
                ),
            )
            kmf.plot_survival_function(
                ci_show=True,
                color=palette[i % len(palette)],
                linewidth=2,
            )

            # If there's a previous group, calculate log-rank p-value
            if i > 0:
                prev_mask = groups == sorted(groups.unique())[i - 1]
                try:
                    results = logrank_test(
                        time[mask],
                        time[prev_mask],
                        event[mask],
                        event[prev_mask],
                        alpha=0.95,
                    )
                    p_values.append(results.p_value)
                except:
                    p_values.append(np.nan)

    # Add overall p-value if more than 2 groups
    if len(p_values) > 0:
        median_p = np.nanmedian(p_values)
        plt.text(
            0.1,
            0.1,
            f"Median log-rank p-value: {median_p:.4f}",
            transform=plt.gca().transAxes,
            bbox=dict(facecolor="white", alpha=0.7),
        )

    # Set labels and title
    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")
    plt.title(f"Kaplan-Meier Curves Stratified by {feature}")

    # Limit x-axis if max_time specified
    if max_time is not None:
        plt.xlim(0, max_time)

    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{output_dir}/{feature}_km_curves.png")
    plt.close()

    # Also create a risk table
    plt.figure(figsize=(10, 6))

    # Set up the time points for the risk table
    if max_time is None:
        max_time = time.max()

    time_points = np.linspace(0, max_time, 10)

    # Create a table of at-risk counts
    at_risk_counts = np.zeros((len(sorted(groups.unique())), len(time_points)))

    for i, group_value in enumerate(sorted(groups.unique())):
        mask = groups == group_value
        group_times = time[mask]

        for j, t in enumerate(time_points):
            at_risk_counts[i, j] = (group_times >= t).sum()

    # Create the risk table plot
    plt.imshow(
        at_risk_counts,
        aspect="auto",
        cmap="YlGnBu",
        extent=[0, max_time, -0.5, len(sorted(groups.unique())) - 0.5],
    )

    # Add count labels to the cells
    for i in range(at_risk_counts.shape[0]):
        for j in range(at_risk_counts.shape[1]):
            plt.text(
                time_points[j],
                i,
                f"{int(at_risk_counts[i, j])}",
                ha="center",
                va="center",
                color=(
                    "black"
                    if at_risk_counts[i, j] < at_risk_counts.max() / 2
                    else "white"
                ),
            )

    # Set labels and title
    plt.xlabel("Time (days)")
    plt.ylabel("Group")
    plt.title(f"Number at Risk by {feature}")
    plt.yticks(
        range(len(sorted(groups.unique()))),
        [
            group_names[i] if i < len(group_names) else f"Group {val}"
            for i, val in enumerate(sorted(groups.unique()))
        ],
    )

    # Add time points
    plt.xticks(time_points)

    plt.colorbar(label="Number at Risk")
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{output_dir}/{feature}_risk_table.png")
    plt.close()


def compare_multiple_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    time_test: pd.Series,
    event_test: pd.Series,
    output_dir: str,
    title: str = "Model Comparison",
):
    """
    Compare multiple survival models on the same test data.

    Args:
        models: Dictionary mapping model names to model objects
        X_test: Test features
        time_test: Test survival times
        event_test: Test event indicators
        output_dir: Output directory
        title: Plot title
    """
    from lifelines.utils import concordance_index
    from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc
    from sksurv.util import Surv

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Prepare result storage
    results = {
        "model": [],
        "c_index": [],
        "ibs": [],  # Integrated Brier Score
        "time_auc": [],  # Time-dependent AUC
    }

    # Convert to sksurv format for some metrics
    y_test = Surv.from_arrays(event_test.astype(bool), time_test)

    # Choose time points for evaluation
    max_time = time_test.quantile(
        0.75
    )  # Use 75th percentile to avoid sparse data at the tail
    time_points = np.linspace(0, max_time, 10)

    # Evaluate each model
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")

        try:
            # Get predictions
            if hasattr(model, "predict_risk"):
                # For scikit-survival models
                risk_scores = model.predict_risk(X_test)
            elif hasattr(model, "predict_partial_hazard"):
                # For Cox models (lifelines)
                risk_scores = model.predict_partial_hazard(X_test).values
            elif hasattr(model, "predict"):
                # For custom models (e.g. DeepSurv)
                risk_scores = model.predict(X_test).numpy()
            else:
                print(f"Cannot get predictions from {model_name}")
                continue

            # Concordance index
            c_index = concordance_index(
                time_test.values, risk_scores, event_test.values
            )

            # Get survival functions if available
            if hasattr(model, "predict_survival_function"):
                # For lifelines models
                surv_funcs = model.predict_survival_function(X_test)

                # Calculate Integrated Brier Score
                try:
                    ibs = integrated_brier_score(
                        y_test, surv_funcs.transpose().values, time_points
                    )
                except:
                    ibs = np.nan

                # Calculate time-dependent AUC
                try:
                    # For each time point, calculate AUC
                    auc_scores = []
                    for t in time_points:
                        # Get survival probability at time t
                        surv_probs = surv_funcs.loc[[t]].values.squeeze()

                        # Calculate ROC AUC
                        auc, _ = cumulative_dynamic_auc(
                            y_test, 1 - surv_probs, times=[t]  # Convert to risk
                        )
                        auc_scores.append(auc[0])

                    time_auc = np.mean(auc_scores)
                except:
                    time_auc = np.nan
            else:
                ibs = np.nan
                time_auc = np.nan

            # Store results
            results["model"].append(model_name)
            results["c_index"].append(c_index)
            results["ibs"].append(ibs)
            results["time_auc"].append(time_auc)

            print(f"  C-index: {c_index:.4f}")
            if not np.isnan(ibs):
                print(f"  Integrated Brier Score: {ibs:.4f}")
            if not np.isnan(time_auc):
                print(f"  Time-dependent AUC: {time_auc:.4f}")

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")

    # Create a results DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv(f"{output_dir}/model_comparison_metrics.csv", index=False)

    # Create comparison plots
    plt.figure(figsize=(10, 6))

    # C-index comparison
    plt.subplot(1, 2, 1)
    bar_positions = np.arange(len(results["model"]))
    plt.bar(bar_positions, results["c_index"], color="steelblue")
    plt.xlabel("Model")
    plt.ylabel("C-index")
    plt.title("C-index Comparison")
    plt.xticks(bar_positions, results["model"], rotation=45, ha="right")
    plt.ylim(0.5, 1.0)  # C-index ranges from 0.5 to 1.0

    # IBS comparison (if available)
    if not all(np.isnan(results["ibs"])):
        plt.subplot(1, 2, 2)
        valid_mask = ~np.isnan(results["ibs"])
        plt.bar(
            np.arange(sum(valid_mask)),
            [results["ibs"][i] for i in range(len(results["ibs"])) if valid_mask[i]],
            color="lightcoral",
        )
        plt.xlabel("Model")
        plt.ylabel("Integrated Brier Score")
        plt.title("IBS Comparison (lower is better)")
        plt.xticks(
            np.arange(sum(valid_mask)),
            [
                results["model"][i]
                for i in range(len(results["model"]))
                if valid_mask[i]
            ],
            rotation=45,
            ha="right",
        )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png")
    plt.close()

    # Time-dependent AUC plot (if available)
    if not all(np.isnan(results["time_auc"])):
        plt.figure(figsize=(8, 6))
        valid_mask = ~np.isnan(results["time_auc"])
        plt.bar(
            np.arange(sum(valid_mask)),
            [
                results["time_auc"][i]
                for i in range(len(results["time_auc"]))
                if valid_mask[i]
            ],
            color="mediumseagreen",
        )
        plt.xlabel("Model")
        plt.ylabel("Time-dependent AUC")
        plt.title("Time-dependent AUC Comparison")
        plt.xticks(
            np.arange(sum(valid_mask)),
            [
                results["model"][i]
                for i in range(len(results["model"]))
                if valid_mask[i]
            ],
            rotation=45,
            ha="right",
        )
        plt.ylim(0.5, 1.0)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/time_dependent_auc_comparison.png")
        plt.close()

    return results_df


def apply_permutation_importance(
    model,
    X_test: pd.DataFrame,
    time_test: pd.Series,
    event_test: pd.Series,
    metric: str = "c_index",
    n_repeats: int = 10,
    random_state: int = 42,
    output_dir: str = None,
    output_name: str = "permutation_importance",
):
    """
    Calculate permutation importance for a survival model.

    Args:
        model: Trained survival model
        X_test: Test features
        time_test: Test survival times
        event_test: Test event indicators
        metric: Metric to use ('c_index', 'brier_score')
        n_repeats: Number of permutation repeats
        random_state: Random seed
        output_dir: Directory to save results
        output_name: Base name for output files

    Returns:
        DataFrame with permutation importance results
    """
    import time as time_lib
    from lifelines.utils import concordance_index

    np.random.seed(random_state)

    # Define scoring function based on metric
    if metric == "c_index":
        if hasattr(model, "predict_risk"):
            # For scikit-survival models
            def score_func(model, X, time, event):
                risk_scores = model.predict_risk(X)
                return concordance_index(time, risk_scores, event)

        elif hasattr(model, "predict_partial_hazard"):
            # For Cox models (lifelines)
            def score_func(model, X, time, event):
                risk_scores = model.predict_partial_hazard(X).values
                return concordance_index(time, risk_scores, event)

        else:
            # Generic model
            def score_func(model, X, time, event):
                predictions = model.predict(X)
                return concordance_index(time, predictions, event)

    else:
        raise ValueError(
            f"Metric '{metric}' not implemented for permutation importance"
        )

    # Calculate baseline score
    baseline_score = score_func(model, X_test, time_test, event_test)

    # Prepare results storage
    importances = {
        "feature": [],
        "importance_mean": [],
        "importance_std": [],
    }

    start_time = time_lib.time()

    # Calculate permutation importance for each feature
    for feature in X_test.columns:
        feature_importances = []

        for _ in range(n_repeats):
            # Create a copy of the test data
            X_permuted = X_test.copy()

            # Permute the feature
            X_permuted[feature] = np.random.permutation(X_permuted[feature].values)

            # Calculate score with permuted feature
            permuted_score = score_func(model, X_permuted, time_test, event_test)

            # Calculate importance (decrease in score)
            importance = baseline_score - permuted_score
            feature_importances.append(importance)

        # Store results
        importances["feature"].append(feature)
        importances["importance_mean"].append(np.mean(feature_importances))
        importances["importance_std"].append(np.std(feature_importances))

        # Print progress every 10 features
        if (len(importances["feature"]) % 10) == 0:
            elapsed = time_lib.time() - start_time
            print(
                f"Processed {len(importances['feature'])}/{len(X_test.columns)} features in {elapsed:.2f} seconds"
            )

    # Create DataFrame
    importance_df = pd.DataFrame(importances)

    # Sort by importance
    importance_df = importance_df.sort_values("importance_mean", ascending=False)

    # Save results if output_dir specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        # Save to CSV
        importance_df.to_csv(f"{output_dir}/{output_name}.csv", index=False)

        # Create plot for top features
        plt.figure(figsize=(10, 8))

        # Limit to top 20 features
        top_n = min(20, len(importance_df))
        top_df = importance_df.head(top_n)

        # Create bar chart
        bars = plt.barh(
            np.arange(top_n),
            top_df["importance_mean"],
            xerr=top_df["importance_std"],
            capsize=5,
            color="steelblue",
            alpha=0.7,
        )

        # Add feature names
        plt.yticks(np.arange(top_n), top_df["feature"])

        # Add labels and title
        plt.xlabel(f"Importance ({metric} decrease)")
        plt.title("Permutation Feature Importance")
        plt.grid(axis="x", linestyle="--", alpha=0.3)
        plt.tight_layout()

        # Save plot
        plt.savefig(f"{output_dir}/{output_name}.png")
        plt.close()

    return importance_df
