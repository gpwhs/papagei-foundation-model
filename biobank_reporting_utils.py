import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any
import numpy as np
import seaborn as sns
from biobank_experiment_utils import compute_expected_calibration_error, ModelTypes
import os
from sklearn.metrics import classification_report, roc_curve, auc
import statsmodels.api as sm


def plot_correlation_matrix(
    features: pd.DataFrame,
    output_filename: str = "correlation_matrix.png",
    annot_threshold: int = 50,
) -> None:
    """
    Plots the correlation matrix.
    If the number of features exceeds annot_threshold, disables annotations for clarity.
    """
    corr = features.corr()
    num_features = corr.shape[0]
    figsize = (num_features / 3, num_features / 3)  # Dynamically adjust figure size
    plt.figure(figsize=figsize)
    # Disable annotations if too many features
    annot = num_features <= annot_threshold
    sns.heatmap(corr, annot=annot, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Features")
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Correlation matrix saved to {output_filename}")


def plot_calibration_curves(
    y_test: pd.Series,
    model_predictions: Dict[str, np.ndarray],
    model_dir: str,
) -> Dict[str, float]:
    """
    Plot calibration curves for all models and compute calibration metrics.

    Args:
        results: Dictionary of model results
        y_test: True labels
        model_predictions: Dictionary mapping model keys to predicted probabilities
        model_dir: Directory to save plot

    Returns:
        Dictionary of calibration metrics (Brier scores) for each model
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    plt.figure(figsize=(12, 9))

    # Set up colors for the different models
    colors = ["blue", "orange", "green", "red", "purple"]

    # Dictionary to store calibration metrics
    calibration_metrics = {}
    ece_values = compute_expected_calibration_error(y_test, model_predictions)

    # For each model, plot calibration curve and compute metrics
    for i, model_key in enumerate(["M0", "M1", "M2", "M3", "M4"]):
        if model_key not in model_predictions:
            continue

        y_pred_proba = model_predictions[model_key]

        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)

        # Compute Brier score (lower is better)
        brier_score = brier_score_loss(y_test, y_pred_proba)
        calibration_metrics[model_key] = brier_score

        # Plot calibration curve with metrics in the label
        plt.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=2,
            label=f"{model_key} - Brier: {brier_score:.4f}, ECE: {ece_values[model_key]:.4f}",
            color=colors[i],
        )

    # Plot perfectly calibrated line
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")

    # Add areas for over/under confidence labeling
    plt.text(0.25, 0.75, "Underconfidence", fontsize=12, alpha=0.7, ha="center")
    plt.text(0.75, 0.25, "Overconfidence", fontsize=12, alpha=0.7, ha="center")

    # Fill areas for better visualization
    plt.fill_between([0, 1], [0, 1], [0, 0], alpha=0.1, color="red", label="_nolegend_")
    plt.fill_between(
        [0, 1], [1, 1], [0, 1], alpha=0.1, color="blue", label="_nolegend_"
    )

    # Add diagonal grid lines to help with assessment
    plt.grid(True, alpha=0.3)

    # Set plot attributes
    plt.xlabel("Predicted probability")
    plt.ylabel("True probability in each bin")
    plt.title("Calibration Curves with Metrics")
    plt.legend(loc="best")

    # Add axis limits and ensure equal aspect ratio
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal", adjustable="box")

    # Add model explanations in top left
    model_names = {
        "M0": "PaPaGei Only",
        "M1": "Traditional Factors",
        "M2": "PaPaGei + Traditional",
        "M3": "pyPPG Only",
        "M4": "pyPPG + Traditional",
    }

    info_text = "\n".join([f"{k}: {v}" for k, v in model_names.items()])
    plt.annotate(
        info_text,
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        fontsize=10,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
    )

    # Save plot
    plt.tight_layout()
    plt.savefig(f"{model_dir}/calibration_curves.png")
    plt.close()

    # Create a bar chart of Brier scores
    plt.figure(figsize=(10, 6))

    model_names = list(calibration_metrics.keys())
    brier_scores = [calibration_metrics[m] for m in model_names]
    x = np.arange(len(model_names))

    bars = plt.bar(x, brier_scores, width=0.5, color=colors[: len(model_names)])

    # Add value labels above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.005,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xlabel("Model")
    plt.ylabel("Brier Score (lower is better)")
    plt.title("Calibration Error (Brier Score)")
    plt.xticks(x, model_names)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{model_dir}/brier_scores.png")
    plt.close()

    # Also create a bar chart of ECE values
    plt.figure(figsize=(10, 6))

    model_names = list(ece_values.keys())
    ece_scores = [ece_values[m] for m in model_names]
    x = np.arange(len(model_names))

    bars = plt.bar(x, ece_scores, width=0.5, color=colors[: len(model_names)])

    # Add value labels above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.005,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xlabel("Model")
    plt.ylabel("Expected Calibration Error (lower is better)")
    plt.title("Expected Calibration Error (ECE)")
    plt.xticks(x, model_names)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{model_dir}/ece_values.png")
    plt.close()

    # Create a summary DataFrame with both metrics
    calibration_summary = pd.DataFrame(
        {"Model": model_names, "Brier_Score": brier_scores, "ECE": ece_scores}
    )
    calibration_summary.to_csv(f"{model_dir}/calibration_metrics.csv", index=False)

    return calibration_metrics


def plot_experiment_comparison(results: dict, model_dir: str):
    """
    Plot experiment comparison metrics with error bars and save the figures.

    This function creates a bar plot for Accuracy, ROC AUC, PR AUC, and F1 Score with
    95% confidence interval error bars, then saves the combined figure as
    'experiment_comparison.png' in the given model_dir. It also generates a legend
    figure saved as 'model_legend.png'.

    Args:
        results (dict): Dictionary mapping experiment keys (e.g., 'M0', 'M1', ...) to objects
                        with metric attributes (accuracy, accuracy_lower_ci, accuracy_upper_ci,
                        auc, auc_lower_ci, auc_upper_ci, aucpr, aucpr_lower_ci, aucpr_upper_ci,
                        f1, f1_lower_ci, f1_upper_ci, etc.).
        model_dir (str): Directory to save the generated plots.
    """

    # Plot results comparison with error bars
    plt.figure(figsize=(20, 6))

    model_names = [f"M{i}" for i in range(5)]
    x = np.arange(len(model_names))
    width = 0.2  # width of the bars

    # Extract data for plotting
    accuracy_values = [results[m].accuracy for m in model_names]
    accuracy_errors = [
        (
            results[m].accuracy - results[m].accuracy_lower_ci,
            results[m].accuracy_upper_ci - results[m].accuracy,
        )
        for m in model_names
    ]
    accuracy_errors = np.array(accuracy_errors).T

    auc_values = [results[m].auc for m in model_names]
    auc_errors = [
        (
            results[m].auc - results[m].auc_lower_ci,
            results[m].auc_upper_ci - results[m].auc,
        )
        for m in model_names
    ]
    auc_errors = np.array(auc_errors).T

    pr_auc_values = [results[m].aucpr for m in model_names]
    pr_auc_errors = [
        (
            results[m].aucpr - results[m].aucpr_lower_ci,
            results[m].aucpr_upper_ci - results[m].aucpr,
        )
        for m in model_names
    ]
    pr_auc_errors = np.array(pr_auc_errors).T

    f1_values = [results[m].f1 for m in model_names]
    f1_errors = [
        (results[m].f1 - results[m].f1_lower_ci, results[m].f1_upper_ci - results[m].f1)
        for m in model_names
    ]
    f1_errors = np.array(f1_errors).T

    # Accuracy subplot
    plt.subplot(1, 4, 1)
    bars = plt.bar(
        x - width * 1.5,
        accuracy_values,
        width,
        color="blue",
        yerr=accuracy_errors,
        capsize=5,
    )
    plt.ylabel("Accuracy")
    plt.title("Accuracy with 95% CI")
    plt.xticks(x, model_names)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # ROC AUC subplot
    plt.subplot(1, 4, 2)
    bars = plt.bar(
        x - width / 2,
        auc_values,
        width,
        color="orange",
        yerr=auc_errors,
        capsize=5,
    )
    plt.ylabel("ROC AUC")
    plt.title("ROC AUC with 95% CI")
    plt.xticks(x, model_names)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # PR AUC subplot
    plt.subplot(1, 4, 3)
    bars = plt.bar(
        x + width / 2,
        pr_auc_values,
        width,
        color="purple",
        yerr=pr_auc_errors,
        capsize=5,
    )
    plt.ylabel("PR AUC")
    plt.title("PR AUC with 95% CI")
    plt.xticks(x, model_names)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # F1 Score subplot
    plt.subplot(1, 4, 4)
    bars = plt.bar(
        x + width * 1.5,
        f1_values,
        width,
        color="green",
        yerr=f1_errors,
        capsize=5,
    )
    plt.ylabel("F1 Score")
    plt.title("F1 Score with 95% CI")
    plt.xticks(x, model_names)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(f"{model_dir}/experiment_comparison.png")
    plt.close()

    # Create and save the legend figure
    plt.figure(figsize=(10, 2))
    plt.axis("off")
    legend_text = "\n".join(
        [
            "M0: PaPaGei Only",
            "M1: Traditional Factors (age, sex, BMI)",
            "M2: PaPaGei + Traditional",
            "M3: pyPPG Only",
            "M4: pyPPG + Traditional",
        ]
    )
    plt.text(0.5, 0.5, legend_text, ha="center", va="center", fontsize=12)
    plt.savefig(f"{model_dir}/model_legend.png", bbox_inches="tight")


def create_summary(results: dict, results_dir: str, model: str):
    """
    Create a summary of the experiment results.

    Args:
        results: Dictionary mapping experiment keys to ClassificationResults objects
        results_dir: Directory to save summary to
        model: Model type name
    """
    # Ensure model type subdirectory exists
    model_dir = f"{results_dir}/{model}"
    os.makedirs(model_dir, exist_ok=True)

    summary = pd.DataFrame(
        {
            "Model": [
                "M0: PaPaGei Only",
                "M1: Traditional Factors",
                "M2: PaPaGei + Traditional",
                "M3: pyPPG Only",
                "M4: pyPPG + Traditional",
            ],
            "Accuracy": [
                f"{results['M0'].accuracy:.4f} ({results['M0'].accuracy_lower_ci:.4f}-{results['M0'].accuracy_upper_ci:.4f})",
                f"{results['M1'].accuracy:.4f} ({results['M1'].accuracy_lower_ci:.4f}-{results['M1'].accuracy_upper_ci:.4f})",
                f"{results['M2'].accuracy:.4f} ({results['M2'].accuracy_lower_ci:.4f}-{results['M2'].accuracy_upper_ci:.4f})",
                f"{results['M3'].accuracy:.4f} ({results['M3'].accuracy_lower_ci:.4f}-{results['M3'].accuracy_upper_ci:.4f})",
                f"{results['M4'].accuracy:.4f} ({results['M4'].accuracy_lower_ci:.4f}-{results['M4'].accuracy_upper_ci:.4f})",
            ],
            "ROC_AUC": [
                f"{results['M0'].auc:.4f} ({results['M0'].auc_lower_ci:.4f}-{results['M0'].auc_upper_ci:.4f})",
                f"{results['M1'].auc:.4f} ({results['M1'].auc_lower_ci:.4f}-{results['M1'].auc_upper_ci:.4f})",
                f"{results['M2'].auc:.4f} ({results['M2'].auc_lower_ci:.4f}-{results['M2'].auc_upper_ci:.4f})",
                f"{results['M3'].auc:.4f} ({results['M3'].auc_lower_ci:.4f}-{results['M3'].auc_upper_ci:.4f})",
                f"{results['M4'].auc:.4f} ({results['M4'].auc_lower_ci:.4f}-{results['M4'].auc_upper_ci:.4f})",
            ],
            "PR_AUC": [
                f"{results['M0'].aucpr:.4f} ({results['M0'].aucpr_lower_ci:.4f}-{results['M0'].aucpr_upper_ci:.4f})",
                f"{results['M1'].aucpr:.4f} ({results['M1'].aucpr_lower_ci:.4f}-{results['M1'].aucpr_upper_ci:.4f})",
                f"{results['M2'].aucpr:.4f} ({results['M2'].aucpr_lower_ci:.4f}-{results['M2'].aucpr_upper_ci:.4f})",
                f"{results['M3'].aucpr:.4f} ({results['M3'].aucpr_lower_ci:.4f}-{results['M3'].aucpr_upper_ci:.4f})",
                f"{results['M4'].aucpr:.4f} ({results['M4'].aucpr_lower_ci:.4f}-{results['M4'].aucpr_upper_ci:.4f})",
            ],
            "F1": [
                f"{results['M0'].f1:.4f} ({results['M0'].f1_lower_ci:.4f}-{results['M0'].f1_upper_ci:.4f})",
                f"{results['M1'].f1:.4f} ({results['M1'].f1_lower_ci:.4f}-{results['M1'].f1_upper_ci:.4f})",
                f"{results['M2'].f1:.4f} ({results['M2'].f1_lower_ci:.4f}-{results['M2'].f1_upper_ci:.4f})",
                f"{results['M3'].f1:.4f} ({results['M3'].f1_lower_ci:.4f}-{results['M3'].f1_upper_ci:.4f})",
                f"{results['M4'].f1:.4f} ({results['M4'].f1_lower_ci:.4f}-{results['M4'].f1_upper_ci:.4f})",
            ],
            "Training_Time": [
                results["M0"].training_time,
                results["M1"].training_time,
                results["M2"].training_time,
                results["M3"].training_time,
                results["M4"].training_time,
            ],
        }
    )

    summary.to_csv(f"{model_dir}/experiment_summary.csv", index=False)
    print("\nExperiment Summary:")
    print(summary)
    print(f"\nSummary saved to {model_dir}/experiment_summary.csv")
    plot_experiment_comparison(results, model_dir)


def plot_pr_curve(
    y_test: pd.Series,
    y_pred_proba: np.ndarray,
    model_type: str,
    model_name: str,
    outcome: str,
    output_dir: str,
) -> None:
    """Plot Precision-Recall curve for a model.

    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
        model_type: Type of model
        model_name: Name of model
        outcome: Outcome being predicted
        output_dir: Directory to save plot
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    # Ensure model type subdirectory exists
    model_dir = f"{output_dir}/{model_type}"
    os.makedirs(model_dir, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)

    # Calculate no-skill line (proportion of positive samples)
    no_skill = sum(y_test) / len(y_test)

    plt.figure(figsize=(10, 8))
    plt.plot(
        recall,
        precision,
        color="darkorange",
        lw=2,
        label=f"PR curve (AP = {avg_precision:.2f})",
    )
    plt.plot(
        [0, 1],
        [no_skill, no_skill],
        color="navy",
        lw=2,
        linestyle="--",
        label=f"No Skill (baseline = {no_skill:.2f})",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"{model_type} {model_name} - Precision-Recall Curve for {outcome} Detection"
    )
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{model_dir}/{model_name}_pr_curve.png")
    plt.close()


def save_results_to_file(
    model_type: str,
    model_name: str,
    outcome: str,
    output_dir: str,
    best_params: Dict[str, Any],
    training_time: float,
    accuracy: float,
    accuracy_ci_lower: float,
    accuracy_ci_upper: float,
    roc_auc: float,
    roc_auc_ci_lower: float,
    roc_auc_ci_upper: float,
    aucpr: float,
    aucpr_ci_lower: float,
    aucpr_ci_upper: float,
    f1: float,
    f1_ci_lower: float,
    f1_ci_upper: float,
    y_test: pd.Series,
    y_pred: np.ndarray,
    cm: np.ndarray,
) -> None:
    """Save model results to a text file.

    Args:
        Various metrics and results to save
    """
    model_dir = f"{output_dir}/{model_type}"
    os.makedirs(model_dir, exist_ok=True)

    with open(f"{model_dir}/{model_name}_results.txt", "w") as f:
        f.write(f"{model_type} {model_name} - {outcome} Classification Results:\n")
        f.write(f"Best parameters: {best_params}\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write(
            f"Accuracy: {accuracy:.4f} (CI: {accuracy_ci_lower:.4f}-{accuracy_ci_upper:.4f})\n"
        )
        f.write(
            f"ROC AUC: {roc_auc:.4f} (CI: {roc_auc_ci_lower:.4f}-{roc_auc_ci_upper:.4f})\n"
        )
        f.write(
            f"PR AUC: {aucpr:.4f} (CI: {aucpr_ci_lower:.4f}-{aucpr_ci_upper:.4f})\n"
        )
        f.write(f"F1 Score: {f1:.4f} (CI: {f1_ci_lower:.4f}-{f1_ci_upper:.4f})\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))


def plot_roc_curve(
    y_test: pd.Series,
    y_pred_proba: np.ndarray,
    model_type: str,
    model_name: str,
    outcome: str,
    output_dir: str,
) -> None:
    """Plot ROC curve for a model.

    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
        model_type: Type of model
        model_name: Name of model
        outcome: Outcome being predicted
        output_dir: Directory to save plot
    """
    # Ensure model type subdirectory exists
    model_dir = f"{output_dir}/{model_type}"
    os.makedirs(model_dir, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_type} {model_name} - ROC Curve for {outcome} Detection")
    plt.legend(loc="lower right")
    plt.savefig(f"{model_dir}/{model_name}_roc_curve.png")
    plt.close()


def save_feature_importance(
    model: Any,
    feature_names: pd.Index,
    model_type: str,
    model_name: str,
    output_dir: str,
) -> None:
    """Save feature importance for a model.

    Args:
        model: Trained model with feature importance
        feature_names: Names of features
        model_type: Type of model
        model_name: Name of model
        output_dir: Directory to save results
    """
    model_dir = f"{output_dir}/{model_type}"
    os.makedirs(model_dir, exist_ok=True)

    # Model-specific handling of feature importance
    if model_type == ModelTypes.LOGISTIC_REGRESSION.value:
        # For logistic regression, use coefficients
        feature_importance = np.abs(model.coef_[0])

        # Create DataFrame for feature importance
        coef_df = pd.DataFrame(
            {"feature": feature_names, "coefficient": model.coef_[0]}
        )
        coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
        coef_df = coef_df.sort_values(by="abs_coefficient", ascending=False)

        # Save to file
        coef_df.to_csv(
            f"{model_dir}/{model_name}_feature_importance.csv",
            index=False,
        )

        # Plot feature importance
        importance_name = "Feature Coefficient (absolute value)"
        importance_values = feature_importance

    elif model_type == ModelTypes.XGBOOST.value:
        # For XGBoost, use feature_importances_
        feature_importance = model.feature_importances_

        # Create DataFrame for feature importance
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": feature_importance}
        )
        importance_df = importance_df.sort_values(by="importance", ascending=False)

        # Save to file
        importance_df.to_csv(
            f"{model_dir}/{model_name}_feature_importance.csv",
            index=False,
        )

        # Plot feature importance
        importance_name = "Feature Importance"
        importance_values = feature_importance

    elif model_type == ModelTypes.CATBOOST.value:
        # For CatBoost, we can extract feature importance in multiple ways

        # 1. Default feature importance (usually PredictionValuesChange)
        try:
            # Get feature importances as a DataFrame with feature names already included
            importance_df = model.get_feature_importance(prettified=True)

            # If the DataFrame doesn't have the expected format, create it manually
            if (
                "Feature Id" not in importance_df.columns
                or "Importance" not in importance_df.columns
            ):
                importance_values = model.get_feature_importance()
                importance_df = pd.DataFrame(
                    {"feature": feature_names, "importance": importance_values}
                )
            else:
                # Rename columns to be consistent with other models
                importance_df = importance_df.rename(
                    columns={"Feature Id": "feature", "Importance": "importance"}
                )

            # Sort by importance
            importance_df = importance_df.sort_values(by="importance", ascending=False)

            # Save to file
            importance_df.to_csv(
                f"{model_dir}/{model_name}_feature_importance.csv",
                index=False,
            )

            # Get values for plotting
            importance_name = "Feature Importance"
            importance_values = importance_df["importance"].values
            feature_names = importance_df["feature"].values

            # Also create a feature interaction plot if available
            try:
                # Save feature interactions if supported by this version of CatBoost
                interaction_importance = model.get_feature_importance(
                    type="Interaction"
                )
                if len(interaction_importance) > 0:
                    # Get the top interactions
                    top_interactions = min(20, len(interaction_importance))
                    plt.figure(figsize=(12, 8))
                    plt.barh(
                        range(top_interactions),
                        interaction_importance[:top_interactions],
                    )
                    plt.yticks(
                        range(top_interactions),
                        [
                            f"{feature_names[int(pair[0])]}<->{feature_names[int(pair[1])]}"
                            for pair in model.feature_interaction_info_[
                                "feature_pairs"
                            ][:top_interactions]
                        ],
                    )
                    plt.xlabel("Interaction Strength")
                    plt.title(f"{model_type} {model_name} - Top Feature Interactions")
                    plt.tight_layout()
                    plt.savefig(f"{model_dir}/{model_name}_feature_interactions.png")
                    plt.close()
            except (AttributeError, TypeError) as e:
                print(f"Feature interaction plot not available: {e}")

            # Get Shapley values-based importances if available
            try:
                # This produces SHAP importances which are often more interpretable
                shap_values = model.get_feature_importance(type="ShapValues")
                if shap_values is not None and len(shap_values) > 0:
                    shap_importance = np.abs(shap_values).mean(axis=0)
                    shap_df = pd.DataFrame(
                        {"feature": feature_names, "shap_importance": shap_importance}
                    )
                    shap_df = shap_df.sort_values(by="shap_importance", ascending=False)
                    shap_df.to_csv(
                        f"{model_dir}/{model_name}_shap_importance.csv",
                        index=False,
                    )

                    # Plot SHAP importance
                    num_features = min(20, len(feature_names))
                    plt.figure(figsize=(12, 8))
                    plt.barh(
                        range(num_features),
                        shap_df["shap_importance"].values[:num_features],
                        align="center",
                    )
                    plt.yticks(
                        range(num_features), shap_df["feature"].values[:num_features]
                    )
                    plt.xlabel("Mean |SHAP Value|")
                    plt.title(f"{model_type} {model_name} - SHAP Feature Importance")
                    plt.tight_layout()
                    plt.savefig(f"{model_dir}/{model_name}_shap_importance.png")
                    plt.close()
            except (AttributeError, TypeError) as e:
                print(f"SHAP values not directly available from model: {e}")

        except Exception as e:
            print(f"Error getting CatBoost feature importance: {e}")
            # Fallback to generic approach
            try:
                importance_values = model.feature_importances_
                importance_name = "Feature Importance"
            except AttributeError:
                print("Feature importance not available for this CatBoost model")
                return
    else:
        print(f"Feature importance not implemented for model type: {model_type}")
        return

    # Plot top 20 features or all if less than 20
    num_features = min(20, len(feature_names))
    plt.figure(figsize=(12, 8))

    # For CatBoost, we might already have sorted importance values
    if model_type == ModelTypes.CATBOOST.value and isinstance(
        feature_names, np.ndarray
    ):
        # Use the top features directly as they're already sorted
        top_features_idx = range(min(num_features, len(feature_names)))
        top_features_values = importance_values[:num_features]
        top_features_names = feature_names[:num_features]
    else:
        # For other models, sort the feature importance
        sorted_idx = np.argsort(importance_values)[::-1]
        top_features_idx = sorted_idx[:num_features]
        top_features_values = importance_values[top_features_idx]
        top_features_names = [feature_names[i] for i in top_features_idx]

    plt.barh(range(len(top_features_idx)), top_features_values, align="center")
    plt.yticks(range(len(top_features_idx)), top_features_names)
    plt.xlabel(importance_name)
    plt.title(f"{model_type} {model_name} - Top {num_features} Features for Detection")
    plt.tight_layout()
    plt.savefig(f"{model_dir}/{model_name}_feature_importance.png")
    plt.close()


def explain_model_predictions(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str,
    model_name: str,
    output_dir: str = "results",
    num_samples: int = 5,
) -> None:
    """
    Generate SHAP values to explain model predictions for individual samples.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_type: Type of model (e.g., "LR", "xgboost", "catboost")
        model_name: Name of the model for saving outputs
        output_dir: Directory to save results
        num_samples: Number of random samples to explain
    """
    # Ensure model type subdirectory exists
    model_dir = f"{output_dir}/{model_type}"
    os.makedirs(model_dir, exist_ok=True)

    try:
        import shap

        print(f"Generating SHAP explanations for {model_name}...")

        # Create model-specific explainer
        if model_type == ModelTypes.XGBOOST.value:
            explainer = shap.Explainer(model)
        elif model_type == ModelTypes.CATBOOST.value:
            # CatBoost needs special handling for SHAP
            # We need to convert the CatBoost model to a SHAP-compatible format
            # Method 1: Use TreeExplainer (works for most tree-based models)
            explainer = shap.TreeExplainer(model)

            # Alternative approach if TreeExplainer doesn't work:
            # 1. First get the CatBoost model predictions
            # prediction_function = lambda x: model.predict_proba(x)[:, 1]
            # background = shap.maskers.Independent(X_test, max_samples=100)
            # explainer = shap.Explainer(prediction_function, background)
        elif model_type == ModelTypes.LOGISTIC_REGRESSION.value:
            # For linear models, we use a different explainer
            explainer = shap.LinearExplainer(model, X_test)
        else:
            print(f"SHAP explanations not implemented for model type: {model_type}")
            return

        # Select random samples
        indices = np.random.choice(len(X_test), size=num_samples, replace=False)
        X_samples = X_test.iloc[indices]
        y_samples = y_test.iloc[indices]

        # Get SHAP values - handle CatBoost specifically
        if model_type == ModelTypes.CATBOOST.value:
            # For CatBoost with TreeExplainer we need to use specific syntax
            shap_values = explainer.shap_values(X_samples)

            # TreeExplainer might return a list for binary classification
            # (one array for each class)
            if isinstance(shap_values, list):
                # Use class 1 (positive class) for binary classification
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            # Convert to SHAP's Explanation object for compatibility with newer SHAP functions

            expected_value = explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = (
                    expected_value[1] if len(expected_value) > 1 else expected_value[0]
                )

            base_values = np.ones(len(X_samples)) * expected_value
            shap_explanation = shap.Explanation(
                values=shap_values,
                base_values=base_values,
                data=X_samples.values,
                feature_names=X_samples.columns.tolist(),
            )
        else:
            # For other models, use the standard approach
            shap_explanation = explainer(X_samples)

        # Plot SHAP values for each sample
        for i in range(num_samples):
            plt.figure(figsize=(12, 6))
            if model_type == ModelTypes.CATBOOST.value:
                # For CatBoost, use the explanation object we created
                shap_values_to_plot = shap_explanation[i : i + 1]
            else:
                shap_values_to_plot = shap_explanation[i]

            shap.plots.waterfall(shap_values_to_plot, max_display=10, show=False)
            plt.title(
                f"Sample {i+1} - True Label: {y_samples.iloc[i]}, Predicted: {model.predict(X_samples.iloc[[i]])[0]}"
            )
            plt.tight_layout()
            plt.savefig(f"{model_dir}/{model_name}_shap_sample_{i+1}.png")
            plt.close()

        # Create summary plot for all test data
        # Use a sample of 100 instances for better visualization
        sample_size = min(100, len(X_test))
        sample_indices = np.random.choice(len(X_test), size=sample_size, replace=False)
        X_for_summary = X_test.iloc[sample_indices]

        if model_type == ModelTypes.CATBOOST.value:
            # Get SHAP values for the summary data
            shap_values_summary = explainer.shap_values(X_for_summary)
            if isinstance(shap_values_summary, list):
                shap_values_summary = (
                    shap_values_summary[1]
                    if len(shap_values_summary) > 1
                    else shap_values_summary[0]
                )

            # Create a proper Explanation object
            base_values_summary = np.ones(len(X_for_summary)) * expected_value
            shap_explanation_summary = shap.Explanation(
                values=shap_values_summary,
                base_values=base_values_summary,
                data=X_for_summary.values,
                feature_names=X_for_summary.columns.tolist(),
            )
        else:
            shap_explanation_summary = explainer(X_for_summary)

        # Bar summary plot
        plt.figure(figsize=(10, 8))
        shap.plots.bar(shap_explanation_summary, show=False)
        plt.title(f"{model_name} - SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(f"{model_dir}/{model_name}_shap_importance.png")
        plt.close()

        # Beeswarm summary plot
        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(shap_explanation_summary, show=False)
        plt.title(f"{model_name} - SHAP Summary Plot")
        plt.tight_layout()
        plt.savefig(f"{model_dir}/{model_name}_shap_summary.png")
        plt.close()

    except ImportError:
        print("SHAP package not installed. Skipping model explanation.")
    except Exception as e:
        print(f"Error generating SHAP explanations: {e}")
        import traceback

        traceback.print_exc()


def plot_odds_ratios(
    X,
    y,
    feature_names=None,
    title="Odds Ratios with 95% CI",
    plot_filename="odds_ratios.png",
    csv_filename="odds_ratios.csv",
):
    """
    Plot Odds Ratios (ORs) and 95% Confidence Intervals from a trained logistic regression model.

    If there are more than 10 features, only the top 10 most important features
    (largest deviation from an OR of 1) are plotted.
    In all cases, all computed metrics are saved to a CSV file.

    Parameters:
    - X: ndarray or DataFrame of shape (n_samples, n_features)
    - y: ndarray of shape (n_samples,)
    - feature_names: list of strings (feature names)
    - title: title of the plot
    - plot_filename: filename for the saved plot
    - csv_filename: filename for saving the full odds ratios data
    """
    # If no feature names provided, create default names.
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]

    # Fit logistic regression using statsmodels for easy CI extraction.
    X_sm = sm.add_constant(X)
    sm_model = sm.Logit(y, X_sm).fit(disp=False)

    # Extract odds ratios and confidence intervals.
    params = sm_model.params[1:]  # Exclude intercept.
    conf = sm_model.conf_int()[1:]
    or_vals = np.exp(params)
    or_lower = np.exp(conf[0])
    or_upper = np.exp(conf[1])

    # Prepare DataFrame with all features.
    df_or = pd.DataFrame(
        {
            "Feature": feature_names,
            "OddsRatio": or_vals,
            "CI_lower": or_lower,
            "CI_upper": or_upper,
        }
    )

    # Compute an importance metric as the absolute deviation from 1 (in log space).
    df_or["Importance"] = np.abs(np.log(df_or["OddsRatio"]))

    # Write all metrics to CSV.
    df_or.to_csv(csv_filename, index=False)

    # Select only the top 10 most important features for plotting,
    # but if there are fewer than 10, just use them all.
    if len(df_or) > 10:
        df_plot = df_or.nlargest(10, "Importance")
    else:
        df_plot = df_or.copy()

    # Sort the features for better visualization (ascending OddsRatio).
    df_plot = df_plot.sort_values(by="OddsRatio", ascending=True)

    # Plotting.
    plt.figure(figsize=(8, len(df_plot) * 0.5))
    ax = plt.gca()
    ax.errorbar(
        df_plot["OddsRatio"],
        df_plot["Feature"],
        xerr=[
            df_plot["OddsRatio"] - df_plot["CI_lower"],
            df_plot["CI_upper"] - df_plot["OddsRatio"],
        ],
        fmt="o",
        color="navy",
        ecolor="gray",
        elinewidth=3,
        capsize=4,
    )
    ax.axvline(x=1, linestyle="--", color="red", linewidth=1)
    ax.set_xscale("linear")
    ax.set_xlabel("Odds Ratios")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(plot_filename)
