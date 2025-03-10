import os
import time
import argparse
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pydantic import BaseModel

from sklearn.metrics import (
    roc_curve,
    auc,
    f1_score,
    classification_report,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from biobank_embeddings_extraction import extract_features
from linearprobing.utils import bootstrap_metric_confidence_interval
from biobank_utils import (
    load_yaml_config,
    remove_highly_corr_features,
    get_embedding_df,
    PYPPG_FEATURES,
    create_summary,
    setup_model,
    ModelTypes,
)


class ClassificationResults(BaseModel):
    """Results from a classification experiment."""

    model: str
    parameters: Dict[str, Any]
    auc: float
    auc_lower_ci: float
    auc_upper_ci: float
    aucpr: float  #
    aucpr_lower_ci: float
    aucpr_upper_ci: float
    f1: float
    f1_lower_ci: float
    f1_upper_ci: float
    accuracy: float
    accuracy_lower_ci: float
    accuracy_upper_ci: float
    training_time: float


class ExperimentConfig:
    """Configuration for an experiment."""

    def __init__(
        self,
        name: str,
        description: str,
        feature_columns: List[str],
    ):
        self.name = name
        self.description = description
        self.feature_columns = feature_columns


def plot_calibration_curves(
    results: Dict[str, Any],
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
    from sklearn.calibration import calibration_curve, CalibrationDisplay
    from sklearn.metrics import brier_score_loss

    plt.figure(figsize=(12, 9))

    # Set up colors for the different models
    colors = ["blue", "orange", "green", "red", "purple"]

    # Dictionary to store calibration metrics
    calibration_metrics = {}
    ece_values = compute_expected_calibration_error(results, y_test, model_predictions)

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


def compute_expected_calibration_error(
    results: Dict[str, Any],
    y_test: pd.Series,
    model_predictions: Dict[str, np.ndarray],
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute Expected Calibration Error (ECE) for each model.

    ECE measures the difference between predicted probabilities and the true fraction
    of positive samples by binning predictions and taking a weighted average.

    Args:
        results: Dictionary of model results
        y_test: True labels
        model_predictions: Dictionary mapping model keys to predicted probabilities
        n_bins: Number of bins for calculating ECE

    Returns:
        Dictionary of ECE values for each model
    """
    ece_values = {}

    for model_key in model_predictions:
        y_pred_proba = model_predictions[model_key]

        # Bin predictions
        bin_indices = np.linspace(0, 1, n_bins + 1)
        bin_assignments = np.digitize(y_pred_proba, bin_indices) - 1
        bin_assignments = np.clip(
            bin_assignments, 0, n_bins - 1
        )  # Clip values at the edges

        # Calculate ECE
        ece = 0
        for bin_idx in range(n_bins):
            # Get samples in this bin
            bin_mask = bin_assignments == bin_idx
            if not np.any(bin_mask):
                continue

            bin_preds = y_pred_proba[bin_mask]
            bin_true = y_test.values[bin_mask]
            bin_size = bin_mask.sum()

            # Calculate average prediction and true fraction for this bin
            avg_pred = np.mean(bin_preds)
            true_frac = np.mean(bin_true)

            # Add weighted absolute difference to ECE
            ece += (bin_size / len(y_test)) * np.abs(avg_pred - true_frac)

        ece_values[model_key] = ece

    return ece_values


def get_embeddings(df: pd.DataFrame, cache_file: str = "embeddings.npy") -> np.ndarray:
    """Get or compute embeddings with caching.

    Args:
        df: DataFrame containing the PPG data
        cache_file: File to cache embeddings to/from

    Returns:
        Array of embeddings
    """
    if os.path.exists(cache_file):
        print(f"Loading pre-computed embeddings from {cache_file}")
        return np.load(cache_file)

    print("Extracting features...")
    embeddings = extract_features(df)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
    np.save(cache_file, embeddings)
    print(f"Embeddings saved to {cache_file}")

    return embeddings


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
    traditional_features = ["age", "sex", "BMI"]
    traditional_df = df[traditional_features]

    # Extract pyPPG features
    pyppg_df = df[PYPPG_FEATURES]

    # Combine all features
    all_features = pd.concat([embedding_df, traditional_df, pyppg_df], axis=1)

    # Drop the outcome column if present
    if outcome in all_features.columns:
        all_features = all_features.drop(columns=[outcome])

    # Extract target variable
    target = df[outcome]

    return all_features, target


def setup_experiments(
    embedding_columns: List[str],
    traditional_features: List[str] = ["age", "sex", "BMI"],
) -> Dict[str, ExperimentConfig]:
    """Set up experiment configurations.

    Args:
        embedding_columns: List of embedding column names
        traditional_features: List of traditional feature names

    Returns:
        Dictionary mapping experiment keys to configurations
    """
    return {
        "M0": ExperimentConfig(
            name="M0_PaPaGei_Only",
            description="Only PaPaGei features",
            feature_columns=embedding_columns,
        ),
        "M1": ExperimentConfig(
            name="M1_Traditional_Only",
            description="Only metadata (age, sex, BMI)",
            feature_columns=traditional_features,
        ),
        "M2": ExperimentConfig(
            name="M2_PaPaGei_Traditional",
            description="Both PaPaGei features and metadata",
            feature_columns=embedding_columns + traditional_features,
        ),
        "M3": ExperimentConfig(
            name="M3_pyPPG_Only",
            description="pyPPG features",
            feature_columns=PYPPG_FEATURES,
        ),
        "M4": ExperimentConfig(
            name="M4_pyPPG_Traditional",
            description="pyPPG features and metadata",
            feature_columns=PYPPG_FEATURES + traditional_features,
        ),
    }


def plot_learning_curves(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str,
    model_name: str,
    output_dir: str = "results",
) -> None:
    """
    Plot learning curves for the model.

    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        model_type: Type of model (e.g., "LR", "xgboost")
        model_name: Name of the model for saving outputs
        output_dir: Directory to save results

    Returns:
        None: Saves learning curve plots to disk
    """
    # This function only works for XGBoost models
    if model_type != ModelTypes.XGBOOST.value:
        return

    # Ensure model type subdirectory exists
    model_dir = f"{output_dir}/{model_type}"
    os.makedirs(model_dir, exist_ok=True)

    try:
        import xgboost as xgb

        # Check if the model is an XGBoost model
        if not isinstance(model, xgb.XGBModel):
            print(
                f"Expected XGBoost model but got {type(model)}. Cannot plot learning curves."
            )
            return

        # Get evaluation results if they exist
        if not hasattr(model, "evals_result_"):
            print(
                "Model does not have evaluation results. Cannot plot learning curves."
            )
            return

        evals_result = model.evals_result_

        # If we have test evaluation results, we can plot curves
        if "validation_0" in evals_result:
            plt.figure(figsize=(12, 5))

            metrics = list(evals_result["validation_0"].keys())

            for i, metric in enumerate(metrics):
                plt.subplot(1, len(metrics), i + 1)

                if "train" in evals_result:
                    x_axis = range(len(evals_result["train"][metric]))
                    plt.plot(x_axis, evals_result["train"][metric], label="Train")

                x_axis = range(len(evals_result["validation_0"][metric]))
                plt.plot(x_axis, evals_result["validation_0"][metric], label="Test")

                plt.title(f"XGBoost {metric.upper()}")
                plt.xlabel("Boosting Rounds")
                plt.ylabel(metric.upper())
                plt.legend()

            plt.tight_layout()
            plt.savefig(f"{model_dir}/{model_name}_learning_curves.png")
            plt.close()

            print(
                f"Learning curves saved to {model_dir}/{model_name}_learning_curves.png"
            )
        else:
            # If no validation results, we'll use a modified approach by training a new model
            # with the same parameters but tracking the training
            print(
                "Model does not have validation results. Training a new model to generate learning curves..."
            )

            # Extract parameters from the existing model
            params = model.get_params()

            # Remove parameters that aren't used by the XGBoost native API
            native_params = {
                k: v
                for k, v in params.items()
                if k
                not in [
                    "verbose",
                    "n_jobs",
                    "random_state",
                    "use_label_encoder",
                    "n_estimators",
                ]
            }

            # Convert parameter names to XGBoost native format
            xgb_params = {}
            for k, v in native_params.items():
                if k == "reg_alpha":
                    xgb_params["alpha"] = v
                elif k == "reg_lambda":
                    xgb_params["lambda"] = v
                elif k == "learning_rate":
                    xgb_params["eta"] = v
                else:
                    xgb_params[k] = v

            xgb_params["objective"] = "binary:logistic"
            xgb_params["eval_metric"] = ["logloss", "auc"]
            xgb_params["seed"] = 42

            # Create DMatrix objects for training
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            # Train with tracking
            evals_result = {}
            xgb.train(
                params=xgb_params,
                dtrain=dtrain,
                num_boost_round=params.get("n_estimators", 100),
                evals=[(dtrain, "train"), (dtest, "validation")],
                evals_result=evals_result,
                verbose_eval=False,
            )

            # Plot the learning curves
            plt.figure(figsize=(12, 5))

            metrics = list(evals_result["validation"].keys())

            for i, metric in enumerate(metrics):
                plt.subplot(1, len(metrics), i + 1)

                x_axis = range(len(evals_result["train"][metric]))
                plt.plot(x_axis, evals_result["train"][metric], label="Train")

                x_axis = range(len(evals_result["validation"][metric]))
                plt.plot(x_axis, evals_result["validation"][metric], label="Test")

                plt.title(f"XGBoost {metric.upper()}")
                plt.xlabel("Boosting Rounds")
                plt.ylabel(metric.upper())
                plt.legend()

            plt.tight_layout()
            plt.savefig(f"{model_dir}/{model_name}_learning_curves.png")
            plt.close()

            print(
                f"Learning curves saved to {model_dir}/{model_name}_learning_curves.png"
            )

    except ImportError:
        print("XGBoost package not installed. Skipping learning curves.")
    except Exception as e:
        print(f"Error generating learning curves: {e}")
        import traceback

        traceback.print_exc()


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
        model_type: Type of model (e.g., "LR", "xgboost")
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

        # Get SHAP values
        shap_values = explainer(X_samples)

        # Plot SHAP values for each sample
        for i in range(num_samples):
            plt.figure(figsize=(12, 6))
            shap.plots.waterfall(shap_values[i], max_display=10, show=False)
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

        shap_values_summary = explainer(X_for_summary)

        # Bar summary plot
        plt.figure(figsize=(10, 8))
        shap.plots.bar(shap_values_summary, show=False)
        plt.title(f"{model_name} - SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(f"{model_dir}/{model_name}_shap_importance.png")
        plt.close()

        # Beeswarm summary plot
        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(shap_values_summary, show=False)
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


def train_and_evaluate_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_type: str,
    model_name: str,
    outcome: str,
    output_dir: str,
    collect_predictions: bool = True,
    handle_imbalance: bool = False,
) -> Tuple[ClassificationResults, Optional[np.ndarray]]:
    """Train a model and evaluate it.

    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        model_type: Type of model (e.g., "LR" for logistic regression)
        model_name: Name of the model for saving outputs
        outcome: Name of the outcome variable
        output_dir: Directory to save results
        collect_predictions: Whether to return prediction probabilities for calibration analysis
        handle_imbalance: Whether to apply class weighting for imbalanced datasets

    Returns:
        Tuple of (ClassificationResults, predicted_probabilities)
    """
    from sklearn.metrics import precision_recall_curve, auc as sklearn_auc
    from sklearn.metrics import average_precision_score

    # Ensure output directory exists
    model_output_dir = f"{output_dir}/{model_type}"
    os.makedirs(model_output_dir, exist_ok=True)

    # Check for class imbalance and print information
    class_counts = y_train.value_counts()
    class_ratio = class_counts.min() / class_counts.max()
    is_imbalanced = class_ratio < 0.3  # A common threshold for imbalance

    print("Class distribution in training set:")
    for class_label, count in class_counts.items():
        print(f"  Class {class_label}: {count} ({count/len(y_train):.2%})")
    print(f"Class ratio (minority/majority): {class_ratio:.3f}")

    if is_imbalanced:
        print("Warning: Dataset appears imbalanced.")
        if handle_imbalance:
            print("Applying class weighting to handle imbalance.")
        else:
            print(
                "Class weighting not enabled. Consider setting handle_imbalance=True for better results."
            )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Set up model and hyperparameter search space
    model, parameter_distributions = setup_model(ModelTypes(model_type))

    # Handle imbalanced data if requested
    if handle_imbalance:
        # For XGBoost, we modify parameter_distributions to include scale_pos_weight
        if model_type == ModelTypes.XGBOOST.value:
            # Calculate the scale_pos_weight based on class distribution
            negative_samples = sum(y_train == 0)
            positive_samples = sum(y_train == 1)
            scale_pos_weight = negative_samples / positive_samples

            # Check if parameter_distributions is a list or dictionary
            if isinstance(parameter_distributions, list):
                # Handle list of dictionaries (like for LogisticRegression)
                for param_dict in parameter_distributions:
                    if "scale_pos_weight" not in param_dict:
                        param_dict["scale_pos_weight"] = [1, scale_pos_weight]
            else:
                # Handle dictionary
                if "scale_pos_weight" not in parameter_distributions:
                    parameter_distributions["scale_pos_weight"] = [1, scale_pos_weight]

            print(f"Set scale_pos_weight options: [1, {scale_pos_weight:.2f}]")

        # For Logistic Regression, we always include class_weight parameter
        elif model_type == ModelTypes.LOGISTIC_REGRESSION.value:
            # For LR, class_weight is already included in parameter_distributions in setup_model
            print("Using class_weight parameter options for Logistic Regression")

    # RandomizedSearchCV for parameter tuning
    random_search = RandomizedSearchCV(
        model,
        param_distributions=parameter_distributions,
        n_iter=60,
        cv=2,
        scoring="roc_auc",
        return_train_score=True,
        n_jobs=-1,
        verbose=3,
        random_state=42,
    )

    # Train the model
    start_time = time.time()
    random_search.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time

    best_model = random_search.best_estimator_
    print(f"{model_name} - Best parameters: {random_search.best_params_}")

    # Fit the best model (necessary? it should already be fitted)
    best_model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Calculate AUCPR (Area Under Precision-Recall Curve)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    aucpr = sklearn_auc(recall, precision)
    # Alternative calculation using average_precision_score
    avg_precision = average_precision_score(y_test, y_pred_proba)

    # Print which metric was used
    print(f"AUCPR: {aucpr:.4f} (calculated from precision-recall curve)")
    print(f"Average Precision: {avg_precision:.4f} (alternative calculation)")

    # Calculate confidence intervals
    roc_auc_ci_lower, roc_auc_ci_upper, _ = bootstrap_metric_confidence_interval(
        y_test, y_pred_proba, roc_auc_score
    )
    accuracy_ci_lower, accuracy_ci_upper, _ = bootstrap_metric_confidence_interval(
        y_test, y_pred, accuracy_score
    )
    f1_ci_lower, f1_ci_upper, _ = bootstrap_metric_confidence_interval(
        y_test, y_pred, f1_score
    )

    # Calculate AUCPR confidence intervals
    def pr_auc_score(y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return sklearn_auc(recall, precision)

    aucpr_ci_lower, aucpr_ci_upper, _ = bootstrap_metric_confidence_interval(
        y_test, y_pred_proba, pr_auc_score
    )

    # Print evaluation metrics
    print(f"\n{model_name} - {outcome} Classification Results:")
    print(
        f"Accuracy: {accuracy:.4f} (CI: {accuracy_ci_lower:.4f}-{accuracy_ci_upper:.4f})"
    )
    print(f"ROC AUC: {roc_auc:.4f} (CI: {roc_auc_ci_lower:.4f}-{roc_auc_ci_upper:.4f})")
    print(f"PR AUC: {aucpr:.4f} (CI: {aucpr_ci_lower:.4f}-{aucpr_ci_upper:.4f})")
    print(f"F1 Score: {f1:.4f} (CI: {f1_ci_lower:.4f}-{f1_ci_upper:.4f})")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Save results to files
    save_results_to_file(
        model_type=model_type,
        model_name=model_name,
        outcome=outcome,
        output_dir=output_dir,
        best_params=random_search.best_params_,
        training_time=training_time,
        accuracy=accuracy,
        accuracy_ci_lower=accuracy_ci_lower,
        accuracy_ci_upper=accuracy_ci_upper,
        roc_auc=roc_auc,
        roc_auc_ci_lower=roc_auc_ci_lower,
        roc_auc_ci_upper=roc_auc_ci_upper,
        aucpr=aucpr,
        aucpr_ci_lower=aucpr_ci_lower,
        aucpr_ci_upper=aucpr_ci_upper,
        f1=f1,
        f1_ci_lower=f1_ci_lower,
        f1_ci_upper=f1_ci_upper,
        y_test=y_test,
        y_pred=y_pred,
        cm=cm,
    )

    # Save the model and scaler
    joblib.dump(best_model, f"{model_output_dir}/{model_name}_classifier.joblib")
    joblib.dump(scaler, f"{model_output_dir}/{model_name}_scaler.joblib")

    # Plot ROC curve
    plot_roc_curve(
        y_test=y_test,
        y_pred_proba=y_pred_proba,
        model_type=model_type,
        model_name=model_name,
        outcome=outcome,
        output_dir=output_dir,
    )

    # Plot Precision-Recall curve
    plot_pr_curve(
        y_test=y_test,
        y_pred_proba=y_pred_proba,
        model_type=model_type,
        model_name=model_name,
        outcome=outcome,
        output_dir=output_dir,
    )

    # Feature importance (for applicable models)
    save_feature_importance(
        model=best_model,
        feature_names=X_train.columns,
        model_type=model_type,
        model_name=model_name,
        output_dir=output_dir,
    )

    # Plot learning curves (for XGBoost)
    if model_type == ModelTypes.XGBOOST.value:
        plot_learning_curves(
            best_model,
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            model_type,
            model_name,
            output_dir,
        )

    # Generate SHAP explanations
    explain_model_predictions(
        best_model,
        X_test,
        y_test,
        model_type,
        model_name,
        output_dir,
    )

    # Create results object
    results = ClassificationResults(
        model=model_type,
        parameters=random_search.best_params_,
        auc=roc_auc,
        auc_lower_ci=roc_auc_ci_lower,
        auc_upper_ci=roc_auc_ci_upper,
        aucpr=aucpr,
        aucpr_lower_ci=aucpr_ci_lower,
        aucpr_upper_ci=aucpr_ci_upper,
        f1=f1,
        f1_lower_ci=f1_ci_lower,
        f1_upper_ci=f1_ci_upper,
        accuracy=accuracy,
        accuracy_lower_ci=accuracy_ci_lower,
        accuracy_upper_ci=accuracy_ci_upper,
        training_time=training_time,
    )

    if collect_predictions:
        return results, y_pred_proba
    else:
        return results, None


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

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
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

    else:
        print(f"Feature importance not implemented for model type: {model_type}")
        return

    # Plot top 20 features or all if less than 20
    num_features = min(20, len(feature_names))
    plt.figure(figsize=(12, 8))
    sorted_idx = np.argsort(importance_values)[::-1]
    top_features = sorted_idx[:num_features]

    plt.barh(range(len(top_features)), importance_values[top_features], align="center")
    plt.yticks(range(len(top_features)), [feature_names[i] for i in top_features])
    plt.xlabel(importance_name)
    plt.title(f"{model_type} {model_name} - Top {num_features} Features for Detection")
    plt.tight_layout()
    plt.savefig(f"{model_dir}/{model_name}_feature_importance.png")
    plt.close()


def run_experiment(
    experiment_config: ExperimentConfig,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_type: str,
    outcome: str,
    output_dir: str,
    handle_imbalance: bool = False,
) -> Tuple[ClassificationResults, np.ndarray]:
    """Run a single experiment.

    Args:
        experiment_config: Configuration for the experiment
        X_train, X_test, y_train, y_test: Data splits
        model_type: Type of model to use
        outcome: Outcome variable name
        output_dir: Directory to save results
        handle_imbalance: Whether to apply class weighting for imbalanced datasets

    Returns:
        Tuple of (ClassificationResults, prediction_probabilities)
    """
    print(
        f"\n--- Running Experiment {experiment_config.name}: {experiment_config.description} ---"
    )

    # Select features for this experiment
    X_train_exp = X_train[experiment_config.feature_columns]
    X_test_exp = X_test[experiment_config.feature_columns]

    # Train and evaluate
    results, y_pred_proba = train_and_evaluate_model(
        X_train_exp,
        X_test_exp,
        y_train,
        y_test,
        model_type=model_type,
        model_name=experiment_config.name,
        outcome=outcome,
        output_dir=output_dir,
        collect_predictions=True,
        handle_imbalance=handle_imbalance,
    )

    return results, y_pred_proba


def main() -> None:
    """Main function to run experiments."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load Configuration from YAML file")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="xgb_config.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_yaml_config(args.config)
    model = config["model"]
    outcome = config["outcome"]
    results_dir = config["results_directory"]
    handle_imbalance = config["handle_imbalance"]
    print(f"Running experiments with model {model} for outcome {outcome}")

    if handle_imbalance:
        print("Class weighting enabled for handling imbalanced data")
    else:
        print("Class weighting not enabled (use --handle_imbalance flag to enable)")

    # Create a nested directory structure
    outcome_dir = f"{results_dir}/{outcome}"
    model_dir = f"{outcome_dir}/{model}"
    os.makedirs(model_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    data_path = os.getenv("BIOBANK_DATA_PATH")
    if not data_path:
        raise ValueError("BIOBANK_DATA_PATH environment variable is not set")

    df = pd.read_parquet(f"{data_path}/215k_pyppg_features_and_conditions.parquet")
    df = df.dropna()

    # Convert string representations of arrays to numpy arrays if needed
    if isinstance(df["ppg"].iloc[0], str):
        df["ppg"] = df["ppg"].apply(lambda x: np.array(eval(x)))

    # Get or compute embeddings
    embeddings = get_embeddings(df, cache_file="embeddings.npy")

    print("creating embedding dataframe")
    # Create embedding DataFrame
    embedding_df = get_embedding_df(embeddings)
    print("removing highly correlated features")
    embedding_df = remove_highly_corr_features(embedding_df)

    # Ensure outcome variable is in the embedding DataFrame for later splitting
    if outcome not in embedding_df.columns:
        embedding_df[outcome] = df[outcome]

    # Prepare data
    all_features, target = preprocess_data(df, outcome, embedding_df)

    # Print class distribution information for the outcome
    class_counts = target.value_counts()
    print(f"\nOutcome ({outcome}) class distribution:")
    for class_label, count in class_counts.items():
        print(f"  Class {class_label}: {count} ({count/len(target):.2%})")

    class_ratio = class_counts.min() / class_counts.max()
    print(f"Class ratio (minority/majority): {class_ratio:.3f}")

    if class_ratio < 0.3:
        print(
            "Warning: Dataset appears imbalanced. Consider using the --handle_imbalance flag."
        )

    # Create train/test splits
    X_train, X_test, y_train, y_test = train_test_split(
        all_features, target, test_size=0.2, random_state=42, stratify=target
    )

    # Setup experiments
    embedding_columns = [col for col in embedding_df.columns if col != outcome]
    experiments = setup_experiments(embedding_columns)

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
        results=results,
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
