import os
from biobank_classification import (
    preprocess_data,
    plot_calibration_curves,
)
from biobank_utils import (
    load_yaml_config,
    remove_highly_corr_features,
    get_embedding_df,
    create_summary,
    PYPPG_FEATURES,
)
from biobank_embeddings_extraction import get_embeddings

# Import TabPFN specific functions
from tabpfn_extensions.rf_pfn import (
    RandomForestTabPFNClassifier,
)
from tabpfn_extensions import TabPFNClassifier
import argparse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from biobank_classification import ExperimentConfig, ClassificationResults
from sklearn.metrics import (
    roc_curve,
    auc,
    f1_score,
    classification_report,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)

from linearprobing.utils import bootstrap_metric_confidence_interval


def train_and_evaluate_tabpfn_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
    outcome: str,
    output_dir: str,
    collect_predictions: bool = True,
    handle_imbalance: bool = False,
) -> Tuple[Any, Optional[np.ndarray]]:
    """Train a TabPFN model and evaluate it.

    This function handles the special requirements of TabPFN models.

    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        model_name: Name of the model for saving outputs
        outcome: Name of the outcome variable
        output_dir: Directory to save results
        collect_predictions: Whether to return prediction probabilities for calibration analysis
        handle_imbalance: Whether to apply class weighting for imbalanced datasets

    Returns:
        Tuple of (ClassificationResults, predicted_probabilities)
    """

    # Ensure output directory exists
    model_type = "tabpfn"
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
            print("Note: TabPFN has its own mechanisms for handling imbalance.")
        else:
            print(
                "Class weighting not enabled, but TabPFN may handle imbalance internally."
            )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Dataset shape - X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Check if dataset is too large for TabPFN
    if X_train.shape[0] > 10000 or X_train.shape[1] > 100:
        print(
            f"Large dataset detected: {X_train.shape[0]} samples, {X_train.shape[1]} features"
        )
        print(
            "Using RandomForestTabPFNClassifier with size limitations to handle large dataset"
        )
        clf_base = TabPFNClassifier(
            ignore_pretraining_limits=True,
            inference_config={"SUBSAMPLE_SAMPLES": 1000},
        )
        tabpfn_tree_clf = RandomForestTabPFNClassifier(
            tabpfn=clf_base,
            verbose=1,
            max_predict_time=60,
        )

        # Create base TabPFN model with ignore_pretraining_limits=True
        tabpfn_tree_clf.fit(X_train_scaled, y_train)
        prediction_probs = tabpfn_tree_clf.predict_proba(X_test_scaled)
        predictions = np.argmax(prediction_probs, axis=1)
        print(f"ROC AUC: {roc_auc_score(y_test, prediction_probs[:, 1]):.4f}")
        print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

    else:
        print("Using TabPFN directly since dataset is within size limits.")
        # Create TabPFN model
        model = TabPFNClassifier(device="cpu", N_ensemble_configurations=32)

        # Train TabPFN directly - TabPFN doesn't benefit from traditional hyperparameter tuning
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time

        best_model = model
        best_params = {"N_ensemble_configurations": 32}  # Default TabPFN parameter

    # Evaluate the model
    # y_pred = best_model.predict(X_test_scaled)
    # y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    #
    # # Calculate metrics
    # roc_auc = roc_auc_score(y_test, y_pred_proba)
    # accuracy = accuracy_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred)
    #
    # # Calculate AUCPR (Area Under Precision-Recall Curve)
    # precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    # aucpr = auc(recall, precision)
    # # Alternative calculation using average_precision_score
    # avg_precision = average_precision_score(y_test, y_pred_proba)
    #
    # # Print which metric was used
    # print(f"AUCPR: {aucpr:.4f} (calculated from precision-recall curve)")
    # print(f"Average Precision: {avg_precision:.4f} (alternative calculation)")
    #
    # # Calculate confidence intervals
    # roc_auc_ci_lower, roc_auc_ci_upper, _ = bootstrap_metric_confidence_interval(
    #     y_test, y_pred_proba, roc_auc_score
    # )
    # accuracy_ci_lower, accuracy_ci_upper, _ = bootstrap_metric_confidence_interval(
    #     y_test, y_pred, accuracy_score
    # )
    # f1_ci_lower, f1_ci_upper, _ = bootstrap_metric_confidence_interval(
    #     y_test, y_pred, f1_score
    # )
    #
    # # Calculate AUCPR confidence intervals
    # def pr_auc_score(y_true, y_score):
    #     precision, recall, _ = precision_recall_curve(y_true, y_score)
    #     return auc(recall, precision)
    #
    # aucpr_ci_lower, aucpr_ci_upper, _ = bootstrap_metric_confidence_interval(
    #     y_test, y_pred_proba, pr_auc_score
    # )
    #
    # # Print evaluation metrics
    # print(f"\n{model_name} - {outcome} Classification Results:")
    # print(
    #     f"Accuracy: {accuracy:.4f} (CI: {accuracy_ci_lower:.4f}-{accuracy_ci_upper:.4f})"
    # )
    # print(f"ROC AUC: {roc_auc:.4f} (CI: {roc_auc_ci_lower:.4f}-{roc_auc_ci_upper:.4f})")
    # print(f"PR AUC: {aucpr:.4f} (CI: {aucpr_ci_lower:.4f}-{aucpr_ci_upper:.4f})")
    # print(f"F1 Score: {f1:.4f} (CI: {f1_ci_lower:.4f}-{f1_ci_upper:.4f})")
    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred))
    #
    # # Confusion matrix
    # cm = confusion_matrix(y_test, y_pred)
    # print("\nConfusion Matrix:")
    # print(cm)
    #
    # # Save results to files
    # save_tabpfn_results_to_file(
    #     model_name=model_name,
    #     outcome=outcome,
    #     output_dir=output_dir,
    #     best_params=best_params,
    #     training_time=training_time,
    #     accuracy=accuracy,
    #     accuracy_ci_lower=accuracy_ci_lower,
    #     accuracy_ci_upper=accuracy_ci_upper,
    #     roc_auc=roc_auc,
    #     roc_auc_ci_lower=roc_auc_ci_lower,
    #     roc_auc_ci_upper=roc_auc_ci_upper,
    #     aucpr=aucpr,
    #     aucpr_ci_lower=aucpr_ci_lower,
    #     aucpr_ci_upper=aucpr_ci_upper,
    #     f1=f1,
    #     f1_ci_lower=f1_ci_lower,
    #     f1_ci_upper=f1_ci_upper,
    #     y_test=y_test,
    #     y_pred=y_pred,
    #     cm=cm,
    # )
    #
    # # Save the model and scaler
    # joblib.dump(best_model, f"{model_output_dir}/{model_name}_classifier.joblib")
    # joblib.dump(scaler, f"{model_output_dir}/{model_name}_scaler.joblib")
    #
    # # Plot ROC curve
    # plot_tabpfn_roc_curve(
    #     y_test=y_test,
    #     y_pred_proba=y_pred_proba,
    #     model_name=model_name,
    #     outcome=outcome,
    #     output_dir=output_dir,
    # )
    #
    # # Plot Precision-Recall curve
    # plot_tabpfn_pr_curve(
    #     y_test=y_test,
    #     y_pred_proba=y_pred_proba,
    #     model_name=model_name,
    #     outcome=outcome,
    #     output_dir=output_dir,
    # )
    #
    # # Try to extract feature importance if available (for RF wrapper)
    # if hasattr(best_model, "feature_importances_"):
    #     plot_tabpfn_feature_importance(
    #         model=best_model,
    #         feature_names=X_train.columns,
    #         model_name=model_name,
    #         output_dir=output_dir,
    #     )
    #
    # # Create results object
    # from biobank_classification import ClassificationResults
    #
    # results = ClassificationResults(
    #     model="tabpfn",
    #     parameters=best_params,
    #     auc=roc_auc,
    #     auc_lower_ci=roc_auc_ci_lower,
    #     auc_upper_ci=roc_auc_ci_upper,
    #     aucpr=aucpr,
    #     aucpr_lower_ci=aucpr_ci_lower,
    #     aucpr_upper_ci=aucpr_ci_upper,
    #     f1=f1,
    #     f1_lower_ci=f1_ci_lower,
    #     f1_upper_ci=f1_ci_upper,
    #     accuracy=accuracy,
    #     accuracy_lower_ci=accuracy_ci_lower,
    #     accuracy_upper_ci=accuracy_ci_upper,
    #     training_time=training_time,
    # )
    #
    # if collect_predictions:
    #     return results, y_pred_proba
    # else:
    #     return results, None


def save_tabpfn_results_to_file(
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
    """Save TabPFN model results to a text file."""
    model_dir = f"{output_dir}/tabpfn"
    os.makedirs(model_dir, exist_ok=True)

    with open(f"{model_dir}/{model_name}_results.txt", "w") as f:
        f.write(f"TabPFN {model_name} - {outcome} Classification Results:\n")
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


def plot_tabpfn_roc_curve(
    y_test: pd.Series,
    y_pred_proba: np.ndarray,
    model_name: str,
    outcome: str,
    output_dir: str,
) -> None:
    """Plot ROC curve for a TabPFN model."""
    # Ensure model type subdirectory exists
    model_dir = f"{output_dir}/tabpfn"
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
    plt.title(f"TabPFN {model_name} - ROC Curve for {outcome} Detection")
    plt.legend(loc="lower right")
    plt.savefig(f"{model_dir}/{model_name}_roc_curve.png")
    plt.close()


def plot_tabpfn_pr_curve(
    y_test: pd.Series,
    y_pred_proba: np.ndarray,
    model_name: str,
    outcome: str,
    output_dir: str,
) -> None:
    """Plot Precision-Recall curve for a TabPFN model."""
    # Ensure model type subdirectory exists
    model_dir = f"{output_dir}/tabpfn"
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
    plt.title(f"TabPFN {model_name} - Precision-Recall Curve for {outcome} Detection")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{model_dir}/{model_name}_pr_curve.png")
    plt.close()


def plot_tabpfn_feature_importance(
    model: Any,
    feature_names: pd.Index,
    model_name: str,
    output_dir: str,
) -> None:
    """
    Plot feature importance for a TabPFN model or its RandomForest wrapper.
    Only works if the model has feature_importances_ attribute.
    """
    model_dir = f"{output_dir}/tabpfn"
    os.makedirs(model_dir, exist_ok=True)

    if not hasattr(model, "feature_importances_"):
        print("Feature importance not available for this TabPFN model variant.")
        return

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

    # Plot top 20 features or all if less than 20
    num_features = min(20, len(feature_names))
    plt.figure(figsize=(12, 8))
    sorted_idx = np.argsort(feature_importance)[::-1]
    top_features = sorted_idx[:num_features]

    plt.barh(range(len(top_features)), feature_importance[top_features], align="center")
    plt.yticks(range(len(top_features)), [feature_names[i] for i in top_features])
    plt.xlabel("Feature Importance")
    plt.title(f"TabPFN {model_name} - Top {num_features} Features")
    plt.tight_layout()
    plt.savefig(f"{model_dir}/{model_name}_feature_importance.png")
    plt.close()


def run_tabpfn_experiment(
    experiment_config: "ExperimentConfig",
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    outcome: str,
    output_dir: str,
    handle_imbalance: bool = False,
) -> Tuple["ClassificationResults", np.ndarray]:
    """Run TabPFN experiment.

    Args:
        experiment_config: Configuration for the experiment
        X_train, X_test, y_train, y_test: Data splits
        outcome: Outcome variable name
        output_dir: Directory to save results
        handle_imbalance: Whether to apply class weighting for imbalanced datasets

    Returns:
        Tuple of (ClassificationResults, prediction_probabilities)
    """
    print(
        f"\n--- Running TabPFN Experiment {experiment_config.name}: {experiment_config.description} ---"
    )

    # Select features for this experiment
    X_train_exp = X_train[experiment_config.feature_columns]
    X_test_exp = X_test[experiment_config.feature_columns]

    # Train and evaluate
    results, y_pred_proba = train_and_evaluate_tabpfn_model(
        X_train_exp,
        X_test_exp,
        y_train,
        y_test,
        model_name=experiment_config.name,
        outcome=outcome,
        output_dir=output_dir,
        collect_predictions=True,
        handle_imbalance=handle_imbalance,
    )

    return results, y_pred_proba


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


def main_with_tabpfn() -> None:
    """Main function to run experiments with TabPFN support."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load Configuration from YAML file")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="model_config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--handle_imbalance",
        action="store_true",
        help="Enable class weighting for imbalanced datasets",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_yaml_config(args.config)
    model = config["model"]
    outcome = config["outcome"]
    results_dir = config["results_directory"]
    handle_imbalance = args.handle_imbalance or config.get("handle_imbalance", False)

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

    print("Creating embedding dataframe")
    # Create embedding DataFrame
    embedding_df = get_embedding_df(embeddings)
    print("Removing highly correlated features")
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
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        all_features, target, test_size=0.2, random_state=42, stratify=target
    )

    # Setup experiments
    embedding_columns = [col for col in embedding_df.columns if col != outcome]
    experiments = setup_experiments(embedding_columns)

    # Run experiments
    results = {}
    model_predictions = {}

    # Handle both TabPFN and other model types
    if model.lower() == "tabpfn":
        for exp_key, exp_config in tqdm(experiments.items()):
            results[exp_key], model_predictions[exp_key] = run_tabpfn_experiment(
                experiment_config=exp_config,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                outcome=outcome,
                output_dir=outcome_dir,
                handle_imbalance=handle_imbalance,
            )
    else:
        # Use existing code for other model types
        from biobank_classification import run_experiment

        for exp_key, exp_config in tqdm(experiments.items()):
            results[exp_key], model_predictions[exp_key] = run_experiment(
                experiment_config=exp_config,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                model_type=model,
                outcome=outcome,
                output_dir=outcome_dir,
                handle_imbalance=handle_imbalance,
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
    create_summary(results, outcome_dir, model)

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
    main_with_tabpfn()
