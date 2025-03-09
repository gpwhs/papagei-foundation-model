import os
import time
import argparse
from typing import Dict, List, Tuple, Any

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
        X: Feature DataFrame
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


def train_and_evaluate_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_type: str,
    model_name: str,
    outcome: str,
    output_dir: str,
) -> ClassificationResults:
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

    Returns:
        ClassificationResults with model and evaluation metrics
    """
    # Ensure output directory exists
    model_output_dir = f"{output_dir}/{model_type}"
    os.makedirs(model_output_dir, exist_ok=True)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Set up model and hyperparameter search space
    model, parameter_distributions = setup_model(ModelTypes[model_type])

    # RandomizedSearchCV for parameter tuning
    random_search = RandomizedSearchCV(
        model,
        param_distributions=parameter_distributions,
        n_iter=60,
        cv=1,
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

    # Print evaluation metrics
    print(f"\n{model_name} - {outcome} Classification Results:")
    print(
        f"Accuracy: {accuracy:.4f} (CI: {accuracy_ci_lower:.4f}-{accuracy_ci_upper:.4f})"
    )
    print(f"ROC AUC: {roc_auc:.4f} (CI: {roc_auc_ci_lower:.4f}-{roc_auc_ci_upper:.4f})")
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
        f1=f1,
        f1_ci_lower=f1_ci_lower,
        f1_ci_upper=f1_ci_upper,
        y_test=y_test,
        y_pred=y_pred,
        cm=cm,
    )

    # Save the model and scaler
    joblib.dump(best_model, f"{output_dir}/{model_type}_{model_name}_classifier.joblib")
    joblib.dump(scaler, f"{output_dir}/{model_type}_{model_name}_scaler.joblib")

    # Plot ROC curve
    plot_roc_curve(
        y_test=y_test,
        y_pred_proba=y_pred_proba,
        model_type=model_type,
        model_name=model_name,
        outcome=outcome,
        output_dir=output_dir,
    )

    # Feature importance (if applicable)
    if model_type == "LR":
        save_feature_importance(
            model=best_model,
            feature_names=X_train.columns,
            model_type=model_type,
            model_name=model_name,
            output_dir=output_dir,
        )

    # Return results
    return ClassificationResults(
        model=model_type,
        parameters=random_search.best_params_,
        auc=roc_auc,
        auc_lower_ci=roc_auc_ci_lower,
        auc_upper_ci=roc_auc_ci_upper,
        f1=f1,
        f1_lower_ci=f1_ci_lower,
        f1_upper_ci=f1_ci_upper,
        accuracy=accuracy,
        accuracy_lower_ci=accuracy_ci_lower,
        accuracy_upper_ci=accuracy_ci_upper,
        training_time=training_time,
    )


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
    plt.savefig(f"{output_dir}/{model_type}_{model_name}_roc_curve.png")
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
        model: Trained model with coefficients
        feature_names: Names of features
        model_type: Type of model
        model_name: Name of model
        output_dir: Directory to save results
    """
    feature_importance = np.abs(model.coef_[0])

    # Create DataFrame for feature importance
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": model.coef_[0]})
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values(by="abs_coefficient", ascending=False)

    # Save top features to file
    coef_df.to_csv(
        f"{output_dir}/{model_type}_{model_name}_feature_importance.csv",
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
    plt.title(f"{model_type} {model_name} - Top {num_features} Features for Detection")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_type}_{model_name}_feature_importance.png")
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
) -> ClassificationResults:
    """Run a single experiment.

    Args:
        experiment_config: Configuration for the experiment
        X_train, X_test, y_train, y_test: Data splits
        model_type: Type of model to use
        outcome: Outcome variable name
        output_dir: Directory to save results

    Returns:
        Results of the experiment
    """
    print(
        f"\n--- Running Experiment {experiment_config.name}: {experiment_config.description} ---"
    )

    # Select features for this experiment
    X_train_exp = X_train[experiment_config.feature_columns]
    X_test_exp = X_test[experiment_config.feature_columns]

    # Train and evaluate
    return train_and_evaluate_model(
        X_train_exp,
        X_test_exp,
        y_train,
        y_test,
        model_type=model_type,
        model_name=experiment_config.name,
        outcome=outcome,
        output_dir=output_dir,
    )


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
    print(f"Running experiments with model {model} for outcome {outcome}")

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/{model}", exist_ok=True)

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

    # Create train/test splits
    X_train, X_test, y_train, y_test = train_test_split(
        all_features, target, test_size=0.2, random_state=42, stratify=target
    )

    # Setup experiments
    embedding_columns = [col for col in embedding_df.columns if col != outcome]
    experiments = setup_experiments(embedding_columns)

    # Run experiments
    results = {}
    for exp_key, exp_config in tqdm(experiments.items()):
        results[exp_key] = run_experiment(
            experiment_config=exp_config,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model_type=model,
            outcome=outcome,
            output_dir=results_dir,
        )

    # Save results
    with open(f"{results_dir}/experiment_results_{model}.json", "w") as f:
        import json

        f.write(json.dumps({k: v.dict() for k, v in results.items()}, indent=2))

    # Create summary of results
    create_summary(results, results_dir, model)
    print(f"All experiments completed successfully! Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
