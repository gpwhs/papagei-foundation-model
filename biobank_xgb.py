import numpy as np
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import os
import pandas as pd
import torch
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_ecg._preprocessors import Normalize
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from models.resnet import ResNet1DMoE
import time

from typing import Dict, List, Union, Optional, Any, Set
import xgboost as xgb


def load_model_without_module_prefix(model, checkpoint_path):
    """Load model weights without module prefix"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith("module."):
            new_key = k[7:]  # Remove `module.` prefix
        else:
            new_key = k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    return model


def process_signals(df, source_fs=250, target_fs=125, target_length=1250):
    """
    Preprocess signals to match the model's expected input format

    Args:
        df: DataFrame with 'ppg' column containing single heartbeat signals
        source_fs: Original sampling frequency of the signals (250Hz)
        target_fs: Target sampling frequency for the model (125Hz)
        target_length: Target length for the model (1250 samples)

    Returns:
        List of processed signals ready for feature extraction
    """
    processed_signals = []
    norm = Normalize(method="z-score")

    for idx, row in tqdm(df.iterrows(), desc="Processing signals"):
        signal = row["ppg"]

        # Apply z-score normalization
        try:
            signal, _ = norm.apply(signal, fs=source_fs)
        except Exception as e:
            print(f"Error normalizing signal at index {idx}: {e}")
            # Use a simple normalization as fallback
            if np.std(signal) > 0:
                signal = (signal - np.mean(signal)) / np.std(signal)

        # Resample to target frequency if needed
        if source_fs != target_fs:
            # Calculate the number of samples after resampling
            new_length = int(len(signal) * (target_fs / source_fs))
            resampled_signal = np.interp(
                np.linspace(0, len(signal) - 1, new_length),
                np.arange(len(signal)),
                signal,
            )
            signal = resampled_signal

        # Pad signal to reach target length (single heartbeat will be much shorter than 10 seconds)
        if len(signal) < target_length:
            # Center the heartbeat in the padded signal
            padding = target_length - len(signal)
            pad_left = padding // 2
            pad_right = padding - pad_left
            signal = np.pad(signal, (pad_left, pad_right), mode="constant")
        else:
            # If signal is somehow too long, take the middle portion
            center = len(signal) // 2
            start = center - (target_length // 2)
            end = start + target_length
            signal = signal[start:end]

        processed_signals.append(signal)

    return processed_signals


def extract_features(model, signals, device, target_length=1250):
    """
    Extract features from processed signals using the PaPaGei model

    Args:
        model: Loaded PaPaGei model
        signals: List of processed signals
        device: Device to run inference on ('cpu' or 'cuda')
        target_length: Expected signal length

    Returns:
        Array of embeddings for each signal
    """
    embeddings = []
    model.eval()

    with torch.inference_mode():
        for signal in tqdm(signals, desc="Extracting embeddings"):
            # Double check signal has correct length
            if len(signal) != target_length:
                padding = target_length - len(signal)
                if padding > 0:
                    signal = np.pad(signal, (0, padding), mode="constant")
                else:
                    signal = signal[:target_length]

            # Prepare for model (add batch and channel dimensions)
            signal_tensor = (
                torch.tensor(signal, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            # Get embeddings from model
            outputs = model(signal_tensor)
            embedding = outputs[0].cpu().detach().numpy().squeeze()
            embeddings.append(embedding)

    return np.array(embeddings)


def plot_correlation_matrix(
    df, output_filename="correlation_matrix.png", annot_threshold=50
):
    """
    Plots the correlation matrix.
    If the number of features exceeds annot_threshold, disables annotations for clarity.
    """
    corr = df.corr()
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
    return corr


def high_corr_features(corr_matrix, threshold=0.9):
    """
    Returns a set of features that have an absolute correlation greater than the threshold.
    """
    correlated_features = set()
    for i in tqdm(
        range(len(corr_matrix.columns)), desc="Finding highly correlated features"
    ):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                correlated_features.add(corr_matrix.columns[i])
    return correlated_features


def train_and_evaluate_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
    output_dir: str = "results",
) -> Dict[str, Any]:
    """
    Train an XGBoost model and evaluate it.

    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        model_name: Name of the model for saving outputs
        output_dir: Directory to save results

    Returns:
        dict: Dictionary with model and evaluation metrics
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define parameter distributions for RandomizedSearchCV
    param_distributions = {
        "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
        "n_estimators": [50, 100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 7, 8, 10],
        "min_child_weight": [1, 2, 3, 5, 7, 10],
        "gamma": [0, 0.1, 0.2, 0.3, 0.5, 1.0],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.001, 0.01, 0.1, 1.0],
        "reg_lambda": [0, 0.001, 0.01, 0.1, 1.0],
        "scale_pos_weight": [1, 3, 5, 10],  # For imbalanced datasets
    }

    # Create XGBoost classifier
    base_model = xgb.XGBClassifier(
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",  # For faster training
        verbosity=0,
    )

    # RandomizedSearchCV for parameter tuning
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=60,
        cv=3,
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

    # Evaluate the model
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Print evaluation metrics
    print(f"\n{model_name} - Hypertension Classification Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Save results to files
    with open(f"{output_dir}/{model_name}_results.txt", "w") as f:
        f.write(f"{model_name} - Hypertension Classification Results:\n")
        f.write(f"Best parameters: {random_search.best_params_}\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    # Save the model and scaler
    joblib.dump(best_model, f"{output_dir}/{model_name}_classifier.joblib")
    joblib.dump(scaler, f"{output_dir}/{model_name}_scaler.joblib")

    # Plot ROC curve
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
    plt.title(f"{model_name} - ROC Curve for Hypertension Detection")
    plt.legend(loc="lower right")
    plt.savefig(f"{output_dir}/{model_name}_roc_curve.png")
    plt.close()

    # Feature importance for XGBoost
    feature_importance = best_model.feature_importances_
    feature_names = X_train.columns

    # Create DataFrame for feature importance
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": feature_importance}
    )
    importance_df = importance_df.sort_values(by="importance", ascending=False)

    # Save feature importance to file
    importance_df.to_csv(
        f"{output_dir}/{model_name}_feature_importance.csv", index=False
    )

    # Plot top 20 features or all if less than 20
    num_features = min(20, len(feature_names))
    plt.figure(figsize=(12, 8))

    # Get top features
    top_features = importance_df.head(num_features)

    # Plot in descending order
    plt.barh(
        range(len(top_features)),
        top_features["importance"].values[::-1],
        align="center",
    )
    plt.yticks(range(len(top_features)), top_features["feature"].values[::-1])
    plt.xlabel("Feature Importance")
    plt.title(f"{model_name} - Top {num_features} Features for Hypertension Detection")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_feature_importance.png")
    plt.close()

    # Return results
    return {
        "model": best_model,
        "scaler": scaler,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "best_params": random_search.best_params_,
        "training_time": training_time,
    }


def plot_learning_curves(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_params: Dict[str, Any],
    model_name: str,
    output_dir: str = "results",
) -> None:
    """
    Plot learning curves for the XGBoost model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        best_params: Best parameters from hyperparameter tuning
        model_name: Name of the model for saving outputs
        output_dir: Directory to save results
    """
    # Create training and validation sets
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Prepare parameters
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],  # Add AUC as evaluation metric
        "seed": 42,
    }

    # Add best parameters (removing n_estimators as we'll vary it)
    best_params_copy = best_params.copy()
    if "n_estimators" in best_params_copy:
        del best_params_copy["n_estimators"]

    # Convert parameters to XGBoost format
    for key, value in best_params_copy.items():
        if key in ["use_label_encoder", "verbosity", "n_jobs", "random_state"]:
            continue
        params[key] = value

    # Train with early stopping and learning curves
    evals_result: Dict[str, Dict[str, List[float]]] = {}

    # Use more boosting rounds than in the final model to see learning curve
    max_boost_rounds = 1000

    xgb_model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=max_boost_rounds,
        evals=[(dtrain, "train"), (dtest, "test")],
        evals_result=evals_result,
        verbose_eval=False,
        early_stopping_rounds=50,
    )

    # Plot learning curves
    plt.figure(figsize=(10, 6))

    # Get the number of boosting rounds
    epochs = len(evals_result["train"]["logloss"])
    x_axis = range(0, epochs)

    # Plot log loss
    plt.subplot(1, 2, 1)
    plt.plot(x_axis, evals_result["train"]["logloss"], label="Train")
    plt.plot(x_axis, evals_result["test"]["logloss"], label="Test")
    plt.legend()
    plt.xlabel("Boosting Rounds")
    plt.ylabel("Log Loss")
    plt.title("XGBoost Log Loss")

    # Plot AUC directly from evaluation results
    plt.subplot(1, 2, 2)
    plt.plot(x_axis, evals_result["train"]["auc"], label="Train")
    plt.plot(x_axis, evals_result["test"]["auc"], label="Test")
    plt.legend()
    plt.xlabel("Boosting Rounds")
    plt.ylabel("AUC")
    plt.title("XGBoost AUC")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_learning_curves.png")
    plt.close()


def explain_model_predictions(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    output_dir: str = "results",
    num_samples: int = 5,
) -> None:
    """
    Generate SHAP values to explain model predictions for individual samples.

    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for saving outputs
        output_dir: Directory to save results
        num_samples: Number of random samples to explain
    """
    try:
        import shap

        # Create explainer
        explainer = shap.Explainer(model)

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
            plt.savefig(f"{output_dir}/{model_name}_shap_sample_{i+1}.png")
            plt.close()

        # Create summary plot for all test data
        # Use a sample of 100 instances for better visualization
        sample_size = min(100, len(X_test))
        sample_indices = np.random.choice(len(X_test), size=sample_size, replace=False)
        X_for_summary = X_test.iloc[sample_indices]

        shap_values_summary = explainer(X_for_summary)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_summary, X_for_summary, show=False)
        plt.title(f"{model_name} - SHAP Summary Plot")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_shap_summary.png")
        plt.close()

    except ImportError:
        print("SHAP package not installed. Skipping model explanation.")


def main() -> None:
    """Main function to run the experiments with XGBoost."""
    # Load your data
    print("Loading data...")
    df = pd.read_parquet(
        f"{os.getenv('BIOBANK_DATA_PATH')}/215k_pyppg_features_and_conditions.parquet"
    )
    df = df.dropna()

    # Convert string representations of arrays to numpy arrays if needed
    if isinstance(df["ppg"].iloc[0], str):
        df["ppg"] = df["ppg"].apply(lambda x: np.array(eval(x)))

    # Set up parameters
    source_fs = 250  # Your original sampling frequency
    target_fs = 125  # Model's expected sampling frequency
    target_length = 1250  # Model's expected input length (10 seconds at 125 Hz)

    # Create results directory
    results_dir = "experiment_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Initialize model
    print("Initializing PaPaGei model...")
    model_config = {
        "base_filters": 32,
        "kernel_size": 3,
        "stride": 2,
        "groups": 1,
        "n_block": 18,
        "n_classes": 512,
        "n_experts": 3,
    }

    model = ResNet1DMoE(
        in_channels=1,
        base_filters=model_config["base_filters"],
        kernel_size=model_config["kernel_size"],
        stride=model_config["stride"],
        groups=model_config["groups"],
        n_block=model_config["n_block"],
        n_classes=model_config["n_classes"],
        n_experts=model_config["n_experts"],
    )

    # Load pre-trained weights
    model_path = "weights/papagei_s.pt"
    try:
        model = load_model_without_module_prefix(model, model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)

    # Extract features
    embeddings_file = "embeddings.npy"
    if os.path.exists(embeddings_file):
        print(f"Loading pre-computed embeddings from {embeddings_file}")
        embeddings = np.load(embeddings_file)
    else:
        # Process signals
        print("Processing signals...")
        processed_signals = process_signals(
            df, source_fs=source_fs, target_fs=target_fs, target_length=target_length
        )
        print("Extracting features...")
        embeddings = extract_features(model, processed_signals, device, target_length)
        np.save(embeddings_file, embeddings)
        print(f"Embeddings saved to {embeddings_file}")

    # Create feature DataFrames for the three experiments
    print("Creating feature DataFrames for experiments...")

    # Get column names for embeddings
    embedding_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]

    # Create basic DataFrame with embeddings
    embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)

    # Add target variable
    embedding_df["Hypertension"] = df["Hypertension"].values

    # Add metadata columns
    metadata_cols = []
    for col in ["age", "sex", "BMI"]:
        if col in df.columns:
            embedding_df[col] = df[col].values
            metadata_cols.append(col)

    # Split data for all experiments at once (to ensure same split for all experiments)
    # Separate features and target
    all_features = embedding_df.drop("Hypertension", axis=1)
    target = embedding_df["Hypertension"]

    # Create train/test splits
    X_train, X_test, y_train, y_test = train_test_split(
        all_features, target, test_size=0.2, random_state=42, stratify=target
    )

    # Experiment M0: Only PaPaGei features
    print("\n--- Running Experiment M0: Only PaPaGei features with XGBoost ---")
    X_train_M0 = X_train[embedding_cols]
    X_test_M0 = X_test[embedding_cols]

    # Experiment M1: Only metadata (age, sex, BMI)
    print("\n--- Running Experiment M1: Only metadata (age, sex, BMI) with XGBoost ---")
    X_train_M1 = X_train[metadata_cols]
    X_test_M1 = X_test[metadata_cols]

    # Experiment M2: Both PaPaGei features and metadata
    print(
        "\n--- Running Experiment M2: Both PaPaGei features and metadata with XGBoost ---"
    )
    X_train_M2 = X_train[embedding_cols + metadata_cols]
    X_test_M2 = X_test[embedding_cols + metadata_cols]

    # For each experiment, train and evaluate a model
    results = {}

    # M0: Only PaPaGei features
    results["M0"] = train_and_evaluate_model(
        X_train_M0,
        X_test_M0,
        y_train,
        y_test,
        model_name="M0_XGBoost_PaPaGei_Only",
        output_dir=results_dir,
    )

    # Add learning curves
    plot_learning_curves(
        X_train_M0.values,
        y_train.values,
        X_test_M0.values,
        y_test.values,
        results["M0"]["best_params"],
        model_name="M0_XGBoost_PaPaGei_Only",
        output_dir=results_dir,
    )

    # Add model explanations
    explain_model_predictions(
        results["M0"]["model"],
        X_test_M0,
        y_test,
        model_name="M0_XGBoost_PaPaGei_Only",
        output_dir=results_dir,
    )

    # M1: Only metadata
    results["M1"] = train_and_evaluate_model(
        X_train_M1,
        X_test_M1,
        y_train,
        y_test,
        model_name="M1_XGBoost_Metadata_Only",
        output_dir=results_dir,
    )

    # Add learning curves
    plot_learning_curves(
        X_train_M1.values,
        y_train.values,
        X_test_M1.values,
        y_test.values,
        results["M1"]["best_params"],
        model_name="M1_XGBoost_Metadata_Only",
        output_dir=results_dir,
    )

    # Add model explanations
    explain_model_predictions(
        results["M1"]["model"],
        X_test_M1,
        y_test,
        model_name="M1_XGBoost_Metadata_Only",
        output_dir=results_dir,
    )

    # M2: Both PaPaGei features and metadata
    results["M2"] = train_and_evaluate_model(
        X_train_M2,
        X_test_M2,
        y_train,
        y_test,
        model_name="M2_XGBoost_PaPaGei_And_Metadata",
        output_dir=results_dir,
    )

    # Add learning curves
    plot_learning_curves(
        X_train_M2.values,
        y_train.values,
        X_test_M2.values,
        y_test.values,
        results["M2"]["best_params"],
        model_name="M2_XGBoost_PaPaGei_And_Metadata",
        output_dir=results_dir,
    )

    # Add model explanations
    explain_model_predictions(
        results["M2"]["model"],
        X_test_M2,
        y_test,
        model_name="M2_XGBoost_PaPaGei_And_Metadata",
        output_dir=results_dir,
    )

    # Create summary of results
    summary = pd.DataFrame(
        {
            "Model": [
                "M0: XGBoost PaPaGei Only",
                "M1: XGBoost Metadata Only",
                "M2: XGBoost PaPaGei + Metadata",
            ],
            "Accuracy": [
                results["M0"]["accuracy"],
                results["M1"]["accuracy"],
                results["M2"]["accuracy"],
            ],
            "ROC_AUC": [
                results["M0"]["roc_auc"],
                results["M1"]["roc_auc"],
                results["M2"]["roc_auc"],
            ],
            "Training_Time": [
                results["M0"]["training_time"],
                results["M1"]["training_time"],
                results["M2"]["training_time"],
            ],
        }
    )

    summary.to_csv(f"{results_dir}/experiment_summary.csv", index=False)
    print("\nExperiment Summary:")
    print(summary)

    # Plot results comparison
    plt.figure(figsize=(10, 6))

    # Bar plot for accuracy
    plt.subplot(1, 2, 1)
    plt.bar(summary["Model"], summary["Accuracy"], color="blue")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Bar plot for ROC AUC
    plt.subplot(1, 2, 2)
    plt.bar(summary["Model"], summary["ROC_AUC"], color="orange")
    plt.ylabel("ROC AUC")
    plt.title("ROC AUC Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(f"{results_dir}/experiment_comparison.png")
    plt.close()

    print(f"All experiments completed successfully! Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
