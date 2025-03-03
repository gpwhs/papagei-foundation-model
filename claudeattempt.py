import numpy as np
import seaborn as sns
import os
import pandas as pd
import torch
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_ecg._preprocessors import Normalize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from models.resnet import ResNet1DMoE


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

    for idx, row in tqdm(df.iterrows()):
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
        for signal in tqdm(signals):
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


def main():
    # Load your data
    print("Loading data...")
    df = pd.read_parquet(
        "./data/215k_pyppg_features_and_conditions.parquet"
    )  # Replace with your actual data path

    df = df.dropna()
    # Convert string representations of arrays to numpy arrays if needed
    if isinstance(df["ppg"].iloc[0], str):
        df["ppg"] = df["ppg"].apply(lambda x: np.array(eval(x)))

    # Set up parameters
    source_fs = 250  # Your original sampling frequency
    target_fs = 125  # Model's expected sampling frequency
    target_length = 1250  # Model's expected input length (10 seconds at 125 Hz)

    # Initialize model
    print("Initializing model...")
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
    model_path = "weights/papagei_s.pt"  # Replace with your actual path
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
    if os.path.exists("embeddings.npy"):
        embeddings = np.load("embeddings.npy")
    else:
        # Process signals
        print("Processing signals...")
        processed_signals = process_signals(
            df, source_fs=source_fs, target_fs=target_fs, target_length=target_length
        )
        print("Extracting features...")
        embeddings = extract_features(model, processed_signals, device, target_length)
        np.save("embeddings.npy", embeddings)

    # Create features DataFrame
    print("Creating features DataFrame...")
    embedding_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    features_df = pd.DataFrame(embeddings, columns=embedding_cols)
    features_df["Hypertension"] = df["Hypertension"].values

    # Add metadata columns if available
    for col in ["age", "sex", "BMI"]:
        if col in df.columns:
            features_df[col] = df[col].values

    # ---------- Correlation Matrix and High-Correlation Check ----------
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

    corr = plot_correlation_matrix(features_df)

    def high_corr_features(corr_matrix, threshold=0.9):
        """
        Returns a set of features that have an absolute correlation greater than the threshold.
        """
        correlated_features = set()
        for i in tqdm(range(len(corr_matrix.columns))):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    correlated_features.add(corr_matrix.columns[i])
        return correlated_features

    high_corr = high_corr_features(corr, threshold=0.9)
    print("Highly correlated features (correlation > 0.9):", high_corr)

    # ---------- Data Preparation ----------
    print("Preparing data for classification...")
    X = features_df.drop("Hypertension", axis=1)
    y = features_df["Hypertension"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------- Logistic Regression Model Training ----------
    print("Training logistic regression model...")
    param_grid = [
        {
            "C": np.logspace(-3, 2, 6),
            "penalty": ["l1"],
            "solver": ["liblinear"],
            "class_weight": ["balanced", None],
        },
        {
            "C": np.logspace(-3, 2, 6),
            "penalty": ["l2"],
            "solver": ["liblinear", "lbfgs", "saga"],
            "class_weight": ["balanced", None],
        },
        {
            "C": np.logspace(-3, 2, 6),
            "penalty": ["elasticnet"],
            "l1_ratio": [0.1, 0.5, 0.9],
            "solver": ["saga"],
            "class_weight": ["balanced", None],
        },
    ]

    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42, verbose=1),
        param_grid,
        cv=5,
        scoring="roc_auc",
        return_train_score=True,
        n_jobs=-1,
        verbose=3,
    )
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    print("Best hyperparameters:", grid_search.best_params_)

    # ---------- Model Evaluation ----------
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    # ---------- Logistic Regression Coefficient Analysis ----------
    coefs = best_model.coef_.flatten()
    feature_names = X.columns
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values(by="abs_coefficient", ascending=False)
    print("Top 10 features based on logistic regression coefficients:")
    print(coef_df.head(10))

    # Plot top 20 features by absolute coefficient value
    plt.figure(figsize=(12, 6))
    sns.barplot(x="abs_coefficient", y="feature", data=coef_df.head(20))
    plt.title("Top 20 Features by Absolute Coefficient Value")
    plt.tight_layout()
    plt.savefig("top_features.png")
    plt.close()
    print("Top features plot saved to top_features.png")

    print("\nHypertension Classification Results:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Save the model and scaler
    print("Saving model and scaler...")
    joblib.dump(best_model, "Hypertension_classifier.joblib")
    joblib.dump(scaler, "feature_scaler.joblib")

    # Plot ROC curve
    from sklearn.metrics import roc_curve, auc

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
    plt.title("Receiver Operating Characteristic for Hypertension Detection")
    plt.legend(loc="lower right")
    plt.savefig("Hypertension_roc_curve.png")
    plt.close()

    # Feature importance
    if hasattr(best_model, "coef_"):
        # Get feature importance for logistic regression
        feature_importance = np.abs(best_model.coef_[0])
        feature_names = X.columns

        # Sort features by importance
        sorted_idx = np.argsort(feature_importance)[::-1]
        top_features = sorted_idx[:20]  # Top 20 features

        plt.figure(figsize=(12, 8))
        plt.barh(
            range(len(top_features)), feature_importance[top_features], align="center"
        )
        plt.yticks(range(len(top_features)), [feature_names[i] for i in top_features])
        plt.xlabel("Feature Importance")
        plt.title("Top 20 Features for Hypertension Detection")
        plt.tight_layout()
        plt.savefig("Hypertension_feature_importance.png")
        plt.close()

    print("Completed successfully!")


if __name__ == "__main__":
    main()
