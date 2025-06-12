#!/usr/bin/env python
"""
Quick and dirty TabPFN test for biobank data
Tests all 5 experiment configurations (M0-M4)
"""
from tqdm import tqdm
from sklearn.decomposition import PCA
from torch_ecg._preprocessors import Normalize
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from tabpfn import TabPFNClassifier
import time
import warnings
import sys
from typing import Optional
import torch


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
        df: DataFrame with 'ppg_resampled' column containing single heartbeat signals
        source_fs: Original sampling frequency of the signals (250Hz)
        target_fs: Target sampling frequency for the model (125Hz)
        target_length: Target length for the model (1250 samples)

    Returns:
        List of processed signals ready for feature extraction
    """
    processed_signals = []
    norm = Normalize(method="z-score")
    df = df.dropna()

    for idx, row in tqdm(df.iterrows(), desc="Processing signals"):
        signal = row["ppg_resampled"]

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


def extract_features(df: pd.DataFrame, embeddings_file: str):
    """
    Extract features from processed signals using the PaPaGei model

    Args:
        df: DataFrame with 'ppg' column containing single heartbeat signals

    Returns:
        Array of embeddings for each signal
    """

    # Set up parameters
    source_fs = 250  # Your original sampling frequency
    target_fs = 125  # Model's expected sampling frequency
    target_length = 1250  # Model's expected input length (10 seconds at 125 Hz)
    print(f"embedding file: {embeddings_file}")

    # Create results directory
    results_dir = "./experiment_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    print(f"Column 'resampled_ppg' exists: {df['ppg_resampled'].notnull().all()}")

    processed_signals = process_signals(df, source_fs, target_fs, target_length)
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
    embeddings = []
    model.eval()

    with torch.inference_mode():
        for signal in tqdm(processed_signals, desc="Extracting embeddings"):
            # Double check signal has correct length
            if len(signal) != target_length:
                padding = target_length - len(signal)
                if padding > 0:
                    signal = np.pad(signal, (0, padding), mode="constant")
                else:
                    signal = signal[
                        :target_length
                    ]  # Prepare for model (add batch and channel dimensions)
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

    np.save(embeddings_file, embeddings)
    return np.array(embeddings)


def get_embeddings(df: pd.DataFrame, cache_file: str) -> np.ndarray:
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
    embeddings = extract_features(df, cache_file)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
    np.save(cache_file, embeddings)
    print(f"Embeddings saved to {cache_file}")

    return embeddings


def get_embedding_df(
    df: pd.DataFrame,
    outcome: str,
    outcome_time: Optional[str] = None,
) -> pd.DataFrame:
    """
    Get the column names for the embeddings.
    """
    if isinstance(df["ppg_resampled"].iloc[0], str):
        df["ppg_resampled"] = df["ppg_resampled"].apply(lambda x: np.array(eval(x)))
    if outcome_time is not None:
        embeddings_file = f"embeddings_{outcome}_survival.npy"
    else:
        embeddings_file = f"embeddings_{outcome}.npy"  # needed since we do PCA
    embeddings = get_embeddings(df, cache_file=embeddings_file)
    embedding_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)
    if outcome not in embedding_df.columns:
        embedding_df[outcome] = df[outcome]
    print("Applying PCA to embeddings...")
    embedding_df = apply_pca_to_embeddings(embedding_df, outcome)
    if outcome_time is not None:
        embedding_df[outcome_time] = df[outcome_time]
    print(f"Embedding DataFrame shape: {embedding_df.shape}")

    return embedding_df


def apply_pca_to_embeddings(embedding_df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """
    Apply PCA to the embeddings and return the new DataFrame.

    Args:
        embedding_df: DataFrame containing the embeddings
        outcome: Name of the outcome variable

    Returns:
        DataFrame with PCA-transformed embeddings
    """
    # Extract original embedding columns (excluding outcome)
    original_embedding_columns = [col for col in embedding_df.columns if col != outcome]
    pca = PCA(n_components=0.99)  # Retain 95% of variance; adjust as needed
    embedding_transformed = pca.fit_transform(embedding_df[original_embedding_columns])
    # Create a new DataFrame with PCA features
    pca_columns = [f"pca_{i}" for i in range(embedding_transformed.shape[1])]
    embedding_df_pca = pd.DataFrame(
        embedding_transformed, columns=pca_columns, index=embedding_df.index
    )
    # Add outcome column back to the PCA-transformed embedding DataFrame
    embedding_df_pca[outcome] = embedding_df[outcome].values

    return embedding_df_pca


def get_pyppg_features(outcome):
    with open(f"final_pyppg_feature_columns_{outcome}.txt", "r") as f:
        final_features = f.read().splitlines()
    return final_features


warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# --- 1. Load Biobank Data ---
print("Loading biobank data...")
data_path = os.getenv("BIOBANK_DATA_PATH")
if not data_path:
    raise ValueError("BIOBANK_DATA_PATH environment variable is not set")

# Load the parquet file
df = pd.read_parquet(f"{data_path}/250k_waves_conditions_pyppg_first_cleaned.parquet")
print(f"Loaded {len(df)} samples")

# Pick an outcome to test
OUTCOME = "Hypertension"  # Change this to test other outcomes
print(f"Testing outcome: {OUTCOME}")

# --- 2. Get Embeddings ---
print("Extracting PaPaGei embeddings...")

# Get embeddings (this might take a moment)
embedding_df = get_embedding_df(df, OUTCOME)
embedding_columns = [col for col in embedding_df.columns if col != OUTCOME]
print(f"Embedding dimensions: {len(embedding_columns)}")

# --- 3. Setup Feature Sets ---
traditional_features = ["age", "sex", "BMI"]

# Get pyPPG features for this outcome

pyppg_features = get_pyppg_features(OUTCOME)
print(f"PyPPG features: {len(pyppg_features)}")

# --- 4. Prepare All Features ---
all_features = pd.concat(
    [embedding_df[embedding_columns], df[traditional_features], df[pyppg_features]],
    axis=1,
)

target = df[OUTCOME]

# Remove any rows with NaN
mask = ~(all_features.isna().any(axis=1) | target.isna())
all_features = all_features[mask]
target = target[mask]
print(f"Clean samples: {len(target)}")

# Class distribution
print(f"\nClass distribution:")
print(target.value_counts())
print(f"Class ratio: {target.value_counts().min() / target.value_counts().max():.3f}")

# --- 5. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    all_features, target, test_size=0.2, random_state=42, stratify=target
)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# --- 6. Define Experiments ---
experiments = {
    "M0_PaPaGei_Only": embedding_columns,
    "M1_Traditional_Only": traditional_features,
    "M2_PaPaGei_Traditional": embedding_columns + traditional_features,
    "M3_pyPPG_Only": pyppg_features,
    "M4_pyPPG_Traditional": pyppg_features + traditional_features,
}

# --- 7. Run TabPFN on Each Experiment ---
print("\n" + "=" * 60)
print("Running TabPFN experiments...")
print("=" * 60)

results = {}

for exp_name, feature_cols in experiments.items():
    print(f"\n--- {exp_name} ---")
    print(f"Features: {len(feature_cols)}")

    # Select features
    X_train_exp = X_train[feature_cols]
    X_test_exp = X_test[feature_cols]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_exp)
    X_test_scaled = scaler.transform(X_test_exp)

    # Sample if too large
    if len(X_train_scaled) > 10000:
        print(f"Sampling 10k from {len(X_train_scaled)} training samples...")
        idx = np.random.choice(len(X_train_scaled), 10000, replace=False)
        X_train_scaled = X_train_scaled[idx]
        y_train_sample = y_train.iloc[idx]
    else:
        y_train_sample = y_train

    # Limit features if needed
    if X_train_scaled.shape[1] > 100:
        print(f"Reducing features from {X_train_scaled.shape[1]} to 100...")
        X_train_scaled = X_train_scaled[:, :100]
        X_test_scaled = X_test_scaled[:, :100]

    # Train TabPFN
    try:
        start_time = time.time()

        tabpfn = TabPFNClassifier(
            device="cpu",
            N_ensemble_configurations=32,
            n_estimators=16,  # Reduce for speed
        )

        tabpfn.fit(X_train_scaled, y_train_sample)
        train_time = time.time() - start_time

        # Predict
        y_pred = tabpfn.predict(X_test_scaled)
        y_pred_proba = tabpfn.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        results[exp_name] = {
            "accuracy": acc,
            "auc": auc,
            "train_time": train_time,
            "n_features": len(feature_cols),
            "n_train": len(y_train_sample),
        }

        print(f"✓ Accuracy: {acc:.4f}")
        print(f"✓ ROC AUC: {auc:.4f}")
        print(f"✓ Time: {train_time:.1f}s")

    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        results[exp_name] = {"error": str(e)}

# --- 8. Summary ---
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# Sort by AUC
successful_results = {k: v for k, v in results.items() if "auc" in v}
sorted_results = sorted(
    successful_results.items(), key=lambda x: x[1]["auc"], reverse=True
)

print(f"\n{'Experiment':<25} {'AUC':<8} {'Accuracy':<10} {'Features':<10} {'Time':<8}")
print("-" * 65)

for exp_name, metrics in sorted_results:
    print(
        f"{exp_name:<25} {metrics['auc']:<8.4f} {metrics['accuracy']:<10.4f} "
        f"{metrics['n_features']:<10} {metrics['train_time']:<8.1f}s"
    )

# Best model
if sorted_results:
    best_exp = sorted_results[0][0]
    best_auc = sorted_results[0][1]["auc"]
    print(f"\nBest model: {best_exp} (AUC: {best_auc:.4f})")

# --- 9. Quick comparison with baseline ---
print("\n" + "=" * 60)
print("BASELINE COMPARISON (Logistic Regression)")
print("=" * 60)

from sklearn.linear_model import LogisticRegression

# Just test on M2 (PaPaGei + Traditional)
X_train_m2 = X_train[experiments["M2_PaPaGei_Traditional"]]
X_test_m2 = X_test[experiments["M2_PaPaGei_Traditional"]]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_m2)
X_test_scaled = scaler.transform(X_test_m2)

# Limit features for LR too
if X_train_scaled.shape[1] > 100:
    X_train_scaled = X_train_scaled[:, :100]
    X_test_scaled = X_test_scaled[:, :100]

lr = LogisticRegression(max_iter=1000, random_state=42)
start_time = time.time()
lr.fit(X_train_scaled, y_train)
lr_time = time.time() - start_time

y_pred_lr = lr.predict(X_test_scaled)
y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

lr_acc = accuracy_score(y_test, y_pred_lr)
lr_auc = roc_auc_score(y_test, y_pred_proba_lr)

print(f"Logistic Regression (M2):")
print(f"  Accuracy: {lr_acc:.4f}")
print(f"  ROC AUC: {lr_auc:.4f}")
print(f"  Time: {lr_time:.1f}s")

if "M2_PaPaGei_Traditional" in successful_results:
    tabpfn_m2 = successful_results["M2_PaPaGei_Traditional"]
    print(f"\nTabPFN vs LR on M2:")
    print(f"  AUC improvement: {tabpfn_m2['auc'] - lr_auc:+.4f}")
    print(f"  Speed ratio: {lr_time / tabpfn_m2['train_time']:.1f}x")

print("\n✅ TabPFN test complete!")
