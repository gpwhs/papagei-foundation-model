import os
import pandas as pd
from torch_ecg._preprocessors import Normalize
import numpy as np
import torch
from tqdm import tqdm
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


def extract_features(df):
    """
    Extract features from processed signals using the PaPaGei model

    Args:
        df: DataFrame with 'ppg' column containing single heartbeat signals

    Returns:
        Array of embeddings for each signal
    """
    embeddings_file = "embeddings.npy"
    if os.path.exists(embeddings_file):
        return np.load(embeddings_file)

    # Set up parameters
    source_fs = 250  # Your original sampling frequency
    target_fs = 125  # Model's expected sampling frequency
    target_length = 1250  # Model's expected input length (10 seconds at 125 Hz)

    # Create results directory
    results_dir = "experiment_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

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

    np.save(embeddings_file, embeddings)
    return np.array(embeddings)


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


def get_embedding_df(embeddings: np.ndarray) -> pd.DataFrame:
    """
    Get the column names for the embeddings.
    """
    embedding_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    return pd.DataFrame(embeddings, columns=embedding_cols)
