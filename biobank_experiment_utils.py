from sklearn.decomposition import PCA
from typing import Dict, Optional, Tuple
import yaml
import numpy as np
import pandas as pd
from biobank_embeddings_extraction import get_embeddings

import os
from biobank_experiment_constants import FULL_PYPPG_FEATURES
from biobank_feature_functions import (
    remove_high_vif_features,
    remove_highly_correlated_features,
)


def preprocess_classification_data(
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


def load_yaml_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_embedding_df(
    df: pd.DataFrame,
    embeddings_file: str = "embeddings_mimic.npy",
    ppg_column: str = "ppg_template",
    # outcome: str,
    # outcome_time: Optional[str] = None,
) -> pd.DataFrame:
    """
    Get the column names for the embeddings.:Warning:warning::w
    """
    if isinstance(df[ppg_column].iloc[0], str):
        df[ppg_column] = df[ppg_column].apply(lambda x: np.array(eval(x)))
    # if outcome_time is not None:
    #     embeddings_file = f"embeddings_{outcome}_survival.npy"
    # else:
    #     embeddings_file = f"embeddings_{outcome}.npy"  # needed since we do PCA
    embeddings = get_embeddings(df, cache_file=embeddings_file)
    embedding_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)
    # if outcome not in embedding_df.columns:
    #     embedding_df[outcome] = df[outcome]
    print("Applying PCA to embeddings...")
    embedding_df = apply_pca_to_embeddings(embedding_df)
    # if outcome_time is not None:
    #     embedding_df[outcome_time] = df[outcome_time]
    # print(f"Embedding DataFrame shape: {embedding_df.shape}")

    return embedding_df


def apply_pca_to_embeddings(embedding_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply PCA to the embeddings and return the new DataFrame.

    Args:
        embedding_df: DataFrame containing the embeddings
        outcome: Name of the outcome variable

    Returns:
        DataFrame with PCA-transformed embeddings
    """
    # Extract original embedding columns (excluding outcome)
    original_embedding_columns = [col for col in embedding_df.columns]
    pca = PCA(n_components=0.99)  # Retain 99% of variance; adjust as needed
    embedding_transformed = pca.fit_transform(embedding_df[original_embedding_columns])
    # Create a new DataFrame with PCA features
    pca_columns = [f"pca_{i}" for i in range(embedding_transformed.shape[1])]
    embedding_df_pca = pd.DataFrame(
        embedding_transformed, columns=pca_columns, index=embedding_df.index
    )
    # Add outcome column back to the PCA-transformed embedding DataFrame

    return embedding_df_pca


def compute_expected_calibration_error(
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


def check_for_imbalance(target: pd.Series, outcome: str) -> None:
    # Print class distribution information for the outcome
    class_counts = target.value_counts()

    class_ratio = class_counts.min() / class_counts.max()

    if class_ratio < 0.3:
        print(f"\nOutcome ({outcome}) class distribution:")
        for class_label, count in class_counts.items():
            print(f"  Class {class_label}: {count} ({count/len(target):.2%})")
        print(f"Class ratio (minority/majority): {class_ratio:.3f}")
        print(
            "Warning: Dataset appears imbalanced. Consider using the handle_imbalance flag."
        )
