from pydantic import BaseModel
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
import os
from typing import Dict, Tuple, List, Any
from scipy.stats import loguniform, uniform
from tqdm import tqdm
import yaml
import numpy as np
import pandas as pd
from enum import Enum
from biobank_experiment_constants import FULL_PYPPG_FEATURES
from biobank_embeddings_extraction import get_embeddings
from biobank_feature_functions import (
    remove_high_vif_features,
    remove_highly_correlated_features,
)


class ModelTypes(Enum):
    """
    Enum for the types of models.
    """

    XGBOOST = "xgboost"
    LOGISTIC_REGRESSION = "LR"
    TABPFN = "tabpfn"
    CATBOOST = "catboost"


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
    f2: float
    f2_lower_ci: float
    f2_upper_ci: float
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


def setup_catboost_model() -> Tuple[object, Dict]:
    """
    Setup CatBoost classifier with hyperparameter search space.

    Returns:
        Tuple[object, Dict]: The model object and hyperparameter search space
    """
    model = CatBoostClassifier(
        verbose=100,  # Reduce verbosity in production
        random_seed=42,
        thread_count=-1,  # Use all available CPU cores
        allow_writing_files=False,  # Disable writing to disk during training
    )

    param_distributions = {
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "depth": [4, 6, 8, 10],
        "l2_leaf_reg": [1, 3, 5, 7, 9],
        "iterations": [100, 200, 300, 500],
        "border_count": [32, 64, 128, 254],
        "bagging_temperature": [0, 1, 10],
        "random_strength": [1, 10, 100],
        # "one_hot_max_size": [2, 10, 25],  # Uncomment if using categorical features
        # Auto class weights for imbalanced datasets
        "auto_class_weights": ["Balanced", None],
    }

    return model, param_distributions


def setup_tabpfn_model() -> Tuple[object, Dict]:
    """
    Setup TabPFN model with RandomForest wrapper for biobank classification.

    Returns:
        Tuple[object, Dict]: The model object and hyperparameter search space
    """
    try:
        from tabpfn_extensions.rf_pfn import RandomForestTabPFNClassifier
        from tabpfn_extensions import TabPFNClassifier

        # Base TabPFN model
        base_model = TabPFNClassifier(device="cpu")

        # RandomForest wrapper for TabPFN
        model = RandomForestTabPFNClassifier(tabpfn=base_model)

        # Hyperparameter search space for RandomForestTabPFNClassifier
        param_distributions = {
            # Base RandomForest parameters
            "n_estimators": [50, 100, 200, 300],
            "max_features": ["sqrt", "log2", None, 0.5, 0.7],
            "max_samples": [0.5, 0.7, 0.9, None],
            "bootstrap": [True, False],
            # TabPFN specific parameters (if exposed by the wrapper)
            "N_ensemble_configurations": [4, 8, 16, 32],
            "mix_method": ["mean", "gmm"],
        }

        return model, param_distributions

    except ImportError:
        raise ImportError("TabPFN extensions not installed. Please install")


def setup_xgboost_model() -> Tuple[object, Dict]:

    from xgboost import XGBClassifier

    model = XGBClassifier()
    param_distributions = {
        "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
        "n_estimators": [50, 100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 7, 8, 10],
        "min_child_weight": [1, 2, 3, 5, 7, 10],
        # "gamma": [0, 0.1, 0.2, 0.3, 0.5, 1.0],
        # "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        # "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        # "reg_alpha": [0, 0.001, 0.01, 0.1, 1.0],
        # "reg_lambda": [0, 0.001, 0.01, 0.1, 1.0],
        # "scale_pos_weight": [1, 3, 5, 10],  # For imbalanced datasets
    }
    return model, param_distributions


def setup_LR_model() -> Tuple[object, Dict]:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000, random_state=42)
    param_distributions = []

    # L1 penalty
    param_distributions.append(
        {
            "C": loguniform(1e-4, 1e3),
            "penalty": ["l1"],
            "solver": ["liblinear"],  # Best solver for L1
            "class_weight": ["balanced", None],
        }
    )

    # L2 penalty
    param_distributions.append(
        {
            "C": loguniform(1e-4, 1e3),
            "penalty": ["l2"],
            "solver": ["lbfgs"],  # Efficient solver for L2
            "class_weight": ["balanced", None],
        }
    )

    # Elasticnet penalty
    param_distributions.append(
        {
            "C": loguniform(1e-4, 1e3),
            "penalty": ["elasticnet"],
            "solver": ["saga"],  # Only solver for elasticnet
            "l1_ratio": uniform(0, 1),
            "class_weight": ["balanced", None],
        }
    )

    # No penalty
    param_distributions.append(
        {
            "C": loguniform(1e-4, 1e3),
            "penalty": [None],
            "solver": ["lbfgs"],  # Efficient when no penalty
            "class_weight": ["balanced", None],
        }
    )
    return model, param_distributions


def setup_model(model_type: ModelTypes) -> Tuple[object, Dict]:
    """
    Setup the model based on the model type.

    Parameters:
    - model_type (ModelTypes): The type of model to setup.

    Returns:
    - Tuple[object, Dict]: The model object and the hyperparameter search space.
    """
    if model_type == ModelTypes.LOGISTIC_REGRESSION:
        return setup_LR_model()

    if model_type == ModelTypes.XGBOOST:
        return setup_xgboost_model()

    if model_type == ModelTypes.TABPFN:
        return setup_tabpfn_model()
    if model_type == ModelTypes.CATBOOST:
        return setup_catboost_model()

    raise ValueError(f"Unsupported model type: {model_type}")


# def remove_highly_corr_features(
#     features: pd.DataFrame, threshold: float = 0.8
# ) -> pd.DataFrame:
#     """
#     Remove highly correlated features from the dataset based on the threshold.
#
#     Parameters:
#     - features (pd.DataFrame): The feature matrix.
#     - threshold (float): The correlation threshold above which one feature is removed.
#
#     Returns:
#     - pd.DataFrame: The filtered DataFrame with highly correlated features removed.
#     """
#     print(f"Calculating correlation matrix for {features.shape[1]} features...")
#     if os.path.exists("filtered_embeddings.parquet"):
#         print("Loading correlation filtered embeddings from disk")
#         return pd.read_parquet("filtered_embeddings.parquet")
#     corr = features.corr().abs()
#
#     upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
#
#     # Create a list to collect columns to drop
#     to_drop = []
#
#     # This is the time-consuming part, so wrap it in tqdm
#     for column in tqdm(upper.columns, desc="Filtering Correlated Features"):
#         # Check if this column has correlation above threshold with any other column
#         if any(upper[column] > threshold):
#             to_drop.append(column)
#
#     print(f"Dropping {len(to_drop)} highly correlated features")
#     filtered_features = features.drop(columns=to_drop, inplace=False)
#     filtered_features.to_parquet("filtered_embeddings.parquet")
#
#     return filtered_features


def load_yaml_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_embedding_df(
    df: pd.DataFrame,
    outcome: str,
    embeddings_file: str = "embeddings.npy",
) -> pd.DataFrame:
    """
    Get the column names for the embeddings.
    """
    if isinstance(df["ppg_resampled"].iloc[0], str):
        df["ppg_resampled"] = df["ppg_resampled"].apply(lambda x: np.array(eval(x)))
    embeddings = get_embeddings(df, cache_file=embeddings_file)
    embedding_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)
    # embedding_df = remove_highly_corr_features(embedding_df) # don't think this is necessary with PCA?
    if outcome not in embedding_df.columns:
        embedding_df[outcome] = df[outcome]
    print("Applying PCA to embeddings...")
    embedding_df = apply_pca_to_embeddings(embedding_df, outcome)

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
