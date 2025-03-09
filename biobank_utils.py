import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple
from scipy.stats import loguniform, uniform
from tqdm import tqdm
import yaml
import seaborn as sns
import numpy as np
import pandas as pd
from enum import Enum


PYPPG_FEATURES = [
    "Tpi",
    "Tpp",
    "Tsys",
    "Tdia",
    "Tsp",
    "Tdp",
    "deltaT",
    "Tsw10",
    "Tsw25",
    "Tsw33",
    "Tsw50",
    "Tsw66",
    "Tsw75",
    "Tsw90",
    "Tdw10",
    "Tdw25",
    "Tdw33",
    "Tdw50",
    "Tdw66",
    "Tdw75",
    "Tdw90",
    "Tpw10",
    "Tpw25",
    "Tpw33",
    "Tpw50",
    "Tpw66",
    "Tpw75",
    "Tpw90",
    "Asp",
    "Adn",
    "Adp",
    "Aoff",
    "AUCpi",
    "AUCsys",
    "AUCdia",
    "IPR",
    "Tsys/Tdia",
    "Tpw25/Tpi",
    "Tpw50/Tpi",
    "Tpw75/Tpi",
    "Tpw25/Tsp",
    "Tpw50/Tsp",
    "Tpw75/Tsp",
    "Tdw10/Tsw10",
    "Tdw25/Tsw25",
    "Tdw33/Tsw33",
    "Tdw50/Tsw50",
    "Tdw66/Tsw66",
    "Tdw75/Tsw75",
    "Tdw90/Tsw90",
    "Tsp/Tpi",
    "Asp/Aoff",
    "Adp/Asp",
    "IPA",
    "Tsp/Asp",
    "Asp/deltaT",
    "Asp/(Tpi-Tsp)",
    "Tu",
    "Tv",
    "Tw",
    "Ta",
    "Tb",
    "Tc",
    "Td",
    "Te",
    "Tf",
    "Tb-c",
    "Tb-d",
    "Tp1",
    "Tp2",
    "Tp1-dp",
    "Tp2-dp",
    "Tu/Tpi",
    "Tv/Tpi",
    "Tw/Tpi",
    "Ta/Tpi",
    "Tb/Tpi",
    "Tc/Tpi",
    "Td/Tpi",
    "Te/Tpi",
    "Tf/Tpi",
    "(Tu-Ta)/Tpi",
    "(Tv-Tb)/Tpi",
    "Au/Asp",
    "Av/Au",
    "Aw/Au",
    "Ab/Aa",
    "Ac/Aa",
    "Ad/Aa",
    "Ae/Aa",
    "Af/Aa",
    "Ap2/Ap1",
    "(Ac-Ab)/Aa",
    "(Ad-Ab)/Aa",
    "AGI",
    "AGImod",
    "AGIinf",
    "AI",
    "RIp1",
    "RIp2",
    "SC",
    "IPAD",
]


class ModelTypes(Enum):
    """
    Enum for the types of models.
    """

    XGBOOST = "xgboost"
    LOGISTIC_REGRESSION = "LR"


def setup_model(model_type: ModelTypes) -> Tuple[object, Dict]:
    """
    Setup the model based on the model type.

    Parameters:
    - model_type (ModelTypes): The type of model to setup.

    Returns:
    - Tuple[object, Dict]: The model object and the hyperparameter search space.
    """
    if model_type == ModelTypes.XGBOOST:
        from xgboost import XGBClassifier

        model = XGBClassifier()
        param_distributions = {
            "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
            "n_estimators": [50, 100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6, 7, 8, 10],
            # "min_child_weight": [1, 2, 3, 5, 7, 10],
            # "gamma": [0, 0.1, 0.2, 0.3, 0.5, 1.0],
            # "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            # "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            # "reg_alpha": [0, 0.001, 0.01, 0.1, 1.0],
            # "reg_lambda": [0, 0.001, 0.01, 0.1, 1.0],
            # "scale_pos_weight": [1, 3, 5, 10],  # For imbalanced datasets
        }
        return model, param_distributions

    if model_type == ModelTypes.LOGISTIC_REGRESSION:
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
                "penalty": ["none"],
                "solver": ["lbfgs"],  # Efficient when no penalty
                "class_weight": ["balanced", None],
            }
        )
        return model, param_distributions

    raise ValueError(f"Unsupported model type: {model_type}")


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


def remove_highly_corr_features(
    features: pd.DataFrame, threshold: float = 0.8
) -> pd.DataFrame:
    """
    Remove highly correlated features from the dataset based on the threshold.

    Parameters:
    - features (pd.DataFrame): The feature matrix.
    - threshold (float): The correlation threshold above which one feature is removed.

    Returns:
    - pd.DataFrame: The filtered DataFrame with highly correlated features removed.
    """
    print(f"Calculating correlation matrix for {features.shape[1]} features...")
    if os.path.exists("filtered_embeddings.parquet"):
        print("Loading correlation filtered embeddings from disk")
        return pd.read_parquet("filtered_embeddings.parquet")
    corr = features.corr().abs()

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Create a list to collect columns to drop
    to_drop = []

    # This is the time-consuming part, so wrap it in tqdm
    for column in tqdm(upper.columns, desc="Filtering Correlated Features"):
        # Check if this column has correlation above threshold with any other column
        if any(upper[column] > threshold):
            to_drop.append(column)

    print(f"Dropping {len(to_drop)} highly correlated features")
    filtered_features = features.drop(columns=to_drop, inplace=False)
    filtered_features.to_parquet("filtered_embeddings.parquet")

    return filtered_features


def load_yaml_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_embedding_df(embeddings: np.ndarray) -> pd.DataFrame:
    """
    Get the column names for the embeddings.
    """
    embedding_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    return pd.DataFrame(embeddings, columns=embedding_cols)


def create_summary(results: dict, results_dir: str, model: str):
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
                f"{results['M0']['accuracy']:.4f} ({results['M0']['accuracy_lower_ci']:.4f}-{results['M0']['accuracy_upper_ci']:.4f})",
                f"{results['M1']['accuracy']:.4f} ({results['M1']['accuracy_lower_ci']:.4f}-{results['M1']['accuracy_upper_ci']:.4f})",
                f"{results['M2']['accuracy']:.4f} ({results['M2']['accuracy_lower_ci']:.4f}-{results['M2']['accuracy_upper_ci']:.4f})",
                f"{results['M3']['accuracy']:.4f} ({results['M3']['accuracy_lower_ci']:.4f}-{results['M3']['accuracy_upper_ci']:.4f})",
                f"{results['M4']['accuracy']:.4f} ({results['M4']['accuracy_lower_ci']:.4f}-{results['M4']['accuracy_upper_ci']:.4f})",
            ],
            "ROC_AUC": [
                f"{results['M0']['auc']:.4f} ({results['M0']['auc_lower_ci']:.4f}-{results['M0']['auc_upper_ci']:.4f})",
                f"{results['M1']['auc']:.4f} ({results['M1']['auc_lower_ci']:.4f}-{results['M1']['auc_upper_ci']:.4f})",
                f"{results['M2']['auc']:.4f} ({results['M2']['auc_lower_ci']:.4f}-{results['M2']['auc_upper_ci']:.4f})",
                f"{results['M3']['auc']:.4f} ({results['M3']['auc_lower_ci']:.4f}-{results['M3']['auc_upper_ci']:.4f})",
                f"{results['M4']['auc']:.4f} ({results['M4']['auc_lower_ci']:.4f}-{results['M4']['auc_upper_ci']:.4f})",
            ],
            "Training_Time": [
                results["M0"]["training_time"],
                results["M1"]["training_time"],
                results["M2"]["training_time"],
                results["M3"]["training_time"],
                results["M4"]["training_time"],
            ],
        }
    )

    summary.to_csv(f"{results_dir}/experiment_summary_{model}.csv", index=False)
    print("\nExperiment Summary:")
    print(summary)

    # Plot results comparison
    plt.figure(figsize=(12, 6))

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
