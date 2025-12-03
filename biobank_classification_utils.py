from pydantic import BaseModel
from enum import Enum
from scipy.stats import loguniform, uniform
from typing import Dict, Tuple, List, Any
import pandas as pd
import os
from biobank_experiment_constants import FULL_PYPPG_FEATURES
from biobank_feature_functions import (
    remove_high_vif_features,
    remove_highly_correlated_features,
)


class ClassificationModelTypes(Enum):
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


class ClassificationExperimentConfig:
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
    from catboost import CatBoostClassifier

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
    raise NotImplementedError(
        "TabPFN model setup is not implemented yet. Please use other models."
    )


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


def setup_classification_model(
    model_type: ClassificationModelTypes,
) -> Tuple[object, Dict]:
    """
    Setup the model based on the model type.

    Parameters:
    - model_type (ModelTypes): The type of model to setup.

    Returns:
    - Tuple[object, Dict]: The model object and the hyperparameter search space.
    """
    if model_type == ClassificationModelTypes.LOGISTIC_REGRESSION:
        return setup_LR_model()

    if model_type == ClassificationModelTypes.XGBOOST:
        return setup_xgboost_model()

    if model_type == ClassificationModelTypes.TABPFN:
        return setup_tabpfn_model()
    if model_type == ClassificationModelTypes.CATBOOST:
        return setup_catboost_model()

    raise ValueError(f"Unsupported model type: {model_type}")


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
    # Keep only embeddings where eid is present in pyppg_df_final

    embedding_df_filtered = embedding_df[
        embedding_df["eid"].isin(pyppg_df_final["eid"])
    ]

    # Combine all features
    all_features = pd.concat(
        [
            embedding_df_filtered.set_index("eid"),
            traditional_df.set_index("eid"),
            pyppg_df_final.set_index("eid"),
        ],
        axis=1,
    ).reset_index()
    # Combine all features
    all_features = pd.concat([embedding_df, traditional_df, pyppg_df_final], axis=1)

    # Drop the outcome column if present
    if outcome in all_features.columns:
        all_features = all_features.drop(columns=[outcome])

    return all_features, target
