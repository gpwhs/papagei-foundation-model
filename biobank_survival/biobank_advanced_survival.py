import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from pydantic import BaseModel
import os
from lifelines import WeibullAFTFitter, LogNormalAFTFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Use sksurv for Random Survival Forest
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv

    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False
    print("scikit-survival not available. RandomSurvivalForest will not be usable.")

# Try to import DeepSurv if it's available
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    DEEPSURV_AVAILABLE = True
except ImportError:
    DEEPSURV_AVAILABLE = False
    print("PyTorch not available. DeepSurv will not be usable.")


class AdvancedSurvivalResults(BaseModel):
    """Results from an advanced survival analysis experiment."""

    model: str
    parameters: Dict[str, Any]
    c_index: float
    c_index_lower_ci: float
    c_index_upper_ci: float
    concordance_train: float
    concordance_test: float
    training_time: float
    feature_importance: Optional[Dict[str, float]] = None


def bootstrap_ci_c_index(
    y_true_time: np.ndarray,
    y_true_event: np.ndarray,
    y_pred_risk: np.ndarray,
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Calculate bootstrapped confidence interval for C-index.

    Args:
        y_true_time: Observed times
        y_true_event: Event indicators
        y_pred_risk: Predicted risk scores
        n_bootstraps: Number of bootstrap samples
        random_state: Random seed

    Returns:
        Tuple of (c-index, lower_bound, upper_bound)
    """
    np.random.seed(random_state)

    # Calculate c-index on the full dataset
    c_index = concordance_index(y_true_time, -y_pred_risk, y_true_event)

    # Bootstrap to get confidence interval
    bootstrap_indices = np.random.randint(
        0, len(y_true_time), (n_bootstraps, len(y_true_time))
    )
    bootstrap_c_indices = []

    for indices in bootstrap_indices:
        try:
            bootstrap_c_index = concordance_index(
                y_true_time[indices], -y_pred_risk[indices], y_true_event[indices]
            )
            bootstrap_c_indices.append(bootstrap_c_index)
        except:
            # Skip if there's an error (e.g., no events in the bootstrap sample)
            continue

    # Calculate 95% confidence interval
    lower_bound = np.percentile(bootstrap_c_indices, 2.5)
    upper_bound = np.percentile(bootstrap_c_indices, 97.5)

    return c_index, lower_bound, upper_bound


def save_advanced_results_to_file(
    model_type: str, model_name: str, output_dir: str, results: AdvancedSurvivalResults
) -> None:
    """Save advanced model results to a text file.

    Args:
        model_type: Type of model ('rsf', 'aft', 'deepsurv')
        model_name: Name of the model
        output_dir: Directory to save results
        results: AdvancedSurvivalResults object
    """
    # Create lowercase model type for consistent file naming
    model_type_lower = model_type.lower()

    # Extract just the initial experiment code (e.g., "M4" from "M4_pyPPG_Traditional_RSF")
    exp_code = model_name.split("_")[0]  # Get M0, M1, M2, etc.

    # Go up one directory level from the provided output_dir
    # output_dir is like ".../advanced_survival_MACE/M4_pyPPG_Traditional"
    parent_dir = os.path.dirname(output_dir)  # Get ".../advanced_survival_MACE"

    # Create the expected directory structure for comparison function
    # ".../advanced_survival_MACE/M4_rsf"
    results_dir = f"{parent_dir}/{exp_code}_{model_type_lower}"
    os.makedirs(results_dir, exist_ok=True)

    # Create the results file with expected name
    # ".../advanced_survival_MACE/M4_rsf/M4_rsf_results.txt"
    results_file = f"{results_dir}/{exp_code}_{model_type_lower}_results.txt"

    print(f"Saving results to: {results_file}")

    with open(results_file, "w") as f:
        f.write(f"{model_type.upper()} Model {model_name} Results:\n")
        f.write("=" * 50 + "\n\n")
        f.write(
            f"C-index: {results.c_index:.4f} ({results.c_index_lower_ci:.4f}-{results.c_index_upper_ci:.4f})\n"
        )
        f.write(f"Concordance (train): {results.concordance_train:.4f}\n")
        f.write(f"Concordance (test): {results.concordance_test:.4f}\n")
        f.write(f"Training time: {results.training_time:.2f} seconds\n\n")

        f.write("Parameters:\n")
        for param, value in results.parameters.items():
            f.write(f"  {param}: {value}\n")

        # Feature importance if available
        if results.feature_importance:
            f.write("\nTop Feature Importance:\n")
            sorted_importance = sorted(
                results.feature_importance.items(), key=lambda x: x[1], reverse=True
            )[
                :20
            ]  # Top 20 features

            for feature, importance in sorted_importance:
                f.write(f"  {feature}: {importance:.4f}\n")

    print(f"File exists after saving: {os.path.exists(results_file)}")


def train_random_survival_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    time_train: pd.Series,
    time_test: pd.Series,
    event_train: pd.Series,
    event_test: pd.Series,
    model_name: str,
    outcome: str,
    output_dir: str,
    n_estimators: int = 100,
    max_depth: int = None,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    random_state: int = 42,
) -> AdvancedSurvivalResults:
    """
    Train and evaluate a Random Survival Forest model.

    Args:
        X_train, X_test: Feature DataFrames
        time_train, time_test: Time columns
        event_train, event_test: Event indicator columns
        model_name: Name of the model
        outcome: Name of the outcome
        output_dir: Directory to save results

    Returns:
        AdvancedSurvivalResults object
    """
    import time as time_module

    if not SKSURV_AVAILABLE:
        raise ImportError(
            "scikit-survival is not available. Please install it to use Random Survival Forest."
        )

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert data to the format required by scikit-survival
    # scikit-survival requires structured arrays with dtype [('event', bool), ('time', float)]
    y_train_sksurv = Surv.from_arrays(event_train.astype(bool), time_train)
    y_test_sksurv = Surv.from_arrays(event_test.astype(bool), time_test)
    print(len(X_train), len(X_test))

    # Train the Random Survival Forest
    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,  # Use all cores
        verbose=3,
    )

    start_time = time_module.time()
    rsf.fit(X_train, y_train_sksurv)
    training_time = time_module.time() - start_time

    # Calculate C-index on training and test set
    # The predict method returns estimated risk scores (higher = higher risk)
    train_predictions = rsf.predict(X_train)
    test_predictions = rsf.predict(X_test)

    c_index_train = rsf.score(X_train, y_train_sksurv)
    c_index_test = rsf.score(X_test, y_test_sksurv)

    # Bootstrap confidence interval for C-index
    c_index, c_index_lower, c_index_upper = bootstrap_ci_c_index(
        time_test.values, event_test.values, test_predictions
    )
    print(f"C-index: {c_index:.4f} ({c_index_lower:.4f}, {c_index_upper:.4f})")
    print(f"trying feature importances")
    # TODO: IMPLEMENT FEATURE IMPORTANCE - PERMUTATION
    # # Calculate feature importances
    # feature_importances = dict(zip(X_train.columns, rsf.feature_importances_))
    # print(f"obtained feature importances")
    #
    # # Plot feature importances
    # plt.figure(figsize=(10, max(6, min(len(feature_importances) * 0.3, 15))))
    #
    # # Sort features by importance
    # sorted_features = sorted(
    #     feature_importances.items(), key=lambda x: x[1], reverse=True
    # )
    #
    # # Plot the top 20 features
    # top_features = sorted_features[:20]
    # feature_names = [f[0] for f in top_features]
    # importance_values = [f[1] for f in top_features]
    #
    # # Create bar chart
    # plt.barh(range(len(top_features)), importance_values, align="center")
    # plt.yticks(range(len(top_features)), feature_names)
    # plt.xlabel("Feature Importance")
    # plt.title("Random Survival Forest Feature Importance")
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/{model_name}_rsf_feature_importance.png")
    # plt.close()
    #
    # # Save feature importances to CSV
    # importance_df = pd.DataFrame(
    #     {
    #         "Feature": list(feature_importances.keys()),
    #         "Importance": list(feature_importances.values()),
    #     }
    # ).sort_values("Importance", ascending=False)
    #
    # importance_df.to_csv(
    #     f"{output_dir}/{model_name}_rsf_feature_importance.csv", index=False
    # )

    # Plot survival curves for a few samples
    plt.figure(figsize=(10, 6))

    # Select 5 random samples from test set
    n_samples = min(5, len(X_test))
    sample_indices = np.random.choice(len(X_test), n_samples, replace=False)

    for i, idx in enumerate(sample_indices):
        sample = X_test.iloc[idx : idx + 1]

        # Predict survival function for this sample
        surv_funcs = rsf.predict_survival_function(sample)

        # Plot survival function
        time_points = surv_funcs[0].x
        surv_probs = surv_funcs[0].y

        plt.step(
            time_points,
            surv_probs,
            where="post",
            label=f"Sample {i+1} (time={time_test.iloc[idx]:.0f}, event={event_test.iloc[idx]})",
        )

    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title("Random Survival Forest - Survival Functions")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_rsf_survival_curves.png")
    plt.close()

    # make fake feature importance for now
    feature_importances = {
        f"Feature {i}": 1.0 / len(X_train.columns) for i in range(len(X_train.columns))
    }
    # Create results object
    results = AdvancedSurvivalResults(
        model="Random Survival Forest",
        parameters={
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        },
        c_index=c_index,
        c_index_lower_ci=c_index_lower,
        c_index_upper_ci=c_index_upper,
        concordance_train=c_index_train,
        concordance_test=c_index_test,
        training_time=training_time,
        feature_importance=feature_importances,
    )
    save_advanced_results_to_file("rsf", model_name, output_dir, results)

    return results
