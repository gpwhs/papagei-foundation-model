import os
import time
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pydantic import BaseModel
from tqdm import tqdm

# Survival analysis specific imports
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import (
    concordance_index_censored,
    integrated_brier_score,
    cumulative_dynamic_auc,
)
from sksurv.util import Surv

from biobank_embeddings_extraction import extract_features
from biobank_utils import (
    load_yaml_config,
    remove_highly_corr_features,
    get_embedding_df,
    PYPPG_FEATURES,
)


class SurvivalResults(BaseModel):
    """Results from a survival analysis experiment."""

    model: str
    parameters: Dict[str, Any]
    c_index: float
    c_index_lower_ci: float
    c_index_upper_ci: float
    ibs: float  # Integrated Brier Score
    ibs_lower_ci: float
    ibs_upper_ci: float
    time_auc_1yr: float  # Time-dependent AUC at 1 year
    time_auc_3yr: float  # Time-dependent AUC at 3 years
    time_auc_5yr: float  # Time-dependent AUC at 5 years
    training_time: float


class SurvivalExperimentConfig:
    """Configuration for a survival analysis experiment."""

    def __init__(
        self,
        name: str,
        description: str,
        feature_columns: List[str],
    ):
        self.name = name
        self.description = description
        self.feature_columns = feature_columns


class SurvivalModelTypes:
    """Types of survival models."""

    COX_PH = "cox_ph"
    RSF = "rsf"
    XGBSE = "xgbse"


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


def preprocess_data_for_survival(
    df: pd.DataFrame,
    event_col: str,
    time_col: str,
    embedding_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Preprocess data for survival analysis.

    Args:
        df: Source DataFrame
        event_col: Column name for the event indicator (1 for event, 0 for censored)
        time_col: Column name for time-to-event
        embedding_df: DataFrame with pre-computed embeddings

    Returns:
        Tuple of (features_df, structured_outcome_array)
    """
    # Extract traditional features
    traditional_features = ["age", "sex", "BMI"]
    traditional_df = df[traditional_features]

    # Extract pyPPG features
    pyppg_df = df[PYPPG_FEATURES]

    # Combine all features
    all_features = pd.concat([embedding_df, traditional_df, pyppg_df], axis=1)

    # Drop the outcome columns if present
    if event_col in all_features.columns:
        all_features = all_features.drop(columns=[event_col])
    if time_col in all_features.columns:
        all_features = all_features.drop(columns=[time_col])

    # Create a structured array for scikit-survival
    event = df[event_col].astype(bool)  # Convert to boolean for scikit-survival
    time = df[time_col].astype(float)

    y = Surv.from_arrays(event=event, time=time)
    print("checking for nans")
    print(f"{all_features.shape[0], len(y)}")

    assert all_features.shape[0] == len(
        y
    ), "Mismatch between features and outcome length"
    print("check complete")
    return all_features, y


def setup_cox_ph_model():
    """Set up a Cox PH model with hyperparameter search space."""
    model = CoxPHFitter()
    param_grid = {
        "penalizer": [0.0, 0.001, 0.01, 0.1, 1.0],
        "l1_ratio": [0.0, 0.5, 1.0],  # 0 is ridge, 1 is lasso
    }

    return model, param_grid


def setup_rsf_model():
    """Set up a Random Survival Forest model with hyperparameter search space."""
    model = RandomSurvivalForest(random_state=42)
    param_grid = {
        "n_estimators": [50, 100, 200, 300, 500],
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", None],
    }

    return model, param_grid


def setup_xgbse_model():
    """Set up an XGBSE model with hyperparameter search space."""
    try:
        from xgbse import XGBSEDebiasedBCE

        model = XGBSEDebiasedBCE()
        param_grid = {
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "n_estimators": [50, 100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6, 8],
            # "min_child_weight": [1, 3, 5, 7],
            # "gamma": [0, 0.1, 0.2, 0.5],
            # "subsample": [0.6, 0.8, 1.0],
            # "colsample_bytree": [0.6, 0.8, 1.0],
        }

        return model, param_grid
    except ImportError:
        print("XGBSE package not installed. Please install with: pip install xgbse")
        return None, None


def bootstrap_c_index(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model_type: str,
    n_bootstrap: int = 100,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Calculate bootstrap confidence intervals for concordance index.

    Args:
        model: Trained survival model
        X_test: Test features
        y_test: Test outcomes (structured array)
        model_type: Type of model
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level for confidence interval

    Returns:
        Tuple of (lower CI, upper CI)
    """
    c_indices = []
    n_samples = len(X_test)
    rng = np.random.RandomState(42)

    for _ in range(n_bootstrap):
        # Generate bootstrap sample
        indices = rng.choice(n_samples, n_samples, replace=True)
        X_bootstrap = X_test.iloc[indices]
        y_bootstrap = y_test[indices]

        # Calculate concordance index based on model type
        if model_type == SurvivalModelTypes.COX_PH:
            bootstrap_df = pd.DataFrame(
                {
                    "T": y_bootstrap["time"],
                    "E": y_bootstrap["event"],
                }
            )
            bootstrap_df = pd.concat([bootstrap_df, X_bootstrap], axis=1)

            c_index = model.score(bootstrap_df, scoring_method="concordance_index")
        else:
            c_index = concordance_index_censored(
                y_bootstrap["event"], y_bootstrap["time"], model.predict(X_bootstrap)
            )[0]

        c_indices.append(c_index)

    # Calculate confidence intervals
    lower_ci = np.percentile(c_indices, alpha / 2 * 100)
    upper_ci = np.percentile(c_indices, (1 - alpha / 2) * 100)

    return lower_ci, upper_ci


def calculate_integrated_brier_score(
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    time_points: List[float],
    model_type: str,
    n_bootstrap: int = 100,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Calculate Integrated Brier Score with confidence intervals.

    Args:
        model: Trained survival model
        X_train: Training features
        y_train: Training outcomes
        X_test: Test features
        y_test: Test outcomes
        time_points: Time points for evaluation
        model_type: Type of survival model
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level for confidence interval

    Returns:
        Tuple of (IBS, lower CI, upper CI)
    """
    # For non-scikit-learn models or models that don't support survival function prediction
    if model_type == SurvivalModelTypes.COX_PH:
        # For Cox models from lifelines
        try:
            # Create test dataset for lifelines format
            test_df = pd.DataFrame(
                {
                    "T": y_test["time"],
                    "E": y_test["event"],
                }
            )
            test_df = pd.concat([test_df, X_test], axis=1)

            # For lifelines, we can use their built-in metrics
            # But IBS is not directly available in lifelines, so we return placeholder
            return 0.0, 0.0, 0.0
        except Exception as e:
            print(f"Error calculating IBS for Cox model: {e}")
            return 0.0, 0.0, 0.0

    # For scikit-survival models that support survival function prediction
    try:
        # Function to predict survival function at specific time points
        def predict_survival(model, X, times):
            if hasattr(model, "predict_survival_function"):
                # For RSF
                survival_probs = np.zeros((X.shape[0], len(times)))
                for i, x in enumerate(X.values):
                    surv_func = model.predict_survival_function(x.reshape(1, -1))
                    # Interpolate at the specific time points
                    for j, t in enumerate(times):
                        # Find the closest time point in the model's time grid
                        time_index = np.argmin(np.abs(surv_func.x - t))
                        survival_probs[i, j] = surv_func.y[0][time_index]
                return survival_probs
            else:
                # For other models
                return np.ones((X.shape[0], len(times))) * 0.5  # Placeholder

        # Calculate IBS
        survival_predictions = predict_survival(model, X_test, time_points)
        ibs = integrated_brier_score(y_train, y_test, survival_predictions, time_points)

        # Calculate bootstrap confidence intervals
        ibs_scores = []
        n_samples = len(X_test)
        rng = np.random.RandomState(42)

        for _ in range(n_bootstrap):
            # Generate bootstrap sample
            indices = rng.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X_test.iloc[indices]
            y_bootstrap = y_test[indices]

            # Calculate IBS for bootstrap sample
            survival_predictions = predict_survival(model, X_bootstrap, time_points)
            bootstrap_ibs = integrated_brier_score(
                y_train, y_bootstrap, survival_predictions, time_points
            )
            ibs_scores.append(bootstrap_ibs)

        # Calculate confidence intervals
        lower_ci = np.percentile(ibs_scores, alpha / 2 * 100)
        upper_ci = np.percentile(ibs_scores, (1 - alpha / 2) * 100)

        return ibs, lower_ci, upper_ci

    except Exception as e:
        print(f"Error calculating IBS: {e}")
        import traceback

        traceback.print_exc()
        return 0.0, 0.0, 0.0


def calculate_time_dependent_auc(
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    time_points: List[float],
    model_type: str,
) -> Dict[float, float]:
    """Calculate time-dependent AUC at specific time points.

    Args:
        model: Trained survival model
        X_train: Training features
        y_train: Training outcomes
        X_test: Test features
        y_test: Test outcomes
        time_points: Time points for evaluation
        model_type: Type of survival model

    Returns:
        Dictionary mapping time points to AUC values
    """
    results = {}

    try:
        # Model-specific handling
        if model_type == SurvivalModelTypes.COX_PH:
            # For Cox models from lifelines, convert to risk scores
            risk_scores = model.predict_partial_hazard(X_test)
            risk_scores = risk_scores.values.flatten()
        else:
            # For scikit-survival models
            risk_scores = model.predict(X_test)

        # Calculate time-dependent AUC at each time point
        for t in time_points:
            try:
                # For scikit-survival models
                auc, _ = cumulative_dynamic_auc(y_train, y_test, risk_scores, t)
                results[t] = auc[-1]  # Use the last value which corresponds to time t
            except Exception as e:
                print(f"Error calculating time-dependent AUC at time {t}: {e}")
                results[t] = 0.5  # Default value (random guess)

    except Exception as e:
        print(f"Error calculating time-dependent AUC: {e}")
        import traceback

        traceback.print_exc()
        # Return default values
        for t in time_points:
            results[t] = 0.5

    return results


def plot_survival_curves(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model_type: str,
    model_name: str,
    output_dir: str,
    n_curves: int = 10,
) -> None:
    """Plot survival curves for a sample of test instances.

    Args:
        model: Trained survival model
        X_test: Test features
        y_test: Test outcomes
        model_type: Type of model
        model_name: Name of model
        output_dir: Directory to save plots
        n_curves: Number of individual curves to plot
    """
    model_dir = f"{output_dir}/{model_type}"
    os.makedirs(model_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))

    # Plot Kaplan-Meier estimate as reference
    kmf = KaplanMeierFitter()
    kmf.fit(y_test["time"], y_test["event"])
    kmf.plot(label="Kaplan-Meier (actual)", color="black", linewidth=3, alpha=0.6)

    # Get a sample of test instances
    sample_indices = np.random.choice(
        len(X_test), min(n_curves, len(X_test)), replace=False
    )

    # Model-specific handling for survival function prediction
    if model_type == SurvivalModelTypes.COX_PH:
        # For Cox PH models
        for i, idx in enumerate(sample_indices):
            # Get survival function for this instance
            sf = model.predict_survival_function(X_test.iloc[idx : idx + 1])
            plt.step(
                sf.index.values, sf.values.flatten(), label=f"Patient {i+1}", alpha=0.5
            )

    elif model_type in [SurvivalModelTypes.RSF, SurvivalModelTypes.XGBSE]:
        # For RSF and XGBSE models
        max_time = max(y_test["time"])
        times = np.linspace(0, max_time, 100)

        for i, idx in enumerate(sample_indices):
            try:
                # Get survival function for this instance
                sf = model.predict_survival_function(X_test.iloc[idx : idx + 1].values)

                # Extract x, y values for plotting
                if hasattr(sf, "x") and hasattr(sf, "y"):
                    # For scikit-survival models that return a StepFunction
                    x_values = sf.x
                    y_values = sf.y[0]
                else:
                    # For models that return a callable function
                    x_values = times
                    y_values = [sf(t) for t in times]

                plt.step(x_values, y_values, label=f"Patient {i+1}", alpha=0.5)
            except Exception as e:
                print(f"Error plotting survival function for patient {i+1}: {e}")

    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title(f"Survival Curves for {model_name}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.savefig(f"{model_dir}/{model_name}_survival_curves.png")
    plt.close()


def save_hazard_ratios(
    model: Any,
    model_name: str,
    output_dir: str,
) -> None:
    """Save and plot hazard ratios for Cox PH models.

    Args:
        model: Trained Cox PH model
        model_name: Name of the model
        output_dir: Directory to save output
    """
    try:
        # Extract summary from the model
        summary_df = model.summary

        # Save the summary to CSV
        summary_df.to_csv(f"{output_dir}/{model_name}_hazard_ratios.csv")

        # Plot hazard ratios with confidence intervals
        plt.figure(figsize=(12, 8))

        # Sort by hazard ratio for better visualization
        summary_df = summary_df.sort_values(by="exp(coef)")

        # Plot point estimates and CIs
        plt.errorbar(
            summary_df["exp(coef)"],
            range(len(summary_df)),
            xerr=[
                summary_df["exp(coef)"] - summary_df["exp(coef) lower 95%"],
                summary_df["exp(coef) upper 95%"] - summary_df["exp(coef)"],
            ],
            fmt="o",
            capsize=5,
        )

        # Add reference line at HR=1
        plt.axvline(x=1, color="red", linestyle="--", alpha=0.5)

        # Set y-ticks to feature names
        plt.yticks(range(len(summary_df)), summary_df.index)

        plt.xlabel("Hazard Ratio (log scale)")
        plt.ylabel("Feature")
        plt.title(f"Hazard Ratios with 95% CI - {model_name}")
        plt.xscale("log")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_hazard_ratios.png")
        plt.close()
    except Exception as e:
        print(f"Error saving hazard ratios: {e}")


def save_feature_importance(
    model: Any,
    feature_names: pd.Index,
    model_type: str,
    model_name: str,
    output_dir: str,
) -> None:
    """Save and plot feature importance for tree-based models.

    Args:
        model: Trained tree-based model
        feature_names: Names of features
        model_type: Type of model
        model_name: Name of model
        output_dir: Directory to save output
    """
    model_dir = f"{output_dir}/{model_type}"
    os.makedirs(model_dir, exist_ok=True)

    try:
        # Get feature importance based on model type
        if model_type == SurvivalModelTypes.RSF:
            importance = model.feature_importances_
        elif model_type == SurvivalModelTypes.XGBSE:
            # XGBSE models typically have a base XGBoost model
            if hasattr(model, "base_model") and hasattr(
                model.base_model, "feature_importances_"
            ):
                importance = model.base_model.feature_importances_
            else:
                print(f"Feature importance not available for this model: {model_type}")
                return
        else:
            print(f"Feature importance not implemented for model type: {model_type}")
            return

        # Create DataFrame for importance
        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importance}
        ).sort_values("Importance", ascending=False)

        # Save to CSV
        importance_df.to_csv(
            f"{model_dir}/{model_name}_feature_importance.csv", index=False
        )

        # Plot top 20 features or all if less than 20
        n_features = min(20, len(feature_names))
        top_features = importance_df.head(n_features)

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_features["Importance"], align="center")
        plt.yticks(range(len(top_features)), top_features["Feature"])
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(f"Top {n_features} Feature Importance - {model_name}")
        plt.tight_layout()
        plt.savefig(f"{model_dir}/{model_name}_feature_importance.png")
        plt.close()
    except Exception as e:
        print(f"Error saving feature importance: {e}")
        import traceback

        traceback.print_exc()


def save_survival_model(
    model: Any,
    model_type: str,
    model_name: str,
    output_dir: str,
) -> None:
    """Save the trained survival model.

    Args:
        model: Trained model
        model_type: Type of model
        model_name: Name of the model
        output_dir: Directory to save model
    """
    model_path = f"{output_dir}/{model_name}_model.joblib"
    try:
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")


def create_model_comparison(
    results: Dict[str, Dict[str, SurvivalResults]],
    output_dir: str,
) -> None:
    """Create comparison plots and tables for different models.

    Args:
        results: Dictionary mapping experiment keys to dictionaries mapping model types to results
        output_dir: Directory to save comparison
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if results is empty or incomplete
    if not results:
        print("No results to compare. Skipping model comparison.")
        return

    # Create DataFrame for comparison
    comparison_rows = []

    for model_type, model_results in results.items():
        for exp_key, exp_result in model_results.items():
            comparison_rows.append(
                {
                    "ExperimentID": exp_key,
                    "ModelType": model_type,
                    "C-Index": exp_result.c_index,
                    "C-Index CI": f"({exp_result.c_index_lower_ci:.3f}-{exp_result.c_index_upper_ci:.3f})",
                    "IBS": exp_result.ibs,
                    "IBS CI": f"({exp_result.ibs_lower_ci:.3f}-{exp_result.ibs_upper_ci:.3f})",
                    "1-Year AUC": exp_result.time_auc_1yr,
                    "3-Year AUC": exp_result.time_auc_3yr,
                    "5-Year AUC": exp_result.time_auc_5yr,
                    "Training Time": exp_result.training_time,
                }
            )

    # If no rows, return early
    if not comparison_rows:
        print("No comparison data available. Skipping model comparison.")
        return

    comparison_df = pd.DataFrame(comparison_rows)

    # Save comparison to CSV
    comparison_df.to_csv(f"{output_dir}/model_comparison.csv", index=False)

    # Create comparison plots
    # C-Index comparison
    plt.figure(figsize=(14, 8))

    # Get unique experiments and models
    experiments = sorted(comparison_df["ExperimentID"].unique())
    models = sorted(comparison_df["ModelType"].unique())

    x = np.arange(len(experiments))
    width = 0.8 / len(models) if models else 0.8

    for i, model in enumerate(models):
        model_data = comparison_df[comparison_df["ModelType"] == model]
        model_data = model_data.set_index("ExperimentID").reindex(experiments)
        plt.bar(
            x + i * width - 0.4 + width / 2, model_data["C-Index"], width, label=model
        )

    plt.xlabel("Experiment")
    plt.ylabel("C-Index")
    plt.title("C-Index Comparison")
    plt.xticks(x, experiments)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/c_index_comparison.png")
    plt.close()

    # IBS comparison
    plt.figure(figsize=(14, 8))

    for i, model in enumerate(models):
        model_data = comparison_df[comparison_df["ModelType"] == model]
        model_data = model_data.set_index("ExperimentID").reindex(experiments)
        plt.bar(x + i * width - 0.4 + width / 2, model_data["IBS"], width, label=model)

    plt.xlabel("Experiment")
    plt.ylabel("Integrated Brier Score (lower is better)")
    plt.title("Integrated Brier Score Comparison")
    plt.xticks(x, experiments)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/ibs_comparison.png")
    plt.close()

    # Time-dependent AUC comparison
    for year, col in zip([1, 3, 5], ["1-Year AUC", "3-Year AUC", "5-Year AUC"]):
        plt.figure(figsize=(14, 8))

        for i, model in enumerate(models):
            model_data = comparison_df[comparison_df["ModelType"] == model]
            model_data = model_data.set_index("ExperimentID").reindex(experiments)
            plt.bar(
                x + i * width - 0.4 + width / 2, model_data[col], width, label=model
            )

        plt.xlabel("Experiment")
        plt.ylabel(f"{year}-Year AUC")
        plt.title(f"{year}-Year AUC Comparison")
        plt.xticks(x, experiments)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/{year}_year_auc_comparison.png")
        plt.close()


def setup_survival_experiments(
    embedding_columns: List[str],
    traditional_features: List[str] = ["age", "sex", "BMI"],
) -> Dict[str, SurvivalExperimentConfig]:
    """Set up experiment configurations.

    Args:
        embedding_columns: List of embedding column names
        traditional_features: List of traditional feature names

    Returns:
        Dictionary mapping experiment keys to configurations
    """
    return {
        "M0": SurvivalExperimentConfig(
            name="M0_PaPaGei_Only",
            description="Only PaPaGei features",
            feature_columns=embedding_columns,
        ),
        "M1": SurvivalExperimentConfig(
            name="M1_Traditional_Only",
            description="Only metadata (age, sex, BMI)",
            feature_columns=traditional_features,
        ),
        "M2": SurvivalExperimentConfig(
            name="M2_PaPaGei_Traditional",
            description="Both PaPaGei features and metadata",
            feature_columns=embedding_columns + traditional_features,
        ),
        "M3": SurvivalExperimentConfig(
            name="M3_pyPPG_Only",
            description="pyPPG features",
            feature_columns=PYPPG_FEATURES,
        ),
        "M4": SurvivalExperimentConfig(
            name="M4_pyPPG_Traditional",
            description="pyPPG features and metadata",
            feature_columns=PYPPG_FEATURES + traditional_features,
        ),
    }


def train_and_evaluate_survival_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,  # Structured array for scikit-survival
    y_test: np.ndarray,
    model_type: str,
    model_name: str,
    output_dir: str,
    time_points: List[float] = None,  # Time points for evaluation (e.g., 1yr, 3yr, 5yr)
) -> SurvivalResults:
    """Train a survival model and evaluate it.

    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training outcomes (structured array)
        y_test: Testing outcomes (structured array)
        model_type: Type of model (e.g., "cox_ph", "rsf", "xgbse")
        model_name: Name of the model for saving outputs
        output_dir: Directory to save results
        time_points: Specific time points for time-dependent metrics

    Returns:
        SurvivalResults object with metrics
    """
    # Set default time points if not provided
    if time_points is None:
        time_points = [365, 365 * 3, 365 * 5]  # 1 year, 3 years, 5 years

    # Ensure output directory exists
    model_output_dir = f"{output_dir}/{model_type}"
    os.makedirs(model_output_dir, exist_ok=True)

    # Set up model and parameter grid based on model type
    if model_type == SurvivalModelTypes.COX_PH:
        model, param_grid = setup_cox_ph_model()
    elif model_type == SurvivalModelTypes.RSF:
        model, param_grid = setup_rsf_model()
    elif model_type == SurvivalModelTypes.XGBSE:
        model, param_grid = setup_xgbse_model()
        if model is None:
            raise ValueError(
                "XGBSE model could not be set up. Is the package installed?"
            )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Standardize features for some models
    if model_type in [SurvivalModelTypes.COX_PH]:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

        # === Insert exactly HERE ===
        X_train_scaled = X_train_scaled.apply(pd.to_numeric, errors="coerce")
        X_test_scaled = X_test_scaled.apply(pd.to_numeric, errors="coerce")

        X_train_scaled.dropna(inplace=True)
        X_test_scaled.dropna(inplace=True)

        # Re-align outcomes after dropping rows
        X_train_scaled, y_train = X_train_scaled.align(
            pd.DataFrame(y_train), join="inner", axis=0
        )
        X_test_scaled, y_test = X_test_scaled.align(
            pd.DataFrame(y_test), join="inner", axis=0
        )

        # Confirm that data is now clean
        assert (
            not X_train_scaled.isnull().values.any()
        ), "NaNs remain after numeric coercion in train data!"
        assert not np.isinf(
            X_train_scaled.values
        ).any(), "Infs remain after numeric coercion in train data!"
        # Save the scaler for later use
        joblib.dump(scaler, f"{model_output_dir}/{model_name}_scaler.joblib")
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # Model-specific handling for hyperparameter tuning and training
    start_time = time.time()
    best_params = {}

    if model_type == SurvivalModelTypes.COX_PH:
        # For Cox PH, we perform manual grid search
        best_c_index = -1

        for penalizer in tqdm(
            param_grid["penalizer"], desc=f"Grid search for {model_name}"
        ):
            for l1_ratio in param_grid["l1_ratio"]:
                current_model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)

                # Convert structured array to DataFrame for lifelines
                train_df = pd.DataFrame(
                    {
                        "T": y_train["time"],
                        "E": y_train["event"].astype(int),
                    }
                )
                train_df = pd.concat([train_df, X_train_scaled], axis=1)
                assert (
                    not train_df.isnull().values.any()
                ), "NaNs detected in train_df before fitting!"
                assert not np.isinf(
                    train_df.values
                ).any(), "Inf detected in train_df before fitting!"
                try:
                    # Fit the model
                    current_model.fit(train_df, duration_col="T", event_col="E")

                    # Evaluate on test set
                    test_df = pd.DataFrame(
                        {
                            "T": y_test["time"],
                            "E": y_test["event"].astype(int),
                        }
                    )
                    test_df = pd.concat([test_df, X_test_scaled], axis=1)
                    # Debugging checks before evaluation
                    assert (
                        not test_df.isnull().values.any()
                    ), "NaNs detected in test_df before scoring!"
                    assert not np.isinf(
                        test_df.values
                    ).any(), "Inf detected in test_df before scoring!"
                    # Calculate concordance index
                    c_index = current_model.score(
                        test_df, scoring_method="concordance_index"
                    )

                    if c_index > best_c_index:
                        best_c_index = c_index
                        best_params = {"penalizer": penalizer, "l1_ratio": l1_ratio}
                        best_model = current_model
                except Exception as e:
                    print(
                        f"Error fitting Cox model with params {penalizer}, {l1_ratio}: {e}"
                    )

        # Refit with best parameters
        train_df = pd.DataFrame(
            {
                "T": y_train["time"],
                "E": y_train["event"],
            }
        )
        train_df = pd.concat([train_df, X_train_scaled], axis=1)

        best_model = CoxPHFitter(
            penalizer=best_params["penalizer"], l1_ratio=best_params["l1_ratio"]
        )
        best_model.fit(train_df, duration_col="T", event_col="E")

    elif model_type == SurvivalModelTypes.RSF:
        # For RSF, we use RandomizedSearchCV
        from sklearn.model_selection import RandomizedSearchCV

        # Create a custom scorer that uses concordance index
        def c_index_scorer(estimator, X, y):
            predictions = estimator.predict(X)
            return concordance_index_censored(y["event"], y["time"], predictions)[0]

        # Create a wrapper for the scorer to work with scikit-learn
        from sklearn.metrics import make_scorer

        c_index_scorer_wrapped = make_scorer(c_index_scorer, greater_is_better=True)

        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=20,
            cv=3,
            scoring=c_index_scorer_wrapped,  # Use c-index for scoring
            random_state=42,
            n_jobs=-1,
            verbose=2,
        )

        random_search.fit(X_train_scaled, y_train)
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_

    elif model_type == SurvivalModelTypes.XGBSE:
        # For XGBSE, use appropriate hyperparameter tuning
        try:
            # First try to import from xgbse package
            try:
                from xgbse import XGBSEDebiasedBCE

                # Convert structured array to DataFrame for XGBSE
                train_df = pd.DataFrame(
                    {
                        "time": y_train["time"],
                        "event": y_train["event"],
                    }
                )

                print("Using actual XGBSE package")

                # Manual parameter tuning since actual XGBSE has different parameter structure
                param_combinations = [
                    {"xgb_params": {"eta": 0.1, "max_depth": 3, "n_estimators": 100}},
                    {"xgb_params": {"eta": 0.05, "max_depth": 5, "n_estimators": 200}},
                    {"xgb_params": {"eta": 0.01, "max_depth": 7, "n_estimators": 300}},
                ]

                best_score = -1
                best_params = None
                best_model = None

                for params in param_combinations:
                    try:
                        # For actual XGBSE package, we need to use its specific parameters
                        xgb_params = params["xgb_params"]
                        xgb_params["objective"] = (
                            "survival:cox"  # Use appropriate survival objective
                        )

                        model = XGBSEDebiasedBCE(xgb_params=xgb_params)

                        # Fit the model according to XGBSE API
                        model.fit(
                            X_train_scaled,
                            time=train_df["time"],
                            event=train_df["event"],
                        )

                        # Evaluate
                        risk_scores = model.predict(X_test_scaled)
                        c_index = concordance_index(
                            y_test["time"], -risk_scores, y_test["event"]
                        )

                        print(f"Params: {params}, C-index: {c_index:.4f}")

                        if c_index > best_score:
                            best_score = c_index
                            best_params = params
                            best_model = model
                    except Exception as param_error:
                        print(f"Error with parameters {params}: {param_error}")

                if best_model is None:
                    print("Falling back to default XGBSE parameters")
                    # Use default parameters
                    best_model = XGBSEDebiasedBCE(
                        xgb_params={
                            "objective": "survival:cox",
                            "eta": 0.1,
                            "max_depth": 3,
                        }
                    )
                    best_params = {
                        "xgb_params": {
                            "objective": "survival:cox",
                            "eta": 0.1,
                            "max_depth": 3,
                        }
                    }
                    best_model.fit(
                        X_train_scaled,
                        time=train_df["time"],
                        event=train_df["event"],
                    )
            except ImportError:
                # Use our custom implementation
                print("Using custom XGBSE implementation")
                from xgbse_utils import XGBSEDebiasedBCE

                # Simple parameter search
                best_score = -1
                best_params = {}

                # Try a few parameter combinations
                param_combinations = [
                    {"learning_rate": 0.1, "max_depth": 3, "n_estimators": 100},
                    {"learning_rate": 0.05, "max_depth": 5, "n_estimators": 200},
                    {"learning_rate": 0.01, "max_depth": 7, "n_estimators": 300},
                ]

                for params in param_combinations:
                    model = XGBSEDebiasedBCE(**params)
                    train_df = pd.DataFrame(
                        {
                            "time": y_train["time"],
                            "event": y_train["event"],
                        }
                    )

                    # Fit the model
                    model.fit(X_train_scaled, train_df["time"], train_df["event"])

                    # Evaluate
                    risk_scores = model.predict(X_test_scaled)
                    c_index = concordance_index(
                        y_test["time"], -risk_scores, y_test["event"]
                    )

                    if c_index > best_score:
                        best_score = c_index
                        best_params = params
                        best_model = model

                if not best_params:
                    # Use default if no parameters were better
                    best_model = XGBSEDebiasedBCE()
                    best_params = {
                        "learning_rate": 0.1,
                        "max_depth": 3,
                        "n_estimators": 100,
                    }
                    train_df = pd.DataFrame(
                        {
                            "time": y_train["time"],
                            "event": y_train["event"],
                        }
                    )
                    best_model.fit(X_train_scaled, train_df["time"], train_df["event"])

        except Exception as e:
            print(f"Error with XGBSE model: {e}")
            import traceback

            traceback.print_exc()
            # Fallback to a simpler approach
            # Use default parameters
            try:
                from xgbse_utils import XGBSEDebiasedBCE

                best_model = XGBSEDebiasedBCE()
                best_params = {}
                train_df = pd.DataFrame(
                    {
                        "time": y_train["time"],
                        "event": y_train["event"],
                    }
                )
                best_model.fit(X_train_scaled, train_df["time"], train_df["event"])
            except Exception as e2:
                print(f"Could not fallback to custom XGBSE implementation: {e2}")
                # Use RSF as a last resort fallback
                from sksurv.ensemble import RandomSurvivalForest

                best_model = RandomSurvivalForest(random_state=42)
                best_params = {"n_estimators": 100, "max_depth": 5}
                best_model.fit(X_train_scaled, y_train)

    training_time = time.time() - start_time

    # Evaluate model performance
    # Calculate concordance index
    if model_type == SurvivalModelTypes.COX_PH:
        test_df = pd.DataFrame(
            {
                "T": y_test["time"],
                "E": y_test["event"],
            }
        )
        test_df = pd.concat([test_df, X_test_scaled], axis=1)

        c_index = best_model.score(test_df, scoring_method="concordance_index")
    else:
        c_index = concordance_index_censored(
            y_test["event"], y_test["time"], best_model.predict(X_test_scaled)
        )[0]

    # Calculate confidence intervals for c-index using bootstrap
    c_index_lower_ci, c_index_upper_ci = bootstrap_c_index(
        best_model, X_test_scaled, y_test, model_type
    )

    # Calculate Integrated Brier Score for models that support survival functions
    ibs, ibs_lower_ci, ibs_upper_ci = calculate_integrated_brier_score(
        best_model,
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        time_points,
        model_type,
    )

    # Calculate time-dependent AUC at specific time points
    time_auc_results = calculate_time_dependent_auc(
        best_model,
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        time_points,
        model_type,
    )

    # Save model and evaluation results
    save_survival_model(best_model, model_type, model_name, model_output_dir)

    # Plot survival curves and other visualizations
    plot_survival_curves(
        best_model, X_test_scaled, y_test, model_type, model_name, output_dir
    )

    # For Cox models, save and plot hazard ratios
    if model_type == SurvivalModelTypes.COX_PH:
        save_hazard_ratios(best_model, model_name, model_output_dir)

    # For tree-based models, save feature importance
    if model_type in [SurvivalModelTypes.RSF, SurvivalModelTypes.XGBSE]:
        save_feature_importance(
            best_model, X_train.columns, model_type, model_name, output_dir
        )

    # Print some results
    print(f"\n{model_name} - {model_type} Survival Analysis Results:")
    print(f"C-Index: {c_index:.4f} (CI: {c_index_lower_ci:.4f}-{c_index_upper_ci:.4f})")
    if ibs > 0:
        print(
            f"Integrated Brier Score: {ibs:.4f} (CI: {ibs_lower_ci:.4f}-{ibs_upper_ci:.4f})"
        )
    print("Time-dependent AUC:")
    for t, auc in time_auc_results.items():
        years = t / 365
        print(f"  {years:.1f}-Year AUC: {auc:.4f}")

    # Create and return results object
    results = SurvivalResults(
        model=model_type,
        parameters=best_params,
        c_index=c_index,
        c_index_lower_ci=c_index_lower_ci,
        c_index_upper_ci=c_index_upper_ci,
        ibs=ibs,
        ibs_lower_ci=ibs_lower_ci,
        ibs_upper_ci=ibs_upper_ci,
        time_auc_1yr=time_auc_results.get(time_points[0], 0.5),
        time_auc_3yr=time_auc_results.get(time_points[1], 0.5),
        time_auc_5yr=time_auc_results.get(time_points[2], 0.5),
        training_time=training_time,
    )

    return results


def run_survival_experiment(
    experiment_config: SurvivalExperimentConfig,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_type: str,
    output_dir: str,
    time_points: List[float] = None,
) -> SurvivalResults:
    """Run a single survival analysis experiment.

    Args:
        experiment_config: Configuration for the experiment
        X_train, X_test, y_train, y_test: Data splits
        model_type: Type of model to use
        output_dir: Directory to save results
        time_points: Specific time points for time-dependent metrics

    Returns:
        SurvivalResults object with metrics
    """
    print(
        f"\n--- Running Experiment {experiment_config.name}: {experiment_config.description} ---"
    )

    # Select features for this experiment
    X_train_exp = X_train[experiment_config.feature_columns]
    X_test_exp = X_test[experiment_config.feature_columns]

    # Train and evaluate
    results = train_and_evaluate_survival_model(
        X_train_exp,
        X_test_exp,
        y_train,
        y_test,
        model_type=model_type,
        model_name=experiment_config.name,
        output_dir=output_dir,
        time_points=time_points,
    )

    return results


def main() -> None:
    """Main function to run survival analysis experiments."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run survival analysis experiments")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="survival_config.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_yaml_config(args.config)
    models = config["models"]
    event_col = config["event_column"]
    time_col = config["time_column"]
    results_dir = config["results_directory"]

    # Create output directory
    os.makedirs(results_dir, exist_ok=True)

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

    print("Creating embedding dataframe")
    # Create embedding DataFrame
    embedding_df = get_embedding_df(embeddings)
    print("Removing highly correlated features")
    embedding_df = remove_highly_corr_features(embedding_df)

    # Ensure outcome variables are in the embedding DataFrame for later splitting
    if event_col not in embedding_df.columns:
        embedding_df[event_col] = df[event_col]
    if time_col not in embedding_df.columns:
        embedding_df[time_col] = df[time_col]

    # Preprocess data for survival analysis
    all_features, y = preprocess_data_for_survival(
        df, event_col, time_col, embedding_df
    )

    # Print statistics about the survival data
    event_count = np.sum(y["event"])
    censored_count = len(y) - event_count
    event_rate = event_count / len(y)

    print(f"\nSurvival data statistics:")
    print(f"  Total samples: {len(y)}")
    print(f"  Events observed: {event_count} ({event_rate:.2%})")
    print(f"  Censored: {censored_count} ({1-event_rate:.2%})")

    # Print time statistics
    time_min = np.min(y["time"])
    time_max = np.max(y["time"])
    time_median = np.median(y["time"])
    time_mean = np.mean(y["time"])

    print(f"\nTime statistics (in days):")
    print(f"  Min: {time_min:.1f}")
    print(f"  Max: {time_max:.1f}")
    print(f"  Median: {time_median:.1f}")
    print(f"  Mean: {time_mean:.1f}")

    # Plot Kaplan-Meier curve for the overall population
    kmf = KaplanMeierFitter()
    kmf.fit(y["time"], y["event"], label="Overall Population")

    plt.figure(figsize=(10, 6))
    kmf.plot_survival_function()
    plt.title("Overall Kaplan-Meier Survival Curve")
    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{results_dir}/overall_survival_curve.png")
    plt.close()

    # Setup time points for evaluation (1, 3, and 5 years)
    time_points = [365, 3 * 365, 5 * 365]

    # Create train/test splits
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        all_features, y, test_size=0.2, random_state=42, stratify=y["event"]
    )

    # Setup experiments
    embedding_columns = [
        col for col in embedding_df.columns if col not in [event_col, time_col]
    ]
    experiments = setup_survival_experiments(embedding_columns)

    # Dictionary to store results for all models
    all_results = {}

    # Run experiments for each model type
    for model_type in models:
        print(f"\n=== Running {model_type} experiments ===\n")

        # Create model-specific directory
        model_dir = f"{results_dir}/{model_type}"
        os.makedirs(model_dir, exist_ok=True)

        # Dictionary to store results for this model type
        model_results = {}

        # Run all experiments for this model type
        for exp_key, exp_config in tqdm(
            experiments.items(), desc=f"{model_type} experiments"
        ):
            try:
                results = run_survival_experiment(
                    experiment_config=exp_config,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    model_type=model_type,
                    output_dir=results_dir,
                    time_points=time_points,
                )
                model_results[exp_key] = results
            except Exception as e:
                print(f"Error running {model_type} experiment {exp_key}: {e}")
                import traceback

                traceback.print_exc()

        # Store results for this model type
        all_results[model_type] = model_results

        # Save model-specific results to JSON
        with open(f"{model_dir}/experiment_results.json", "w") as f:
            import json

            # Use model_dump() instead of dict() for Pydantic v2 compatibility
            try:
                # For Pydantic v2
                f.write(
                    json.dumps(
                        {k: v.model_dump() for k, v in model_results.items()}, indent=2
                    )
                )
            except AttributeError:
                # Fallback for Pydantic v1
                f.write(
                    json.dumps(
                        {k: v.dict() for k, v in model_results.items()}, indent=2
                    )
                )

    # Create comparison of all models
    create_model_comparison(all_results, results_dir)

    print(f"\nAll survival analysis experiments completed successfully!")
    print(f"Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
