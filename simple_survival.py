#!/usr/bin/env python3
"""
Survival Analysis comparing multiple feature sets using Cox Proportional Hazards and Random Survival Forests
"""
import os
import time
import pickle
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from biobank_utils import (
    remove_highly_corr_features,
    PYPPG_FEATURES,
)
from biobank_embeddings_extraction import get_embeddings, get_embedding_df


def create_output_directories(base_path: str = "survival_analysis") -> Dict[str, str]:
    """
    Create output directories for saving results, visualizations, and models.

    Args:
        base_path: Base directory path

    Returns:
        Dictionary with paths to different output directories
    """
    # Create a timestamp-based directory to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{base_path}_{timestamp}")

    # Create subdirectories
    paths = {
        "base": str(output_dir),
        "models": str(output_dir / "models"),
        "results": str(output_dir / "results"),
        "visualizations": str(output_dir / "visualizations"),
    }

    # Create the directories if they don't exist
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)

    print(f"Created output directories in: {output_dir}")
    return paths


def setup_feature_sets(
    df: pd.DataFrame,
    outcome: str,
    outcome_time: str,
    embedding_df: pd.DataFrame,
    pyppg_features: List[str],
) -> Dict[str, Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]]:
    """
    Create feature sets for modeling based on the specified experimental design.

    Args:
        df: Source DataFrame
        outcome: Target variable name (event indicator)
        outcome_time: Time-to-event variable name
        embedding_df: DataFrame with pre-computed embeddings (PaPaGei features)
        pyppg_features: List of PyPPG feature column names

    Returns:
        Dictionary of feature sets with format {model_name: (features_df, structured_arr, survival_df)}
    """
    # Make sure outcome variables are in the right format
    df = df.copy()
    df[outcome] = df[outcome].astype(bool)  # ensure binary
    df[outcome_time] = df[outcome_time].astype(float)  # ensure numeric

    # Extract traditional features
    traditional_features = ["age", "sex", "BMI"]
    traditional_df = df[traditional_features]

    # Extract PyPPG features
    pyppg_df = df[pyppg_features]

    # Create structured array for scikit-survival (for all model types)
    y_structured = Surv.from_dataframe(outcome, outcome_time, df)

    # Clean embedding_df to ensure no outcome variables are included
    papagei_df = embedding_df.copy()
    if outcome in papagei_df.columns:
        papagei_df = papagei_df.drop(columns=[outcome])
    if outcome_time in papagei_df.columns:
        papagei_df = papagei_df.drop(columns=[outcome_time])

    feature_sets = {}

    # M0: PaPaGei features alone
    m0_features = papagei_df.copy()
    m0_survival_df = m0_features.copy()
    m0_survival_df["time"] = df[outcome_time]
    m0_survival_df["event"] = df[outcome]
    feature_sets["M0_PaPaGei"] = (m0_features, y_structured, m0_survival_df)

    # M1: Traditional features alone
    m1_features = traditional_df.copy()
    m1_survival_df = m1_features.copy()
    m1_survival_df["time"] = df[outcome_time]
    m1_survival_df["event"] = df[outcome]
    feature_sets["M1_Traditional"] = (m1_features, y_structured, m1_survival_df)

    # M2: M0+M1 (PaPaGei + Traditional)
    m2_features = pd.concat([papagei_df, traditional_df], axis=1)
    m2_survival_df = m2_features.copy()
    m2_survival_df["time"] = df[outcome_time]
    m2_survival_df["event"] = df[outcome]
    feature_sets["M2_PaPaGei_Traditional"] = (m2_features, y_structured, m2_survival_df)

    # M3: PyPPG alone
    m3_features = pyppg_df.copy()
    m3_survival_df = m3_features.copy()
    m3_survival_df["time"] = df[outcome_time]
    m3_survival_df["event"] = df[outcome]
    feature_sets["M3_PyPPG"] = (m3_features, y_structured, m3_survival_df)

    # M4: M3+M1 (PyPPG + Traditional)
    m4_features = pd.concat([pyppg_df, traditional_df], axis=1)
    m4_survival_df = m4_features.copy()
    m4_survival_df["time"] = df[outcome_time]
    m4_survival_df["event"] = df[outcome]
    feature_sets["M4_PyPPG_Traditional"] = (m4_features, y_structured, m4_survival_df)

    return feature_sets


def split_and_scale_data(
    feature_sets: Dict[str, Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]],
    test_size: float = 0.3,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Split and scale each feature set for modeling.

    Args:
        feature_sets: Dictionary of feature sets
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Dictionary of split and scaled datasets for each feature set
    """
    results = {}

    for model_name, (features, y_structured, survival_df) in feature_sets.items():
        print(f"Processing {model_name}...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            y_structured,
            test_size=test_size,
            random_state=random_state,
            stratify=survival_df["event"],
        )

        # Split survival dataframe
        train_indices = X_train.index
        test_indices = X_test.index
        df_train = survival_df.loc[train_indices].copy()
        df_test = survival_df.loc[test_indices].copy()

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

        # Update survival dataframes with scaled features
        feature_cols = [col for col in df_train.columns if col not in ["time", "event"]]
        df_train[feature_cols] = X_train_scaled
        df_test[feature_cols] = X_test_scaled

        results[model_name] = {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "df_train": df_train,
            "df_test": df_test,
            "scaler": scaler,  # Save scaler for future use
        }

    return results


def fit_cox_models(
    data_dict: Dict[str, Dict[str, Any]], output_paths: Dict[str, str]
) -> Dict[str, Tuple[CoxPHFitter, float]]:
    """
    Fit Cox Proportional Hazards models for each feature set and save models.

    Args:
        data_dict: Dictionary of preprocessed datasets
        output_paths: Dictionary with paths to output directories

    Returns:
        Dictionary of fitted Cox models and their concordance indices
    """
    cox_models = {}

    for model_name, data in data_dict.items():
        print(f"\nFitting Cox model for {model_name}...")
        start_time = time.time()

        # Fit Cox model
        cph = CoxPHFitter()
        try:
            cph.fit(data["df_train"], duration_col="time", event_col="event")

            # Evaluate on test set
            c_index = concordance_index(
                data["df_test"]["time"],
                -cph.predict_partial_hazard(data["df_test"]),
                data["df_test"]["event"],
            )

            # Save model
            model_path = Path(output_paths["models"]) / f"{model_name}_cox.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(cph, f)

            # Save summary as CSV file
            summary_df = cph.summary.copy()
            # Add model name and c-index as metadata columns
            summary_df["model"] = model_name
            summary_df["c_index"] = c_index

            # Save to CSV
            csv_path = Path(output_paths["results"]) / f"{model_name}_cox_summary.csv"
            summary_df.to_csv(csv_path)

            # Also save a text version for quick inspection
            summary_path = (
                Path(output_paths["results"]) / f"{model_name}_cox_summary.txt"
            )
            with open(summary_path, "w") as f:
                f.write(f"Model: {model_name} (Cox Proportional Hazards)\n")
                f.write(f"Concordance Index: {c_index:.4f}\n\n")
                f.write(str(cph.summary))

            # Plot survival curves and save
            plt.figure(figsize=(12, 8))
            cph.plot_partial_effects_on_outcome(
                covariates=data["X_train"].columns[0],
                values=[data["X_train"][data["X_train"].columns[0]].mean()],
                cmap="coolwarm",
            )
            plt.title(f"{model_name} Cox Model - Survival Curves")
            plt.tight_layout()
            plot_path = (
                Path(output_paths["visualizations"])
                / f"{model_name}_cox_survival_curves.png"
            )
            plt.savefig(plot_path, dpi=300)
            plt.close()

            cox_models[model_name] = (cph, c_index)
            print(f"  - Concordance index: {c_index:.4f}")
            print(f"  - Training time: {time.time() - start_time:.2f} seconds")
            print(f"  - Model saved to: {model_path}")
            print(f"  - Summary CSV saved to: {csv_path}")
        except Exception as e:
            print(f"  - Error fitting Cox model: {e}")
            cox_models[model_name] = (None, 0.0)

    return cox_models


def fit_rsf_models(
    data_dict: Dict[str, Dict[str, Any]],
    output_paths: Dict[str, str],
    n_estimators: int = 100,
    max_depth: Optional[int] = 7,
    min_samples_split: int = 10,
    random_state: int = 42,
) -> Dict[str, Tuple[RandomSurvivalForest, float]]:
    """
    Fit Random Survival Forest models for each feature set and save models.

    Args:
        data_dict: Dictionary of preprocessed datasets
        output_paths: Dictionary with paths to output directories
        n_estimators: Number of trees
        max_depth: Maximum depth of trees
        min_samples_split: Minimum samples required to split a node
        random_state: Random seed for reproducibility

    Returns:
        Dictionary of fitted RSF models and their concordance indices
    """
    rsf_models = {}

    for model_name, data in data_dict.items():
        print(f"\nFitting Random Survival Forest for {model_name}...")
        start_time = time.time()

        # Fit Random Survival Forest
        rsf = RandomSurvivalForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,  # Use all available cores
        )

        try:
            rsf.fit(data["X_train"], data["y_train"])

            # Calculate concordance index
            c_index = rsf.score(data["X_test"], data["y_test"])

            # Save model
            model_path = Path(output_paths["models"]) / f"{model_name}_rsf.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(rsf, f)

            # Save feature importance
            importance_df = pd.DataFrame(
                {
                    "Feature": data["X_train"].columns,
                    "Importance": rsf.feature_importances_,
                }
            ).sort_values("Importance", ascending=False)

            importance_path = (
                Path(output_paths["results"])
                / f"{model_name}_rsf_feature_importance.csv"
            )
            importance_df.to_csv(importance_path, index=False)

            # Plot feature importance and save
            plt.figure(figsize=(12, 8))
            importance_df.head(20).sort_values("Importance").plot(
                kind="barh", x="Feature", y="Importance", figsize=(12, 8)
            )
            plt.title(f"{model_name} RSF - Feature Importance (Top 20)")
            plt.tight_layout()
            plot_path = (
                Path(output_paths["visualizations"])
                / f"{model_name}_rsf_feature_importance.png"
            )
            plt.savefig(plot_path, dpi=300)
            plt.close()

            # Plot survival curves for a few test samples
            plt.figure(figsize=(12, 8))
            for i in range(min(5, len(data["X_test"]))):
                surv_funcs = rsf.predict_survival_function(
                    data["X_test"].iloc[i : i + 1]
                )
                for surv_func in surv_funcs:
                    plt.step(
                        rsf.event_times_,
                        surv_func(rsf.event_times_),
                        where="post",
                        label=f"Sample {i+1}",
                    )

            plt.xlabel("Time")
            plt.ylabel("Survival Probability")
            plt.title(f"{model_name} RSF - Survival Curves for Sample Patients")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            surv_path = (
                Path(output_paths["visualizations"])
                / f"{model_name}_rsf_survival_curves.png"
            )
            plt.savefig(surv_path, dpi=300)
            plt.close()

            rsf_models[model_name] = (rsf, c_index)
            print(f"  - Concordance index: {c_index:.4f}")
            print(f"  - Training time: {time.time() - start_time:.2f} seconds")
            print(f"  - Model saved to: {model_path}")
        except Exception as e:
            print(f"  - Error fitting RSF model: {e}")
            rsf_models[model_name] = (None, 0.0)

    return rsf_models


def plot_kaplan_meier(
    df: pd.DataFrame, outcome: str, outcome_time: str, output_paths: Dict[str, str]
) -> None:
    """
    Create and save Kaplan-Meier plot for the dataset.

    Args:
        df: DataFrame with survival data
        outcome: Event indicator column name
        outcome_time: Time-to-event column name
        output_paths: Dictionary with paths to output directories
    """
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    kmf.fit(df[outcome_time], df[outcome], label="All Patients")
    kmf.plot_survival_function()

    plt.title("Kaplan-Meier Survival Curve")
    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    plt.tight_layout()

    km_path = Path(output_paths["visualizations"]) / "kaplan_meier_curve.png"
    plt.savefig(km_path, dpi=300)
    plt.close()

    print(f"Saved Kaplan-Meier plot to: {km_path}")


def compare_models(
    cox_models: Dict[str, Tuple[CoxPHFitter, float]],
    rsf_models: Dict[str, Tuple[RandomSurvivalForest, float]],
    output_paths: Dict[str, str],
) -> pd.DataFrame:
    """
    Create a comparison table of model performance and save results.

    Args:
        cox_models: Dictionary of fitted Cox models and their concordance indices
        rsf_models: Dictionary of fitted RSF models and their concordance indices
        output_paths: Dictionary with paths to output directories

    Returns:
        DataFrame with performance comparison
    """
    results = []

    for model_name in cox_models.keys():
        cox_model, cox_c_index = cox_models.get(model_name, (None, 0.0))
        rsf_model, rsf_c_index = rsf_models.get(model_name, (None, 0.0))

        results.append(
            {
                "Feature_Set": model_name,
                "Cox_C_Index": cox_c_index,
                "RSF_C_Index": rsf_c_index,
                "Best_Model": "Cox" if cox_c_index > rsf_c_index else "RSF",
                "Best_C_Index": max(cox_c_index, rsf_c_index),
            }
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Best_C_Index", ascending=False)

    # Save comparison table
    comparison_path = Path(output_paths["results"]) / "model_comparison.csv"
    results_df.to_csv(comparison_path, index=False)

    # Create bar plot of performance by feature set
    plt.figure(figsize=(14, 8))

    x = np.arange(len(results_df))
    width = 0.35

    plt.bar(x - width / 2, results_df["Cox_C_Index"], width, label="Cox PH")
    plt.bar(x + width / 2, results_df["RSF_C_Index"], width, label="RSF")

    plt.xlabel("Feature Set")
    plt.ylabel("Concordance Index")
    plt.title("Model Performance by Feature Set")
    plt.xticks(x, results_df["Feature_Set"], rotation=45, ha="right")
    plt.ylim(0.5, 1.0)  # C-index is typically between 0.5 and 1.0
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()

    plot_path = Path(output_paths["visualizations"]) / "model_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved model comparison to: {comparison_path}")
    print(f"Saved comparison plot to: {plot_path}")

    return results_df


def main() -> None:
    """Main function to run the survival analysis pipeline."""
    # Create output directories
    output_paths = create_output_directories()

    # Load data
    df = pd.read_parquet(
        f"{os.getenv('BIOBANK_DATA_PATH')}/215K_pyppg_features_and_conditions.parquet"
    )
    df = df.dropna()  # Drop rows with missing values

    outcome = "MACE"
    outcome_time = "MACE_days"

    # Define PyPPG features (replace with actual column names)
    # Example: PYPPG_FEATURES = ["pyppg_feature1", "pyppg_feature2", ...]
    # If you have these in a constant, just use that instead

    # Get embeddings (PaPaGei features)
    print("Loading embeddings...")
    embeddings = get_embeddings(df, cache_file="embeddings.npy")

    print("Creating embedding dataframe...")
    embedding_df = get_embedding_df(embeddings)

    print("Removing highly correlated features...")
    embedding_df = remove_highly_corr_features(embedding_df)

    # Create Kaplan-Meier plot for overall survival
    print("Creating Kaplan-Meier plot...")
    plot_kaplan_meier(df, outcome, outcome_time, output_paths)

    # Create feature sets
    print("Setting up feature sets...")
    feature_sets = setup_feature_sets(
        df=df,
        outcome=outcome,
        outcome_time=outcome_time,
        embedding_df=embedding_df,
        pyppg_features=PYPPG_FEATURES,
    )

    # Save experiment configuration
    experiment_config = {
        "outcome": outcome,
        "outcome_time": outcome_time,
        "n_samples": len(df),
        "event_rate": df[outcome].mean(),
        "feature_sets": {
            name: {"n_features": len(features.columns)}
            for name, (features, _, _) in feature_sets.items()
        },
    }

    config_path = Path(output_paths["results"]) / "experiment_config.txt"
    with open(config_path, "w") as f:
        f.write("Experiment Configuration:\n")
        f.write(f"Outcome: {experiment_config['outcome']}\n")
        f.write(f"Time variable: {experiment_config['outcome_time']}\n")
        f.write(f"Number of samples: {experiment_config['n_samples']}\n")
        f.write(f"Event rate: {experiment_config['event_rate']:.4f}\n\n")
        f.write("Feature sets:\n")
        for name, info in experiment_config["feature_sets"].items():
            f.write(f"  {name}: {info['n_features']} features\n")

    # Split and scale data
    print("Splitting and scaling data...")
    data_dict = split_and_scale_data(feature_sets)

    # Fit Cox models
    print("Fitting Cox Proportional Hazards models...")
    cox_models = fit_cox_models(data_dict, output_paths)

    # Fit RSF models
    print("Fitting Random Survival Forest models...")
    rsf_models = fit_rsf_models(data_dict, output_paths)

    # Compare models
    print("\nModel Comparison:")
    comparison_df = compare_models(cox_models, rsf_models, output_paths)
    print(comparison_df)

    # Display detailed results for the best model
    best_model = comparison_df.iloc[0]["Feature_Set"]
    best_type = comparison_df.iloc[0]["Best_Model"]

    print(f"\nBest model: {best_model} using {best_type}")

    if best_type == "Cox":
        best_cox, _ = cox_models[best_model]
        print("\nCox model summary for best model:")
        print(best_cox.summary)
    else:
        best_rsf, _ = rsf_models[best_model]
        print("\nRSF feature importance for best model (top 10):")
        importance = pd.DataFrame(
            {
                "Feature": data_dict[best_model]["X_train"].columns,
                "Importance": best_rsf.feature_importances_,
            }
        ).sort_values("Importance", ascending=False)
        print(importance.head(10))

    # Save all dataset splits for reproducibility
    splits_path = Path(output_paths["results"]) / "dataset_splits.pkl"
    with open(splits_path, "wb") as f:
        pickle.dump(data_dict, f)

    print(f"\nAll results saved to: {output_paths['base']}")
    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
