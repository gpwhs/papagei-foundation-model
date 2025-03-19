import os
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.model_selection import train_test_split

# Import the feature selection functions
from biobank_survival_feature_functions import preprocess_survival_data
from biobank_experiment_utils import get_embedding_df


def prepare_survival_data(
    df: pd.DataFrame,
    outcome: str,
    time_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Dict[str, Any]:
    """
    Prepare data for survival analysis, including train/test splitting.

    Args:
        df: Source DataFrame
        outcome: Event indicator column name (1=event, 0=censored)
        time_column: Time-to-event column name
        embedding_df: Optional pre-computed embeddings
        test_size: Proportion of data to use for testing
        random_state: Random seed
        stratify: Whether to stratify by event indicator
        feature_selection: Whether to perform feature selection

    Returns:
        Dictionary containing the prepared data
    """
    # Check if required columns exist
    if outcome not in df.columns:
        raise ValueError(f"Outcome column '{outcome}' not found in dataset")
    if time_column not in df.columns:
        raise ValueError(f"Time column '{time_column}' not found in dataset")
    embedding_df = get_embedding_df(df, outcome, time_column)

    # Preprocess features specifically for survival analysis
    all_features, time_series, event_series = preprocess_survival_data(
        df=df,
        outcome=outcome,
        time_column=time_column,
        embedding_df=embedding_df,
    )
    print(f"Preprocessed data shape: {all_features.shape}")
    print(f"time_series shape: {time_series.shape}")
    print(f"event_series shape: {event_series.shape}")

    # Create train/test split
    if stratify:
        # Stratify by event indicator to ensure balanced event distribution
        X_train, X_test, y_train, y_test, time_train, time_test = train_test_split(
            all_features,
            event_series,
            time_series,
            test_size=test_size,
            random_state=random_state,
            stratify=event_series,
        )
    else:
        # Regular random split
        X_train, X_test, y_train, y_test, time_train, time_test = train_test_split(
            all_features,
            event_series,
            time_series,
            test_size=test_size,
            random_state=random_state,
        )

    # Display information about the train/test split
    print("\nTrain/Test Split:")
    print(f"Training samples: {len(X_train)} ({y_train.mean()*100:.1f}% events)")
    print(f"Testing samples: {len(X_test)} ({y_test.mean()*100:.1f}% events)")

    # Return all relevant data
    return {
        "X_train": X_train,
        "X_test": X_test,
        "event_train": y_train,
        "event_test": y_test,
        "time_train": time_train,
        "time_test": time_test,
        "feature_names": all_features.columns.tolist(),
        "outcome": outcome,
        "time_column": time_column,
    }


def save_processed_survival_data(data_dict: Dict[str, Any], output_dir: str) -> None:
    """
    Save processed survival data to disk.

    Args:
        data_dict: Dictionary with processed data
        output_dir: Directory to save the data
    """
    import joblib

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save the data dictionary
    joblib.dump(data_dict, f"{output_dir}/survival_data.joblib")

    # Also save individual components for easier access
    joblib.dump(data_dict["X_train"], f"{output_dir}/X_train.joblib")
    joblib.dump(data_dict["X_test"], f"{output_dir}/X_test.joblib")
    joblib.dump(data_dict["event_train"], f"{output_dir}/event_train.joblib")
    joblib.dump(data_dict["event_test"], f"{output_dir}/event_test.joblib")
    joblib.dump(data_dict["time_train"], f"{output_dir}/time_train.joblib")
    joblib.dump(data_dict["time_test"], f"{output_dir}/time_test.joblib")

    # Save feature names
    pd.Series(data_dict["feature_names"]).to_csv(
        f"{output_dir}/feature_names.csv", index=False, header=["feature_name"]
    )

    print(f"Saved processed survival data to {output_dir}")


def load_processed_survival_data(input_dir: str) -> Dict[str, Any]:
    """
    Load processed survival data from disk.

    Args:
        input_dir: Directory containing the saved data

    Returns:
        Dictionary with processed data
    """
    import joblib

    data_path = f"{input_dir}/survival_data.joblib"

    if os.path.exists(data_path):
        # Load the complete data dictionary
        data_dict = joblib.load(data_path)
        return data_dict
    else:
        # Try to load individual files
        try:
            data_dict = {
                "X_train": joblib.load(f"{input_dir}/X_train.joblib"),
                "X_test": joblib.load(f"{input_dir}/X_test.joblib"),
                "event_train": joblib.load(f"{input_dir}/event_train.joblib"),
                "event_test": joblib.load(f"{input_dir}/event_test.joblib"),
                "time_train": joblib.load(f"{input_dir}/time_train.joblib"),
                "time_test": joblib.load(f"{input_dir}/time_test.joblib"),
            }

            # Load feature names if available
            if os.path.exists(f"{input_dir}/feature_names.csv"):
                feature_names = pd.read_csv(f"{input_dir}/feature_names.csv")[
                    "feature_name"
                ].tolist()
                data_dict["feature_names"] = feature_names

            return data_dict
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load survival data from {input_dir}: {e}"
            )
