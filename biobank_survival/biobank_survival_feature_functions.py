import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Dict, Any
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from statsmodels.stats.outliers_influence import variance_inflation_factor
from biobank_experiment_constants import FULL_PYPPG_FEATURES


def get_univariate_cox_pvalue(
    feature_series: pd.Series, time_series: pd.Series, event_series: pd.Series
) -> Tuple[float, float]:
    """
    Compute the p-value from a univariate Cox regression of a single feature.

    Args:
        feature_series: Feature values
        time_series: Time-to-event values
        event_series: Event indicator (1=event, 0=censored)

    Returns:
        p-value for the feature, c_index
    """
    try:
        # Normalize the feature for numerical stability
        feature_mean = feature_series.mean()
        feature_std = feature_series.std()

        if feature_std > 0:
            feature_norm = (feature_series - feature_mean) / feature_std
        else:
            # If std is 0, use the original feature (though it's not informative)
            feature_norm = feature_series - feature_mean

        # Create a DataFrame for Cox regression
        df = pd.DataFrame(
            {"feature": feature_norm, "time": time_series, "event": event_series}
        )

        # Fit Cox model
        cph = CoxPHFitter(penalizer=0.0)
        cph.fit(df, duration_col="time", event_col="event")

        # Get p-value
        summary = cph.summary
        p_value = summary.loc["feature", "p"]
        c_index = cph.concordance_index_

        return p_value, c_index
    except Exception as e:
        print(f"Error in univariate Cox regression: {e}")
        return 1.0  # Return 1.0 (not significant) in case of error


def remove_highly_correlated_features(
    df: pd.DataFrame,
    time_series: pd.Series,
    event_series: pd.Series,
    corr_threshold: float = 0.9,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Identify pairs of features with a correlation above corr_threshold.
    For each pair, drop the feature with the higher p-value in a univariate Cox regression.

    Args:
        df: DataFrame containing features
        time_series: Series with time-to-event values
        event_series: Series with event indicators (1=event, 0=censored)
        corr_threshold: Correlation threshold for feature dropping

    Returns:
        Tuple of (reduced_dataframe, list_of_dropped_features)
    """

    corr_matrix = df.corr().abs()
    # Use the upper triangle of the correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    features_to_drop = set()

    for col in tqdm(upper_tri.columns, desc="Checking feature correlations"):
        for row in upper_tri.index:
            # Skip NaNs in the upper triangle
            if pd.isna(upper_tri.loc[row, col]):
                continue
            if upper_tri.loc[row, col] > corr_threshold:
                # Get the univariate Cox p-value and c-index for each feature
                pval_col, cindex_col = get_univariate_cox_pvalue(
                    df[col], time_series, event_series
                )
                pval_row, cindex_row = get_univariate_cox_pvalue(
                    df[row], time_series, event_series
                )

                print(
                    f"Feature {col}: p-value {pval_col:.4f}, c-index {cindex_col:.4f}"
                )
                print(
                    f"Feature {row}: p-value {pval_row:.4f}, c-index {cindex_row:.4f}"
                )

                # Decision rule:
                # Case 1: Both features are statistically significant
                if pval_col < 0.05 and pval_row < 0.05:
                    if cindex_col < cindex_row:
                        features_to_drop.add(col)
                        print(f"  Dropping {col} (lower c-index)")
                    else:
                        features_to_drop.add(row)
                        print(f"  Dropping {row} (lower c-index)")
                # Case 2: One feature is significant and the other is not
                elif pval_col >= 0.05 and pval_row < 0.05:
                    features_to_drop.add(col)
                    print(f"  Dropping {col} (not significant)")
                elif pval_row >= 0.05 and pval_col < 0.05:
                    features_to_drop.add(row)
                    print(f"  Dropping {row} (not significant)")
                # Case 3: Neither feature is significant – drop the one with the higher p-value
                else:
                    if pval_col > pval_row:
                        features_to_drop.add(col)
                        print(f"  Dropping {col} (higher p-value)")
                    else:
                        features_to_drop.add(row)
                        print(f"  Dropping {row} (higher p-value)")

    df_reduced = df.drop(columns=list(features_to_drop))
    return df_reduced, list(features_to_drop)


def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Variance Inflation Factor (VIF) for each feature in the DataFrame X.

    Args:
        X: Feature DataFrame

    Returns:
        DataFrame with feature names and VIF values
    """
    vif_df = pd.DataFrame()
    vif_df["feature"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_df


def remove_high_vif_features(
    X: pd.DataFrame, vif_threshold: float = 5.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Iteratively remove features with VIF above the specified threshold.

    Args:
        X: Feature DataFrame
        vif_threshold: VIF threshold for feature removal

    Returns:
        Tuple of (reduced_dataframe, vif_dataframe)
    """
    X_reduced = X.copy()
    if os.path.exists("final_pyppg_features_post_corr.txt"):
        with open("final_pyppg_features_post_corr.txt", "r") as f:
            final_features = f.read().splitlines()
        X_reduced = X_reduced[final_features]
        return X_reduced, None

    while True:
        vif_df = calculate_vif(X_reduced)
        max_vif = vif_df["VIF"].max()

        if max_vif > vif_threshold:
            feature_to_drop = vif_df.sort_values("VIF", ascending=False)[
                "feature"
            ].iloc[0]
            print(f"Dropping {feature_to_drop} with VIF: {max_vif:.2f}")
            X_reduced = X_reduced.drop(columns=[feature_to_drop])
        else:
            break

    # save list of final columns to a file
    with open("final_pyppg_features_post_vif.txt", "w") as f:
        for col in X_reduced.columns:
            f.write(f"{col}\n")
    return X_reduced, vif_df


def select_features_for_survival(
    X: pd.DataFrame,
    time: pd.Series,
    event: pd.Series,
    corr_threshold: float = 0.9,
    vif_threshold: float = 5.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform feature selection specialized for survival analysis.

    Steps:
    1. Remove highly correlated features based on univariate Cox p-values
    2. Remove features with high VIF (multicollinearity)
    3. Check proportional hazards assumption (optional)

    Args:
        X: Feature DataFrame
        time: Time-to-event Series
        event: Event indicator Series
        corr_threshold: Correlation threshold
        vif_threshold: VIF threshold
        check_ph_assumption: Whether to check the proportional hazards assumption

    Returns:
        Tuple of (selected_features_df, selection_info_dict)
    """
    selection_info = {
        "n_features_original": X.shape[1],
        "dropped_correlated": [],
        "dropped_vif": [],
        "ph_assumption_violations": [],
    }

    # Step 1: Remove highly correlated features
    print(f"Starting feature selection with {X.shape[1]} features")
    X_reduced, dropped_corr = remove_highly_correlated_features(
        X, time, event, corr_threshold
    )
    selection_info["dropped_correlated"] = dropped_corr
    print(
        f"Removed {len(dropped_corr)} correlated features, {X_reduced.shape[1]} remaining"
    )

    # Step 2: Remove features with high VIF
    X_reduced_vif, _ = remove_high_vif_features(X_reduced, vif_threshold)
    dropped_vif = list(set(X_reduced.columns) - set(X_reduced_vif.columns))
    selection_info["dropped_vif"] = dropped_vif
    print(
        f"Removed {len(dropped_vif)} high-VIF features, {X_reduced_vif.shape[1]} remaining"
    )

    # Return the selected features and selection info
    selection_info["n_features_selected"] = X_reduced_vif.shape[1]

    return X_reduced_vif, selection_info


def preprocess_survival_data(
    df: pd.DataFrame,
    outcome: str,
    time_column: str,
    embedding_df: pd.DataFrame,
    corr_threshold: float = 0.9,
    vif_threshold: float = 5.0,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Preprocess data specifically for survival analysis.

    Args:
        df: Source DataFrame with outcome and time columns
        outcome: Name of the event indicator column (1=event, 0=censored)
        time_column: Name of the time-to-event column
        embedding_df: DataFrame with pre-computed embeddings (post pca)
        feature_selection: Whether to perform feature selection
        corr_threshold: Correlation threshold for feature selection
        vif_threshold: VIF threshold for feature selection

    Returns:
        Tuple of (features_df, time_series, event_series)
    """
    print(len(df))
    # Extract time and event information
    time_data = df[time_column]
    event_data = df[outcome]
    # this treatment of dataframes is kinda janky
    # Extract traditional features
    traditional_features = ["age", "sex", "BMI"]
    traditional_df = df[traditional_features].copy()
    if outcome or time_column in embedding_df.columns:
        embedding_df = embedding_df.drop(columns=[outcome, time_column])
    pyppg_columns = [col for col in FULL_PYPPG_FEATURES if col in df.columns]
    pyppg_df = df[pyppg_columns].copy()
    print("performing feature reduction on pyppg features")
    if os.path.exists("final_pyppg_features_post_vif.txt"):
        print("loading previous pyppg features from file")
        with open("final_pyppg_features_post_vif.txt", "r") as f:
            final_features = f.read().splitlines()
        pyppg_df_reduced = pyppg_df[final_features]
    else:
        pyppg_df_reduced, _ = select_features_for_survival(
            pyppg_df, time_data, event_data, corr_threshold, vif_threshold
        )
    print(f"length of embedding_df: {len(embedding_df)}")
    print(f"length of traditional_df: {len(traditional_df)}")
    print(f"length of pyppg_df_reduced: {len(pyppg_df_reduced)}")

    combined_features = pd.concat(
        [embedding_df, traditional_df, pyppg_df_reduced], axis=1
    )
    print(f"Final feature set shape: {combined_features.shape}")
    print(f"Time series shape: {time_data.shape}")
    print(f"Event series shape: {event_data.shape}")

    return combined_features, time_data, event_data
