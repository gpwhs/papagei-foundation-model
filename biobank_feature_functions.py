import pandas as pd
from tqdm import tqdm
import numpy as np
from statsmodels.api import Logit, add_constant
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from statsmodels.stats.outliers_influence import variance_inflation_factor


def get_univariate_pvalue(feature_series, target_series):
    """Compute the p-value from a univariate logistic regression of a single feature."""
    try:
        # need to normalize the feature for the logistic regression
        feature_series = (feature_series - feature_series.mean()) / feature_series.std()

        X_const = add_constant(feature_series)
        model = Logit(target_series, X_const).fit(disp=0)
        # p-value for the feature (index 1 because index 0 is the constant)
        return model.pvalues.iloc[1]
    except PerfectSeparationError:
        return 1.0  # if perfect separation occurs, treat as non-significant
    except Exception as e:
        print("Error computing univariate p-value:", e)
        return 1.0


def remove_highly_correlated_features(df, target, corr_threshold=0.9):
    """
    Identify pairs of features with a correlation above corr_threshold.
    For each pair, drop the feature with the higher p-value in a univariate logistic regression.
    """
    corr_matrix = df.corr().abs()
    # Use the upper triangle of the correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    features_to_drop = set()

    for col in tqdm(upper_tri.columns):
        for row in upper_tri.index:
            # Skip NaNs in the upper triangle
            if pd.isna(upper_tri.loc[row, col]):
                continue
            if upper_tri.loc[row, col] > corr_threshold:
                # Compare univariate significance
                pval_col = get_univariate_pvalue(df[col], target)
                pval_row = get_univariate_pvalue(df[row], target)
                # Drop the feature with the higher p-value
                print(f"Feature {col} has p-value {pval_col:.4f}")
                if pval_col > pval_row:
                    features_to_drop.add(col)
                else:
                    features_to_drop.add(row)

    df_reduced = df.drop(columns=list(features_to_drop))
    return df_reduced, list(features_to_drop)


def calculate_vif(X: pd.DataFrame):
    """
    Calculate the Variance Inflation Factor (VIF) for each feature in the DataFrame X.
    """
    vif_df = pd.DataFrame()
    vif_df["feature"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_df


def remove_high_vif_features(X, vif_threshold=5.0):
    """
    Iteratively remove features with VIF above the specified threshold.
    """
    while True:
        vif_df = calculate_vif(X)
        max_vif = vif_df["VIF"].max()
        if max_vif > vif_threshold:
            feature_to_drop = vif_df.sort_values("VIF", ascending=False)[
                "feature"
            ].iloc[0]
            print(f"Dropping {feature_to_drop} with VIF: {max_vif:.2f}")
            X = X.drop(columns=[feature_to_drop])
        else:
            break
    return X, vif_df
