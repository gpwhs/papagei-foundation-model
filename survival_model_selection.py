"""
Utility functions for survival model selection and parameter tuning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

from sklearn.model_selection import KFold, ParameterGrid
from lifelines.utils import concordance_index


def evaluate_cox_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[float, Any]:
    """
    Evaluate a Cox PH model with given parameters.

    Args:
        X_train: Training features
        y_train: Training outcomes (structured array)
        X_val: Validation features
        y_val: Validation outcomes (structured array)
        params: Model parameters

    Returns:
        Tuple of (c_index, fitted_model)
    """
    from lifelines import CoxPHFitter

    # Create model with parameters
    model = CoxPHFitter(
        penalizer=params.get("penalizer", 0.0), l1_ratio=params.get("l1_ratio", 0.0)
    )

    # Prepare data for lifelines
    train_df = pd.DataFrame(
        {
            "T": y_train["time"],
            "E": y_train["event"],
        }
    )
    train_df = pd.concat([train_df, X_train], axis=1)

    # Fit model
    model.fit(train_df, duration_col="T", event_col="E")

    # Evaluate on validation set
    val_df = pd.DataFrame(
        {
            "T": y_val["time"],
            "E": y_val["event"],
        }
    )
    val_df = pd.concat([val_df, X_val], axis=1)

    c_index = model.score(val_df, scoring_method="concordance_index")

    return c_index, model


def evaluate_rsf_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[float, Any]:
    """
    Evaluate a Random Survival Forest model with given parameters.

    Args:
        X_train: Training features
        y_train: Training outcomes (structured array)
        X_val: Validation features
        y_val: Validation outcomes (structured array)
        params: Model parameters

    Returns:
        Tuple of (c_index, fitted_model)
    """
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import concordance_index_censored

    # Create model with parameters
    model = RandomSurvivalForest(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", None),
        min_samples_split=params.get("min_samples_split", 2),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        max_features=params.get("max_features", "sqrt"),
        random_state=42,
    )

    # Fit model
    model.fit(X_train, y_train)

    # Predict on validation set
    risk_scores = model.predict(X_val)

    # Calculate c-index
    c_index = concordance_index_censored(y_val["event"], y_val["time"], risk_scores)[0]

    return c_index, model


def evaluate_xgbse_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[float, Any]:
    """
    Evaluate an XGBSE model with given parameters.

    Args:
        X_train: Training features
        y_train: Training outcomes (structured array)
        X_val: Validation features
        y_val: Validation outcomes (structured array)
        params: Model parameters

    Returns:
        Tuple of (c_index, fitted_model)
    """
    try:
        from xgbse import XGBSEDebiasedBCE

        # Create model with parameters
        model = XGBSEDebiasedBCE(
            learning_rate=params.get("learning_rate", 0.1),
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 3),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            min_child_weight=params.get("min_child_weight", 1),
            gamma=params.get("gamma", 0),
            reg_alpha=params.get("reg_alpha", 0),
            reg_lambda=params.get("reg_lambda", 1),
            random_state=42,
        )

        # Prepare data
        train_df = pd.DataFrame(
            {
                "time": y_train["time"],
                "event": y_train["event"],
            }
        )

        # Fit model
        model.fit(
            X_train,
            time=train_df["time"],
            event=train_df["event"],
        )

        # Predict on validation set
        risk_scores = model.predict(X_val)

        # Convert to array for concordance_index
        val_times = y_val["time"]
        val_events = y_val["event"]

        # Calculate c-index (negative risk_scores because higher risk = shorter survival)
        c_index = concordance_index(val_times, -risk_scores, val_events)

        return c_index, model

    except ImportError:
        # Fall back to custom implementation
        from xgbse_utils import XGBSEDebiasedBCE

        # Create model with parameters
        model = XGBSEDebiasedBCE(
            learning_rate=params.get("learning_rate", 0.1),
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 3),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            min_child_weight=params.get("min_child_weight", 1),
            gamma=params.get("gamma", 0),
            reg_alpha=params.get("reg_alpha", 0),
            reg_lambda=params.get("reg_lambda", 1),
            random_state=42,
        )

        # Prepare data
        train_df = pd.DataFrame(
            {
                "time": y_train["time"],
                "event": y_train["event"],
            }
        )

        # Fit model
        model.fit(
            X_train,
            train_df["time"],
            train_df["event"],
        )

        # Predict on validation set
        risk_scores = model.predict(X_val)

        # Convert to array for concordance_index
        val_times = y_val["time"]
        val_events = y_val["event"]

        # Calculate c-index (negative risk_scores because higher risk = shorter survival)
        c_index = concordance_index(val_times, -risk_scores, val_events)

        return c_index, model


def tune_survival_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_type: str,
    param_grid: Dict[str, List[Any]],
    n_folds: int = 3,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Any]:
    """
    Tune a survival model using cross-validation.

    Args:
        X_train: Training features
        y_train: Training outcomes
        model_type: Type of model ("cox_ph", "rsf", or "xgbse")
        param_grid: Parameter grid to search
        n_folds: Number of cross-validation folds
        random_state: Random state for reproducibility

    Returns:
        Tuple of (best_params, best_model)
    """
    # Choose evaluation function based on model type
    if model_type == "cox_ph":
        evaluate_fn = evaluate_cox_model
    elif model_type == "rsf":
        evaluate_fn = evaluate_rsf_model
    elif model_type == "xgbse":
        evaluate_fn = evaluate_xgbse_model
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Create parameter grid
    grid = list(ParameterGrid(param_grid))
    print(f"Searching over {len(grid)} parameter combinations")

    # Setup cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    best_score = -np.inf
    best_params = None

    # Iterate over parameter combinations
    for params in grid:
        print(f"Evaluating parameters: {params}")
        fold_scores = []

        # Cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            print(f"  Fold {fold+1}/{n_folds}")

            # Split data
            X_train_fold = X_train.iloc[train_idx]
            y_train_fold = y_train[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_val_fold = y_train[val_idx]

            # Evaluate model
            try:
                score, _ = evaluate_fn(
                    X_train_fold, y_train_fold, X_val_fold, y_val_fold, params
                )
                fold_scores.append(score)
                print(f"    C-index: {score:.4f}")
            except Exception as e:
                print(f"    Error: {e}")
                fold_scores.append(0.0)

        # Calculate mean score
        mean_score = np.mean(fold_scores)
        print(f"  Mean C-index: {mean_score:.4f}")

        # Update best parameters if better
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            print(f"  New best parameters!")

    print(f"Best parameters: {best_params}, C-index: {best_score:.4f}")

    # Train final model with best parameters
    if model_type == "cox_ph":
        from lifelines import CoxPHFitter

        best_model = CoxPHFitter(
            penalizer=best_params.get("penalizer", 0.0),
            l1_ratio=best_params.get("l1_ratio", 0.0),
        )

        # Prepare data for lifelines
        train_df = pd.DataFrame(
            {
                "T": y_train["time"],
                "E": y_train["event"],
            }
        )
        train_df = pd.concat([train_df, X_train], axis=1)

        # Fit model
        best_model.fit(train_df, duration_col="T", event_col="E")

    elif model_type == "rsf":
        from sksurv.ensemble import RandomSurvivalForest

        best_model = RandomSurvivalForest(
            n_estimators=best_params.get("n_estimators", 100),
            max_depth=best_params.get("max_depth", None),
            min_samples_split=best_params.get("min_samples_split", 2),
            min_samples_leaf=best_params.get("min_samples_leaf", 1),
            max_features=best_params.get("max_features", "sqrt"),
            random_state=42,
        )
        best_model.fit(X_train, y_train)

    elif model_type == "xgbse":
        try:
            from xgbse import XGBSEDebiasedBCE

            best_model = XGBSEDebiasedBCE(
                learning_rate=best_params.get("learning_rate", 0.1),
                n_estimators=best_params.get("n_estimators", 100),
                max_depth=best_params.get("max_depth", 3),
                subsample=best_params.get("subsample", 0.8),
                colsample_bytree=best_params.get("colsample_bytree", 0.8),
                min_child_weight=best_params.get("min_child_weight", 1),
                gamma=best_params.get("gamma", 0),
                reg_alpha=best_params.get("reg_alpha", 0),
                reg_lambda=best_params.get("reg_lambda", 1),
                random_state=42,
            )

            train_df = pd.DataFrame(
                {
                    "time": y_train["time"],
                    "event": y_train["event"],
                }
            )

            best_model.fit(
                X_train,
                time=train_df["time"],
                event=train_df["event"],
            )
        except ImportError:
            from xgbse_utils import XGBSEDebiasedBCE

            best_model = XGBSEDebiasedBCE(
                learning_rate=best_params.get("learning_rate", 0.1),
                n_estimators=best_params.get("n_estimators", 100),
                max_depth=best_params.get("max_depth", 3),
                subsample=best_params.get("subsample", 0.8),
                colsample_bytree=best_params.get("colsample_bytree", 0.8),
                min_child_weight=best_params.get("min_child_weight", 1),
                gamma=best_params.get("gamma", 0),
                reg_alpha=best_params.get("reg_alpha", 0),
                reg_lambda=best_params.get("reg_lambda", 1),
                random_state=42,
            )

            train_df = pd.DataFrame(
                {
                    "time": y_train["time"],
                    "event": y_train["event"],
                }
            )

            best_model.fit(
                X_train,
                train_df["time"],
                train_df["event"],
            )

    return best_params, best_model
