"""
Utility functions for XGBoost Survival Embeddings (XGBSE).
This module provides a simplified implementation for XGBSE functionality,
which can be used if the full XGBSE package is not available.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

try:
    import xgboost as xgb
except ImportError:
    raise ImportError(
        "XGBoost is required for XGBSE. Install with: pip install xgboost"
    )


class XGBSEBaseEstimator:
    """Base class for XGBSE estimators."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        max_depth: int = 3,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        gamma: float = 0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        random_state: int = 42,
    ):
        """Initialize XGBSE base estimator.

        Args:
            learning_rate: Step size shrinkage used to prevent overfitting
            n_estimators: Number of gradient boosted trees
            max_depth: Maximum tree depth
            subsample: Subsample ratio of the training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            min_child_weight: Minimum sum of instance weight needed in a child
            gamma: Minimum loss reduction required to make a further partition
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            random_state: Random seed
        """
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state

        # Initialize model
        self.model = None
        self.time_bins = None

    def fit(
        self,
        X: pd.DataFrame,
        time: Union[pd.Series, np.ndarray] = None,
        event: Union[pd.Series, np.ndarray] = None,
        time_bins: int = 10,
        **kwargs,
    ) -> "XGBSEBaseEstimator":
        """Fit the model.

        Args:
            X: Feature matrix
            time: Time-to-event or censoring
            event: Event indicator (1=event, 0=censored)
            time_bins: Number of time bins for discrete-time survival

        Returns:
            Self
        """
        # Handle alternative parameter formats for compatibility
        if time is None and "time" in kwargs:
            time = kwargs["time"]
        if event is None and "event" in kwargs:
            event = kwargs["event"]

        if time is None or event is None:
            raise ValueError("Both time and event must be provided")

        # Create time bins for discrete-time modeling
        self.time_bins = np.linspace(0, np.max(time) * 1.01, time_bins + 1)

        # Instead of using survival:cox, we'll use a standard binary classification approach
        # Higher risk score = higher hazard = shorter survival

        # Use standard binary classification with rank-based approach
        self.model = xgb.XGBClassifier(
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            eval_metric="logloss",
        )

        # Fit to event status (basic binary classification)
        self.model.fit(X, event)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict risk scores.

        Args:
            X: Feature matrix

        Returns:
            Risk scores (higher values indicate higher risk)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Use predicted probabilities as risk scores
        return self.model.predict_proba(X)[:, 1]

    def predict_survival_function(self, X: pd.DataFrame) -> Callable:
        """Predict survival function.

        This is a simplified implementation that returns a step function
        based on the risk score.

        Args:
            X: Feature matrix (one row only)

        Returns:
            Callable function that takes time as input and returns survival probability
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get risk score
        risk_score = self.predict(X)

        # Convert to scalar if single row
        if isinstance(risk_score, np.ndarray) and len(risk_score) == 1:
            risk_score = risk_score[0]

        # Create time points for the survival function
        times = np.linspace(0, self.time_bins[-1], 100)

        # Simple exponential survival function based on risk score
        # S(t) = exp(-H(t)), where H(t) = risk_score * t / max_time
        survival_probs = np.exp(-risk_score * times / self.time_bins[-1])

        # Create a step function object
        class StepFunction:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __call__(self, t):
                if np.isscalar(t):
                    # Find closest time point
                    idx = np.argmin(np.abs(self.x - t))
                    return self.y[idx]
                else:
                    # Vectorized version
                    result = np.zeros_like(t, dtype=float)
                    for i, ti in enumerate(t):
                        idx = np.argmin(np.abs(self.x - ti))
                        result[i] = self.y[idx]
                    return result

        return StepFunction(times, survival_probs)


class XGBSEDebiasedBCE(XGBSEBaseEstimator):
    """XGBSE with Debiased Binary Cross Entropy."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        max_depth: int = 3,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        gamma: float = 0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        random_state: int = 42,
    ):
        """Initialize XGBSE with Debiased BCE."""
        super().__init__(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
        )


def rsf_grid_search(
    params: Dict[str, List[Any]],
    X_train: pd.DataFrame,
    T_train: pd.Series,
    E_train: pd.Series,
    n_splits: int = 3,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], XGBSEDebiasedBCE]:
    """Simple grid search for XGBSE models.

    Args:
        params: Dictionary of parameter grids
        X_train: Training features
        T_train: Training times
        E_train: Training events
        n_splits: Number of cross-validation splits
        random_state: Random seed

    Returns:
        Tuple of (best_params, best_model)
    """
    from sklearn.model_selection import ParameterGrid, KFold
    from lifelines.utils import concordance_index

    # Create parameter grid
    param_grid = list(ParameterGrid(params))
    print(f"Grid search with {len(param_grid)} parameter combinations")

    # Setup cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    best_score = -np.inf
    best_params = None

    # Iterate over parameter combinations
    for param_dict in param_grid:
        scores = []

        # Cross-validation
        for train_idx, val_idx in kf.split(X_train):
            # Split data
            X_train_cv = X_train.iloc[train_idx]
            T_train_cv = T_train.iloc[train_idx]
            E_train_cv = E_train.iloc[train_idx]

            X_val_cv = X_train.iloc[val_idx]
            T_val_cv = T_train.iloc[val_idx]
            E_val_cv = E_train.iloc[val_idx]

            # Create and fit model
            model = XGBSEDebiasedBCE(**param_dict)
            model.fit(X_train_cv, T_train_cv, E_train_cv)

            # Predict on validation set
            preds = model.predict(X_val_cv)

            # Calculate concordance index
            c_index = concordance_index(T_val_cv, -preds, E_val_cv)
            scores.append(c_index)

        # Calculate mean score
        mean_score = np.mean(scores)
        print(f"Params: {param_dict}, Mean C-index: {mean_score:.4f}")

        # Update best parameters if better
        if mean_score > best_score:
            best_score = mean_score
            best_params = param_dict

    print(f"Best parameters: {best_params}, C-index: {best_score:.4f}")

    # Create and fit model with best parameters
    best_model = XGBSEDebiasedBCE(**best_params)
    best_model.fit(X_train, T_train, E_train)

    return best_params, best_model
