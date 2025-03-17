from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str = "borderline",
    sampling_strategy: float = 0.5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE oversampling to handle class imbalance.

    Args:
        X_train: Training features DataFrame
        y_train: Training labels Series
        strategy: SMOTE strategy ('regular', 'borderline', 'svm', 'adasyn')
        sampling_strategy: Target ratio of minority to majority class
                           (0.5 = 1:2 ratio, 1.0 = 1:1 ratio)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (resampled_X, resampled_y)
    """
    print(f"Applying {strategy} SMOTE with sampling strategy {sampling_strategy}")
    print(
        f"Original class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}"
    )

    # Select the appropriate oversampling algorithm
    if strategy == "regular":
        oversampler = SMOTE(
            sampling_strategy=sampling_strategy, random_state=random_state
        )
    elif strategy == "borderline":
        oversampler = BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=5,  # Default is 5, adjust based on dataset size
            m_neighbors=10,  # For borderline-2 SMOTE
            kind="borderline-2",  # More informative than borderline-1
        )
    elif strategy == "adasyn":
        oversampler = ADASYN(
            sampling_strategy=sampling_strategy, random_state=random_state
        )
    else:
        raise ValueError(f"Unknown SMOTE strategy: {strategy}")

    # Apply oversampling
    X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)

    # Convert back to DataFrame/Series to preserve column names
    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled, name=y_train.name)

    print(
        f"New class distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}"
    )
    return X_resampled, y_resampled


def apply_hybrid_sampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str = "smote_enn",
    sampling_strategy: float = 0.5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply hybrid sampling methods that combine oversampling and undersampling.

    Args:
        X_train: Training features DataFrame
        y_train: Training labels Series
        strategy: Sampling strategy ('smote_tomek' or 'smote_enn')
        sampling_strategy: Target ratio of minority to majority class
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (resampled_X, resampled_y)
    """
    print(f"Applying {strategy} with sampling strategy {sampling_strategy}")
    print(
        f"Original class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}"
    )

    if strategy == "smote_tomek":
        sampler = SMOTETomek(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            smote=BorderlineSMOTE(random_state=random_state),
        )
    elif strategy == "smote_enn":
        sampler = SMOTEENN(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            smote=BorderlineSMOTE(random_state=random_state),
        )
    else:
        raise ValueError(f"Unknown hybrid sampling strategy: {strategy}")

    # Apply hybrid sampling
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    # Convert back to DataFrame/Series to preserve column names
    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled, name=y_train.name)

    print(
        f"New class distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}"
    )
    return X_resampled, y_resampled


def stratified_split_with_sampling(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    handle_imbalance: bool = False,
    sampling_method: str = "borderline_smote",
    sampling_strategy: float = 0.5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified train-test split and apply sampling techniques to the training set.

    Args:
        features: Feature DataFrame
        target: Target Series
        test_size: Proportion of data to use for testing
        sampling_method: Method for handling imbalance
            ('none', 'borderline_smote', 'regular_smote', 'adasyn',
             'smote_tomek', 'smote_enn')
        sampling_strategy: Target ratio for sampling
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Perform stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    # Apply sampling only if a method is specified
    if sampling_method == "none" or not handle_imbalance:
        print("No sampling applied")
        return X_train, X_test, y_train, y_test

    # Apply the appropriate sampling method
    if sampling_method == "borderline_smote":
        X_train, y_train = apply_smote(
            X_train,
            y_train,
            strategy="borderline",
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )
    elif sampling_method == "regular_smote":
        X_train, y_train = apply_smote(
            X_train,
            y_train,
            strategy="regular",
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )
    elif sampling_method == "adasyn":
        X_train, y_train = apply_smote(
            X_train,
            y_train,
            strategy="adasyn",
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )
    elif sampling_method == "smote_tomek":
        X_train, y_train = apply_hybrid_sampling(
            X_train,
            y_train,
            strategy="smote_tomek",
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )
    elif sampling_method == "smote_enn":
        X_train, y_train = apply_hybrid_sampling(
            X_train,
            y_train,
            strategy="smote_enn",
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    return X_train, X_test, y_train, y_test
