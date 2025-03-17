import os
import time
from typing import Tuple, Optional, List, Dict
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    f1_score,
    classification_report,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    fbeta_score,
    make_scorer,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from linearprobing.utils import bootstrap_metric_confidence_interval
from biobank_experiment_utils import (
    setup_model,
    ModelTypes,
    ClassificationResults,
    ExperimentConfig,
)

from biobank_reporting_utils import (
    save_results_to_file,
    plot_roc_curve,
    plot_pr_curve,
    save_feature_importance,
    explain_model_predictions,
    plot_odds_ratios,
)
from biobank_experiment_constants import get_pyppg_features


def train_and_evaluate_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_type: str,
    model_name: str,
    outcome: str,
    output_dir: str,
    collect_predictions: bool = True,
    handle_imbalance: bool = False,
) -> Tuple[ClassificationResults, Optional[np.ndarray]]:
    """Train a model and evaluate it.

    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        model_type: Type of model (e.g., "LR" for logistic regression)
        model_name: Name of the model for saving outputs
        outcome: Name of the outcome variable
        output_dir: Directory to save results
        collect_predictions: Whether to return prediction probabilities for calibration analysis
        handle_imbalance: Whether to apply class weighting for imbalanced datasets

    Returns:
        Tuple of (ClassificationResults, predicted_probabilities)
    """
    from sklearn.metrics import precision_recall_curve, auc as sklearn_auc
    from sklearn.metrics import average_precision_score

    # Ensure output directory exists
    model_output_dir = f"{output_dir}/{model_type}"
    os.makedirs(model_output_dir, exist_ok=True)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Set up model and hyperparameter search space
    model, parameter_distributions = setup_model(ModelTypes(model_type))

    # Handle imbalanced data if requested
    if handle_imbalance:
        # For XGBoost, we modify parameter_distributions to include scale_pos_weight
        if model_type == ModelTypes.XGBOOST.value:
            # Calculate the scale_pos_weight based on class distribution
            negative_samples = sum(y_train == 0)
            positive_samples = sum(y_train == 1)
            scale_pos_weight = negative_samples / positive_samples

            # Check if parameter_distributions is a list or dictionary
            if isinstance(parameter_distributions, list):
                # Handle list of dictionaries (like for LogisticRegression)
                for param_dict in parameter_distributions:
                    if "scale_pos_weight" not in param_dict:
                        param_dict["scale_pos_weight"] = [1, 10, 20, scale_pos_weight]
            else:
                # Handle dictionary
                if "scale_pos_weight" not in parameter_distributions:
                    parameter_distributions["scale_pos_weight"] = [
                        1,
                        10,
                        20,
                        scale_pos_weight,
                    ]

            print(f"Set scale_pos_weight options: [1, 10, 20, {scale_pos_weight:.2f}]")

        # For Logistic Regression, we always include class_weight parameter
        elif model_type == ModelTypes.LOGISTIC_REGRESSION.value:
            # For LR, class_weight is already included in parameter_distributions in setup_model
            print("Using class_weight parameter options for Logistic Regression")

    # RandomizedSearchCV for parameter tuning

    if handle_imbalance:
        f2_scorer = make_scorer(fbeta_score, beta=2)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(
            model,
            param_distributions=parameter_distributions,
            n_iter=200,
            cv=cv,
            scoring=f2_scorer,
            return_train_score=True,
            n_jobs=-1,
            verbose=3,
            random_state=42,
        )
    else:
        random_search = RandomizedSearchCV(
            model,
            param_distributions=parameter_distributions,
            n_iter=60,
            cv=3,
            scoring="roc_auc",
            return_train_score=True,
            n_jobs=-1,
            verbose=3,
            random_state=42,
        )

    # Train the model
    start_time = time.time()
    random_search.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time

    best_model = random_search.best_estimator_
    print(f"{model_name} - Best parameters: {random_search.best_params_}")

    # Fit the best model (necessary? it should already be fitted)
    best_model.fit(X_train_scaled, y_train)
    # calibrate using isotonic regression
    calibrated_best_model = CalibratedClassifierCV(
        best_model, method="isotonic", cv="prefit"
    )
    calibrated_best_model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = calibrated_best_model.predict(X_test_scaled)
    y_pred_proba = calibrated_best_model.predict_proba(X_test_scaled)[:, 1]
    if handle_imbalance:
        thresholds = np.arange(0, 1, 0.01)
        f2_scores = [
            fbeta_score(y_test, (y_pred_proba >= t).astype(int), beta=2)
            for t in thresholds
        ]
        best_threshold = thresholds[np.argmax(f2_scores)]
        best_score = max(f2_scores)
        print(
            f"Optimal threshold based on F2 score: {best_threshold:.2f} (F2: {best_score:.4f})"
        )
        y_pred_custom = (y_pred_proba >= best_threshold).astype(int)
        y_pred = y_pred_custom

    else:
        thresholds = np.arange(0, 1, 0.01)

        f1_scores = [
            f1_score(y_test, (y_pred_proba >= t).astype(int)) for t in thresholds
        ]
        best_threshold = thresholds[np.argmax(f1_scores)]
        print(f"Optimal threshold based on F1 score: {best_threshold:.2f}")

        # Use the optimal threshold to generate final predictions
        y_pred_custom = (y_pred_proba >= best_threshold).astype(int)
        y_pred = y_pred_custom

    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)

    # Calculate AUCPR (Area Under Precision-Recall Curve)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    aucpr = sklearn_auc(recall, precision)
    # Alternative calculation using average_precision_score
    avg_precision = average_precision_score(y_test, y_pred_proba)

    # Print which metric was used
    print(f"AUCPR: {aucpr:.4f} (calculated from precision-recall curve)")
    print(f"Average Precision: {avg_precision:.4f} (alternative calculation)")

    # Calculate confidence intervals
    roc_auc_ci_lower, roc_auc_ci_upper, _ = bootstrap_metric_confidence_interval(
        y_test, y_pred_proba, roc_auc_score
    )
    accuracy_ci_lower, accuracy_ci_upper, _ = bootstrap_metric_confidence_interval(
        y_test, y_pred, accuracy_score
    )
    f1_ci_lower, f1_ci_upper, _ = bootstrap_metric_confidence_interval(
        y_test, y_pred, f1_score
    )
    f2_ci_lower, f2_ci_upper, _ = bootstrap_metric_confidence_interval(
        y_test, y_pred, fbeta_score, beta=2
    )

    # Calculate AUCPR confidence intervals
    def pr_auc_score(y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return sklearn_auc(recall, precision)

    aucpr_ci_lower, aucpr_ci_upper, _ = bootstrap_metric_confidence_interval(
        y_test, y_pred_proba, pr_auc_score
    )

    # Print evaluation metrics
    print(f"\n{model_name} - {outcome} Classification Results:")
    print(
        f"Accuracy: {accuracy:.4f} (CI: {accuracy_ci_lower:.4f}-{accuracy_ci_upper:.4f})"
    )
    print(f"ROC AUC: {roc_auc:.4f} (CI: {roc_auc_ci_lower:.4f}-{roc_auc_ci_upper:.4f})")
    print(f"PR AUC: {aucpr:.4f} (CI: {aucpr_ci_lower:.4f}-{aucpr_ci_upper:.4f})")
    print(f"F1 Score: {f1:.4f} (CI: {f1_ci_lower:.4f}-{f1_ci_upper:.4f})")
    print(f"F2 Score: {f2:.4f} (CI: {f2_ci_lower:.4f}-{f2_ci_upper:.4f})")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Save results to files
    save_results_to_file(
        model_type=model_type,
        model_name=model_name,
        outcome=outcome,
        output_dir=output_dir,
        best_params=random_search.best_params_,
        training_time=training_time,
        accuracy=accuracy,
        accuracy_ci_lower=accuracy_ci_lower,
        accuracy_ci_upper=accuracy_ci_upper,
        roc_auc=roc_auc,
        roc_auc_ci_lower=roc_auc_ci_lower,
        roc_auc_ci_upper=roc_auc_ci_upper,
        aucpr=aucpr,
        aucpr_ci_lower=aucpr_ci_lower,
        aucpr_ci_upper=aucpr_ci_upper,
        f1=f1,
        f1_ci_lower=f1_ci_lower,
        f1_ci_upper=f1_ci_upper,
        y_test=y_test,
        y_pred=y_pred,
        cm=cm,
    )

    # Save the model and scaler
    joblib.dump(best_model, f"{model_output_dir}/{model_name}_classifier.joblib")
    joblib.dump(scaler, f"{model_output_dir}/{model_name}_scaler.joblib")

    # Plot ROC curve
    plot_roc_curve(
        y_test=y_test,
        y_pred_proba=y_pred_proba,
        model_type=model_type,
        model_name=model_name,
        outcome=outcome,
        output_dir=output_dir,
    )

    # Plot Precision-Recall curve
    plot_pr_curve(
        y_test=y_test,
        y_pred_proba=y_pred_proba,
        model_type=model_type,
        model_name=model_name,
        outcome=outcome,
        output_dir=output_dir,
    )

    # Feature importance (for applicable models)
    save_feature_importance(
        model=best_model,
        feature_names=X_train.columns,
        model_type=model_type,
        model_name=model_name,
        output_dir=output_dir,
    )

    # Generate SHAP explanations
    explain_model_predictions(
        best_model,
        X_test,
        y_test,
        model_type,
        model_name,
        output_dir,
    )

    # Create results object
    results = ClassificationResults(
        model=model_type,
        parameters=random_search.best_params_,
        auc=roc_auc,
        auc_lower_ci=roc_auc_ci_lower,
        auc_upper_ci=roc_auc_ci_upper,
        aucpr=aucpr,
        aucpr_lower_ci=aucpr_ci_lower,
        aucpr_upper_ci=aucpr_ci_upper,
        f1=f1,
        f1_lower_ci=f1_ci_lower,
        f1_upper_ci=f1_ci_upper,
        f2=f2,
        f2_lower_ci=f2_ci_lower,
        f2_upper_ci=f2_ci_upper,
        accuracy=accuracy,
        accuracy_lower_ci=accuracy_ci_lower,
        accuracy_upper_ci=accuracy_ci_upper,
        training_time=training_time,
    )

    if model_type == ModelTypes.LOGISTIC_REGRESSION.value:
        # Create odds ratio forest plot for logistic regression
        plot_odds_ratios(
            X_train_scaled,
            y_train,
            feature_names=X_train.columns,
            title=f"{model_name} - Odds Ratios with 95% CI",
            plot_filename=f"{model_output_dir}/{model_name}_odds_ratios.png",
            csv_filename=f"{model_output_dir}/{model_name}_odds_ratios.csv",
        )

    if collect_predictions:
        return results, y_pred_proba
    else:
        return results, None


def setup_experiments(
    embedding_columns: List[str],
    outcome: str,
) -> Dict[str, ExperimentConfig]:
    """Set up experiment configurations.

    Args:
        embedding_columns: List of embedding column names
        traditional_features: List of traditional feature names

    Returns:
        Dictionary mapping experiment keys to configurations
    """
    print(embedding_columns)
    traditional_features: List[str] = ["age", "sex", "BMI"]
    pyppg_features = get_pyppg_features(outcome)

    return {
        "M0": ExperimentConfig(
            name="M0_PaPaGei_Only",
            description="Only PaPaGei features",
            feature_columns=embedding_columns,
        ),
        "M1": ExperimentConfig(
            name="M1_Traditional_Only",
            description="Only metadata (age, sex, BMI)",
            feature_columns=traditional_features,
        ),
        "M2": ExperimentConfig(
            name="M2_PaPaGei_Traditional",
            description="Both PaPaGei features and metadata",
            feature_columns=embedding_columns + traditional_features,
        ),
        "M3": ExperimentConfig(
            name="M3_pyPPG_Only",
            description="pyPPG features",
            feature_columns=pyppg_features,
        ),
        "M4": ExperimentConfig(
            name="M4_pyPPG_Traditional",
            description="pyPPG features and metadata",
            feature_columns=pyppg_features + traditional_features,
        ),
    }


def run_experiment(
    experiment_config: ExperimentConfig,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_type: str,
    outcome: str,
    output_dir: str,
    handle_imbalance: bool = False,
) -> Tuple[ClassificationResults, np.ndarray]:
    """Run a single experiment.

    Args:
        experiment_config: Configuration for the experiment
        X_train, X_test, y_train, y_test: Data splits
        model_type: Type of model to use
        outcome: Outcome variable name
        output_dir: Directory to save results
        handle_imbalance: Whether to apply class weighting for imbalanced datasets

    Returns:
        Tuple of (ClassificationResults, prediction_probabilities)
    """
    print(
        f"\n--- Running Experiment {experiment_config.name}: {experiment_config.description} ---"
    )

    # Select features for this experiment
    X_train_exp = X_train[experiment_config.feature_columns]
    X_test_exp = X_test[experiment_config.feature_columns]

    # Train and evaluate
    results, y_pred_proba = train_and_evaluate_model(
        X_train_exp,
        X_test_exp,
        y_train,
        y_test,
        model_type=model_type,
        model_name=experiment_config.name,
        outcome=outcome,
        output_dir=output_dir,
        collect_predictions=True,
        handle_imbalance=handle_imbalance,
    )

    return results, y_pred_proba
