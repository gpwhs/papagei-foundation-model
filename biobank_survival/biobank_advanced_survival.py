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
                y_true_time[indices], 
                -y_pred_risk[indices], 
                y_true_event[indices]
            )
            bootstrap_c_indices.append(bootstrap_c_index)
        except:
            # Skip if there's an error (e.g., no events in the bootstrap sample)
            continue
    
    # Calculate 95% confidence interval
    lower_bound = np.percentile(bootstrap_c_indices, 2.5)
    upper_bound = np.percentile(bootstrap_c_indices, 97.5)
    
    return c_index, lower_bound, upper_bound


def train_aft_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    time_train: pd.Series,
    time_test: pd.Series,
    event_train: pd.Series,
    event_test: pd.Series,
    model_name: str,
    outcome: str,
    output_dir: str,
    aft_model_type: str = "weibull",
    penalizer: float = 0.1,
    standardize: bool = True,
) -> AdvancedSurvivalResults:
    """
    Train and evaluate an Accelerated Failure Time (AFT) model.
    
    Args:
        X_train, X_test: Feature DataFrames
        time_train, time_test: Time columns
        event_train, event_test: Event indicator columns
        model_name: Name of the model
        outcome: Name of the outcome
        output_dir: Directory to save results
        aft_model_type: Type of AFT model ('weibull' or 'lognormal')
        penalizer: Regularization strength
        standardize: Whether to standardize features
        
    Returns:
        AdvancedSurvivalResults object
    """
    import time as time_module
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Standardize features if requested
    if standardize:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Create datasets and dataloaders
        train_dataset = SurvivalDataset(X_train_scaled, time_train, event_train)
        test_dataset = SurvivalDataset(X_test_scaled, time_test, event_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create the model
        model = DeepSurv(in_features=X_train.shape[1], hidden_layers=hidden_layers)
        
        # Move model to the specified device
        device_obj = torch.device(device)
        model = model.to(device_obj)
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        start_time = time_module.time()
        
        best_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            epoch_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                x = batch['features'].to(device_obj)
                times = batch['time'].to(device_obj)
                events = batch['event'].to(device_obj)
                
                risk_scores = model(x).squeeze()
                loss = negative_log_likelihood_loss(risk_scores, times, events)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    x = batch['features'].to(device_obj)
                    times = batch['time'].to(device_obj)
                    events = batch['event'].to(device_obj)
                    
                    risk_scores = model(x).squeeze()
                    loss = negative_log_likelihood_loss(risk_scores, times, events)
                    
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(test_loader)
            val_losses.append(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Load the best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        training_time = time_module.time() - start_time
        
        # Plot training curve
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f"{output_dir}/{model_name}_deepsurv_training_curve.png")
        plt.close()
        
        # Calculate C-index on training and test set
        model.eval()
        
        # Get risk scores for training data
        train_risk_scores = []
        with torch.no_grad():
            for batch in train_loader:
                x = batch['features'].to(device_obj)
                risk_scores = model(x).squeeze().cpu().numpy()
                train_risk_scores.extend(risk_scores)
        
        train_risk_scores = np.array(train_risk_scores)
        
        # Get risk scores for test data
        test_risk_scores = []
        with torch.no_grad():
            for batch in test_loader:
                x = batch['features'].to(device_obj)
                risk_scores = model(x).squeeze().cpu().numpy()
                test_risk_scores.extend(risk_scores)
        
        test_risk_scores = np.array(test_risk_scores)
        
        # Calculate C-index
        c_index_train = concordance_index(
            time_train.values, train_risk_scores, event_train.values
        )
        c_index_test = concordance_index(
            time_test.values, test_risk_scores, event_test.values
        )
        
        # Bootstrap confidence interval for C-index
        c_index, c_index_lower, c_index_upper = bootstrap_ci_c_index(
            time_test.values, event_test.values, test_risk_scores
        )
        
        # Save the model
        torch.save(model.state_dict(), f"{output_dir}/{model_name}_deepsurv_model.pt")
        
        # Create results object
        results = AdvancedSurvivalResults(
            model="DeepSurv Neural Network",
            parameters={
                "hidden_layers": hidden_layers,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "patience": patience,
            },
            c_index=c_index,
            c_index_lower_ci=c_index_lower,
            c_index_upper_ci=c_index_upper,
            concordance_train=c_index_train,
            concordance_test=c_index_test,
            training_time=training_time,
        )
        
        return resultsindex=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        train_df = X_train_scaled.copy()
        test_df = X_test_scaled.copy()
    else:
        train_df = X_train.copy()
        test_df = X_test.copy()
    
    # Add time and event columns
    train_df['time'] = time_train
    train_df['event'] = event_train
    test_df['time'] = time_test
    test_df['event'] = event_test
    
    # Initialize the appropriate AFT model
    if aft_model_type.lower() == 'weibull':
        aft = WeibullAFTFitter(penalizer=penalizer)
        model_full_name = "Weibull AFT"
    elif aft_model_type.lower() == 'lognormal':
        aft = LogNormalAFTFitter(penalizer=penalizer)
        model_full_name = "LogNormal AFT"
    else:
        raise ValueError(f"Unknown AFT model type: {aft_model_type}")
    
    # Train the model
    start_time = time_module.time()
    aft.fit(train_df, duration_col='time', event_col='event')
    training_time = time_module.time() - start_time
    
    # Generate predictions (predicted median survival time)
    # For concordance, lower predictions should be higher risk, so we negate median survival
    train_predictions = -aft.predict_median(train_df)
    test_predictions = -aft.predict_median(test_df)
    
    # Calculate C-index
    c_index_train = concordance_index(
        time_train, train_predictions, event_train
    )
    c_index_test = concordance_index(
        time_test, test_predictions, event_test
    )
    
    # Bootstrap confidence interval for C-index
    c_index, c_index_lower, c_index_upper = bootstrap_ci_c_index(
        time_test.values, event_test.values, -test_predictions.values
    )
    
    # Plot the AFT survival curves for different feature values
    if len(X_train.columns) > 0:
        for i, feature in enumerate(X_train.columns[:3]):  # Plot first 3 features
            plt.figure(figsize=(10, 6))
            
            # Create a sample with median values
            median_values = X_test.median().to_dict()
            q1_values = X_test.quantile(0.25).to_dict()
            q3_values = X_test.quantile(0.75).to_dict()
            
            # Plot median feature values
            median_sample = pd.DataFrame([median_values])
            aft.plot_survival_function(median_sample, label='Median values')
            
            # Plot Q1 value for this feature
            q1_sample = pd.DataFrame([q1_values])
            q1_sample[feature] = X_test[feature].quantile(0.25)
            aft.plot_survival_function(q1_sample, label=f'{feature} (Q1)')
            
            # Plot Q3 value for this feature
            q3_sample = pd.DataFrame([q3_values])
            q3_sample[feature] = X_test[feature].quantile(0.75)
            aft.plot_survival_function(q3_sample, label=f'{feature} (Q3)')
            
            plt.title(f"AFT Survival Curves for {feature}")
            plt.xlabel("Time")
            plt.ylabel("Survival Probability")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(f"{output_dir}/{model_name}_{feature}_aft_curves.png")
            plt.close()
    
    # Create summary DataFrame for coefficients
    summary = aft.summary
    summary.to_csv(f"{output_dir}/{model_name}_aft_summary.csv")
    
    # Create results object
    results = AdvancedSurvivalResults(
        model=model_full_name,
        parameters={
            "aft_model_type": aft_model_type,
            "penalizer": penalizer,
            "standardize": standardize,
        },
        c_index=c_index,
        c_index_lower_ci=c_index_lower,
        c_index_upper_ci=c_index_upper,
        concordance_train=c_index_train,
        concordance_test=c_index_test,
        training_time=training_time,
    )
    
    return results


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
        raise ImportError("scikit-survival is not available. Please install it to use Random Survival Forest.")
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert data to the format required by scikit-survival
    # scikit-survival requires structured arrays with dtype [('event', bool), ('time', float)]
    y_train_sksurv = Surv.from_arrays(event_train.astype(bool), time_train)
    y_test_sksurv = Surv.from_arrays(event_test.astype(bool), time_test)
    
    # Train the Random Survival Forest
    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,  # Use all cores
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
    
    # Calculate feature importances
    feature_importances = dict(zip(X_train.columns, rsf.feature_importances_))
    
    # Plot feature importances
    plt.figure(figsize=(10, max(6, min(len(feature_importances) * 0.3, 15))))
    
    # Sort features by importance
    sorted_features = sorted(
        feature_importances.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Plot the top 20 features
    top_features = sorted_features[:20]
    feature_names = [f[0] for f in top_features]
    importance_values = [f[1] for f in top_features]
    
    # Create bar chart
    plt.barh(range(len(top_features)), importance_values, align='center')
    plt.yticks(range(len(top_features)), feature_names)
    plt.xlabel('Feature Importance')
    plt.title('Random Survival Forest Feature Importance')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_rsf_feature_importance.png")
    plt.close()
    
    # Save feature importances to CSV
    importance_df = pd.DataFrame({
        'Feature': list(feature_importances.keys()),
        'Importance': list(feature_importances.values())
    }).sort_values('Importance', ascending=False)
    
    importance_df.to_csv(f"{output_dir}/{model_name}_rsf_feature_importance.csv", index=False)
    
    # Plot survival curves for a few samples
    plt.figure(figsize=(10, 6))
    
    # Select 5 random samples from test set
    n_samples = min(5, len(X_test))
    sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        sample = X_test.iloc[idx:idx+1]
        
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
    
    return results


