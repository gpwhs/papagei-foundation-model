"""
Test script for biobank_survival.py

This script can be used to test the survival analysis implementation
with a small synthetic dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.datasets import load_rossi


# Helper function to generate synthetic survival data
def generate_synthetic_survival_data(n_samples=1000, n_features=10, random_state=42):
    """Generate synthetic survival data."""
    # Generate features and binary outcome
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        random_state=random_state,
    )

    # Convert to DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)

    # Add age, sex, BMI as traditional factors
    X_df["age"] = np.random.normal(50, 10, n_samples)
    X_df["sex"] = np.random.binomial(1, 0.5, n_samples)
    X_df["BMI"] = np.random.normal(25, 5, n_samples)

    # Generate time-to-event (higher risk score -> shorter time)
    risk_score = 0.3 * X[:, 0] + 0.2 * X[:, 1] - 0.15 * X[:, 2] + 0.1 * X_df["age"] / 50
    baseline_time = np.random.exponential(scale=1000, size=n_samples)
    time = baseline_time * np.exp(-risk_score)

    # Generate censoring times and determine observed events
    censoring_time = np.random.exponential(scale=1000, size=n_samples)
    observed_time = np.minimum(time, censoring_time)
    event = (time <= censoring_time).astype(int)

    # Create PPG features (simulated)
    pyppg_features = []
    for i in range(100):
        feature_name = f"pyppg_{i}"
        if i < 5:  # Make first 5 informative
            X_df[feature_name] = risk_score + np.random.normal(0, 0.5, n_samples)
        else:  # Rest are random
            X_df[feature_name] = np.random.normal(0, 1, n_samples)
        pyppg_features.append(feature_name)

    # Add outcome variables
    X_df["time"] = observed_time
    X_df["event"] = event

    return X_df, pyppg_features


# Test with real data from lifelines
def test_with_rossi_data():
    """Test survival analysis with the Rossi recidivism dataset."""
    print("Testing with Rossi recidivism dataset...")

    # Load dataset
    rossi = load_rossi()
    print(rossi.head())

    # Basic survival analysis with Cox PH
    cph = CoxPHFitter()
    cph.fit(rossi, duration_col="week", event_col="arrest")

    # Print summary
    print(cph.summary)

    # Plot survival function
    plt.figure(figsize=(10, 6))
    cph.plot_partial_effects_on_outcome("prio", values=[0, 5, 10])
    plt.title("Effect of number of prior arrests on survival")
    plt.tight_layout()
    plt.savefig("rossi_survival_curves.png")
    plt.close()

    print("Rossi dataset test completed.")


# Test with synthetic data
def test_with_synthetic_data():
    """Test survival analysis with synthetic data."""
    print("Testing with synthetic data...")

    # Generate data
    df, pyppg_features = generate_synthetic_survival_data(n_samples=500)

    # Print summary statistics
    print(f"Dataset shape: {df.shape}")
    print(f"Event rate: {df['event'].mean():.2f}")
    print(f"Median follow-up time: {df['time'].median():.2f}")

    # Plot Kaplan-Meier curve
    kmf = KaplanMeierFitter()
    kmf.fit(df["time"], df["event"], label="Overall")

    plt.figure(figsize=(10, 6))
    kmf.plot_survival_function()
    plt.title("Kaplan-Meier Survival Curve")
    plt.tight_layout()
    plt.savefig("synthetic_km_curve.png")
    plt.close()

    # Split by a feature (e.g., sex)
    plt.figure(figsize=(10, 6))
    for sex_val in [0, 1]:
        mask = df["sex"] == sex_val
        kmf.fit(df.loc[mask, "time"], df.loc[mask, "event"], label=f"Sex = {sex_val}")
        kmf.plot_survival_function()

    plt.title("Kaplan-Meier Curves by Sex")
    plt.tight_layout()
    plt.savefig("synthetic_km_by_sex.png")
    plt.close()

    # Fit Cox PH model
    cph = CoxPHFitter()
    features_to_use = ["age", "sex", "BMI", "feature_0", "feature_1", "feature_2"]
    cph.fit(
        df[features_to_use + ["time", "event"]], duration_col="time", event_col="event"
    )

    # Print summary
    print(cph.summary)

    print("Synthetic data test completed.")


# Main function
def main():
    """Run tests."""
    print("Running survival analysis tests...")

    # Test with Rossi data
    test_with_rossi_data()

    # Test with synthetic data
    test_with_synthetic_data()

    print("All tests completed.")


if __name__ == "__main__":
    main()
