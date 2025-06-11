import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tabpfn import TabPFNClassifier
import torch

# Detect device for TabPFN
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. Generate Synthetic Data (Replace with your data loading) ---
print("Generating synthetic data...")
X, y = np.random.randn(10000, 20), np.random.randint(0, 2, 10000)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)
print(f"Data ready: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print("-" * 30)

# --- 2. Train a Single TabPFN Model ---
print("Training TabPFN model on the full training set...")
tabpfn = TabPFNClassifier(
    device=DEVICE, n_estimators=32, ignore_pretraining_limits=True
)
tabpfn.fit(X_train, y_train)
print("Training complete.")
print("-" * 30)

# --- 3. Make Predictions on the Test Set ---
print("Making predictions on the test set...")
predictions = tabpfn.predict(X_test)
print("Predictions complete.")
print("-" * 30)

# --- 4. Evaluate Performance ---
accuracy = accuracy_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)

print(f"Final Accuracy on test set: {accuracy:.4f}")
print(f"ROC AUC on test set: {roc_auc:.4f}")
