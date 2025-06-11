import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.tree import (
    DecisionTreeClassifier,
)  # <-- Changed from RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from tabpfn import TabPFNClassifier
import time
import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. Generate Synthetic Data (Replace with your data loading) ---
# Creating a large dataset similar to your description (215,000 samples)
print("Generating synthetic data...")
X, y = np.random.randn(10000, 20), np.random.randint(0, 2, 10000)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)
print(f"Data ready: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print("-" * 30)


# --- 2. Partition the Data with a Decision Tree ---
# *** THIS IS THE CORRECTED LINE ***
# Using a single Decision Tree ensures .apply() returns a 1D array of leaf indices.
print("Step 1: Partitioning data with a Decision Tree...")
partitioner = DecisionTreeClassifier(
    max_leaf_nodes=100,  # Controls the max number of TabPFN models to train. Adjust as needed.
    min_samples_leaf=1000,  # Each leaf will have at least 1000 samples.
    random_state=42,
)
partitioner.fit(X_train, y_train)

# Get the leaf index for each sample. This will now be a 1D array as expected.
train_leaf_indices = partitioner.apply(X_train)
unique_leaves = np.unique(train_leaf_indices)
print(f"Data partitioned into {len(unique_leaves)} leaves/subsets.")
print("-" * 30)


# --- 3. Train a TabPFN Model for Each Leaf ---
# This part of the code remains the same and will now work correctly.
tabpfn_models = {}
print("Step 2: Training a TabPFN model for each data partition...")
start_time = time.time()

for leaf_index in tqdm(unique_leaves):
    print(f"  Training model for leaf {leaf_index}...")

    # This boolean mask will now have the correct shape for indexing
    subset_mask = train_leaf_indices == leaf_index
    X_subset, y_subset = X_train[subset_mask], y_train[subset_mask]

    if len(y_subset) < 10:
        print(f"    Skipping leaf {leaf_index}, too few samples: {len(y_subset)}")
        continue

    leaf_classifier = TabPFNClassifier(
        device=DEVICE, n_estimators=32, ignore_pretraining_limits=True
    )
    leaf_classifier.fit(X_subset, y_subset)

    tabpfn_models[leaf_index] = leaf_classifier

end_time = time.time()
print(
    f"Finished training {len(tabpfn_models)} models in {end_time - start_time:.2f} seconds."
)
print("-" * 30)


# --- 4. Make Predictions on the Test Set ---
print("Step 3: Making predictions...")
test_leaf_indices = partitioner.apply(X_test)
predictions = np.zeros(len(X_test), dtype=int)

for i, leaf_index in tqdm(enumerate(test_leaf_indices)):
    if leaf_index in tabpfn_models:
        model = tabpfn_models[leaf_index]
        prediction = model.predict(X_test[i].reshape(1, -1))
        predictions[i] = prediction[0]
    else:
        # Fallback: if a test sample falls into a leaf without a trained model
        predictions[i] = partitioner.predict(X_test[i].reshape(1, -1))[0]

print("Predictions complete.")
print("-" * 30)

# --- 5. Evaluate Performance ---
accuracy = accuracy_score(y_test, predictions)
print(f"Final Accuracy on test set: {accuracy:.4f}")
roc_auc = roc_auc_score(y_test, predictions)
print(f"ROC AUC on test set: {roc_auc:.4f}")
