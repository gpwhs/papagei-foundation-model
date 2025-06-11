import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from tabpfn import TabPFNClassifier
import time
import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. Generate Synthetic Data ---
print("Generating synthetic data...")
X, y = np.random.randn(200000, 20), np.random.randint(0, 2, 200000)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)
print(f"Data ready: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print("-" * 30)

# --- 2. Partition the Data with a Decision Tree ---
print("Step 1: Partitioning data with a Decision Tree...")
partitioner = DecisionTreeClassifier(
    max_leaf_nodes=100,
    min_samples_leaf=1000,
    random_state=42,
)
partitioner.fit(X_train, y_train)

train_leaf_indices = partitioner.apply(X_train)
unique_leaves = np.unique(train_leaf_indices)
print(f"Data partitioned into {len(unique_leaves)} leaves/subsets.")
print("-" * 30)

# --- 3. Train a TabPFN Model for Each Leaf ---
tabpfn_models = {}
print("Step 2: Training a TabPFN model for each data partition...")
start_time = time.time()

for leaf_index in tqdm(unique_leaves):
    X_subset = X_train[train_leaf_indices == leaf_index]
    y_subset = y_train[train_leaf_indices == leaf_index]

    if len(y_subset) < 10:
        print(f"  Skipping leaf {leaf_index}, too few samples: {len(y_subset)}")
        continue

    clf = TabPFNClassifier(
        device=DEVICE, n_estimators=32, ignore_pretraining_limits=True
    )
    clf.fit(X_subset, y_subset)
    tabpfn_models[leaf_index] = clf

end_time = time.time()
print(
    f"Finished training {len(tabpfn_models)} models in {end_time - start_time:.2f} seconds."
)
print("-" * 30)

# --- 4. Make Predictions on the Test Set & Log AUC per Leaf ---
print("Step 3: Making predictions and logging AUC per leaf...")
test_leaf_indices = partitioner.apply(X_test)
predictions = np.zeros(len(X_test), dtype=int)

for leaf_index in np.unique(test_leaf_indices):
    idx = np.where(test_leaf_indices == leaf_index)[0]
    if leaf_index in tabpfn_models:
        preds = tabpfn_models[leaf_index].predict(X_test[idx])
    else:
        preds = partitioner.predict(X_test[idx])

    predictions[idx] = preds

    # Compute and log AUC for this leaf
    y_true_leaf = y_test[idx]
    try:
        auc = roc_auc_score(y_true_leaf, preds)
        print(f"Leaf {leaf_index}: {len(idx)} samples → AUC = {auc:.4f}")
    except ValueError:
        # Happens if only one class present in y_true_leaf
        print(
            f"Leaf {leaf_index}: {len(idx)} samples → AUC cannot be computed (only one class)"
        )

print("All leaf AUCs logged.")
print("-" * 30)

# --- 5. Evaluate Overall Performance ---
accuracy = accuracy_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)
print(f"Final Accuracy on test set: {accuracy:.4f}")
print(f"ROC AUC on test set: {roc_auc:.4f}")
