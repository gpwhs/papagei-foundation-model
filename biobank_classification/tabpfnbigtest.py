import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.datasets import fetch_covtype, load_breast_cancer
from tabpfn import TabPFNClassifier
import time
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. Load a real dataset (Wisconsin Breast Cancer) ---
X, y = fetch_covtype(return_X_y=True)
print(f"Total samples: {X.shape[0]}, features: {X.shape[1]}")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)
print(f"Data ready: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print("-" * 30)

# --- 2. Partition the Data with a Decision Tree ---
print("Partitioning data with a Decision Tree...")
partitioner = DecisionTreeClassifier(
    max_leaf_nodes=100,
    min_samples_leaf=50,
    random_state=42,
)
partitioner.fit(X_train, y_train)
train_leaf_indices = partitioner.apply(X_train)
unique_leaves = np.unique(train_leaf_indices)
print(f"Data partitioned into {len(unique_leaves)} leaves.")
print("-" * 30)

# --- 3. Train a TabPFN Model for Each Leaf ---
tabpfn_models = {}
print("Training a TabPFN model for each leaf...")
start_time = time.time()

for leaf_index in tqdm(unique_leaves):
    mask = train_leaf_indices == leaf_index
    X_leaf, y_leaf = X_train[mask], y_train[mask]
    if len(y_leaf) < 10:
        continue
    clf = TabPFNClassifier(
        device=DEVICE, n_estimators=32, ignore_pretraining_limits=True
    )
    clf.fit(X_leaf, y_leaf)
    tabpfn_models[leaf_index] = clf

print(f"Trained {len(tabpfn_models)} models in {time.time() - start_time:.1f}s")
print("-" * 30)

# --- 4. Make Predictions & Log AUC per Leaf ---
print("Making predictions and logging AUC per leaf...")
test_leaf_indices = partitioner.apply(X_test)
predictions = np.zeros(len(X_test), dtype=int)

for leaf_index in np.unique(test_leaf_indices):
    idx = np.where(test_leaf_indices == leaf_index)[0]
    if leaf_index in tabpfn_models:
        preds = tabpfn_models[leaf_index].predict(X_test[idx])
    else:
        preds = partitioner.predict(X_test[idx])
    predictions[idx] = preds

    # Leaf-level AUC
    y_true = y_test[idx]
    try:
        auc = roc_auc_score(y_true, preds)
        print(f"Leaf {leaf_index}: {len(idx)} samples → AUC = {auc:.4f}")
    except ValueError:
        print(f"Leaf {leaf_index}: {len(idx)} samples → AUC N/A (single class)")

print("All leaf AUCs logged.")
print("-" * 30)

# --- 5. Overall Performance ---
acc = accuracy_score(y_test, predictions)
roc = roc_auc_score(y_test, predictions)
print(f"Overall accuracy: {acc:.4f}")
print(f"Overall ROC AUC: {roc:.4f}")
