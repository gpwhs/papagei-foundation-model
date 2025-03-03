import torch
from models.resnet import ResNet1DMoE
from linearprobing.feature_extraction_papagei import save_embeddings_df
from linearprobing.utils import load_model_without_module_prefix
from linearprobing.classification import classification_model
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os

# 1. Set up the model
model_config = {
    "base_filters": 32,
    "kernel_size": 3,
    "stride": 2,
    "groups": 1,
    "n_block": 18,
    "n_classes": 512,
    "n_experts": 3,
}

model = ResNet1DMoE(
    in_channels=1,
    base_filters=model_config["base_filters"],
    kernel_size=model_config["kernel_size"],
    stride=model_config["stride"],
    groups=model_config["groups"],
    n_block=model_config["n_block"],
    n_classes=model_config["n_classes"],
    n_experts=model_config["n_experts"],
)
# 2. Load pretrained weights
model_path = "weights/papagei_s.pt"  # You'll need to specify this
model = load_model_without_module_prefix(model, model_path)
model.dense = torch.nn.Linear(512, 2)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

# 3. Extract features for each split
data_dir = "data/ppg-bp"
save_dir = os.path.join(data_dir, "features/papagei")
os.makedirs(save_dir, exist_ok=True)

# for split in ['train', 'val', 'test']:
#     # Load split metadata
#     df = pd.read_csv(f"{data_dir}/{split}.csv")
#
#     # Extract features
#     save_embeddings_df(
#         path=os.path.join(data_dir, "ppg"),
#         df=df,
#         case_name='case_id',
#         child_dirs=df['case_id'].unique(),
#         save_dir=os.path.join(save_dir, split),
#         model=model,
#         batch_size=256,
#         device=device,
#         output_idx=1
#     )

# 4. Train and evaluate classifier

# Load the processed features
X_train, y_train, X_test, y_test, train_keys, _, test_keys = (
    load_linear_probe_dataset_objs(
        dataset_name="ppg-bp",
        model_name="papagei",
        label="hypertension",
        func="get_data_for_ml",
        content="_patient",
        level="patient",
        string_convert=True,
        concat=True,
    )
)

# Set up classifier
estimator = LogisticRegression()
param_grid = {
    "penalty": ["l1", "l2"],
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear"],
    "max_iter": [200],
}

# Train and evaluate
results = classification_model(
    estimator=estimator,
    param_grid=param_grid,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    bootstrap=True,
)

# Print results
print("\nResults:")
print(f"Best parameters: {results['parameters']}")
print(
    f"AUC: {results['auc']:.3f} ({results['auc_lower_ci']:.3f}-{results['auc_upper_ci']:.3f})"
)
print(
    f"F1: {results['f1']:.3f} ({results['f1_lower_ci']:.3f}-{results['f1_upper_ci']:.3f})"
)

