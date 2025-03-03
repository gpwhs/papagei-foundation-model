from linearprobing.feature_extraction_papagei import extract_features_and_save
from linearprobing.utils import load_model_without_module_prefix
from linearprobing.feature_extraction_papagei import e
from models.resnet import ResNet1DMoE

# Then define your model, load checkpoint, and call extract_features_and_save
if __name__ == "__main__":
    model_path = "weights/papagei_s.pt"
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

    model = load_model_without_module_prefix(model, model_path)
    device = "cpu"  # or "cuda" if available
    model.to(device)

    extract_features_and_save(
        model=model,
        ppg_dir="some_ppg_folder",
        batch_size=256,
        device=device,
        output_idx=0,
        resample=False,
        normalize=False,
        fs=125,
        fs_target=125,
        content="patient",
    )
