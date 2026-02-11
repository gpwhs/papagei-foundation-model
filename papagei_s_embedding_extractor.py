"""Clean PaPaGei-S embedding extraction from a pandas DataFrame.

Usage:
    from papagei_s_embedding_extractor import extract_papagei_s_embeddings

    embeddings = extract_papagei_s_embeddings(
        df=my_df,
        signal_col="ppg_resampled",   # column with 1D PPG signals
        source_fs=125,
        target_fs=125,
        target_length=1250,
        batch_size=256,
    )

    # Option 2: return original DataFrame with embeddings appended.
    df_with_embeddings = append_papagei_s_embeddings(
        df=my_df,
        signal_col="ppg_resampled",
    )
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from models.resnet import ResNet1DMoE


def _default_model_path() -> Path:
    return Path(__file__).resolve().parent / "weights" / "papagei_s.pt"


def _load_model_without_module_prefix(
    model: torch.nn.Module, checkpoint_path: str | Path
) -> torch.nn.Module:
    checkpoint = torch.load(str(checkpoint_path), map_location=torch.device("cpu"))
    state_dict = {}
    for key, value in checkpoint.items():
        state_dict[key[7:] if key.startswith("module.") else key] = value
    model.load_state_dict(state_dict)
    return model


def _build_papagei_s_model() -> ResNet1DMoE:
    model_config = {
        "base_filters": 32,
        "kernel_size": 3,
        "stride": 2,
        "groups": 1,
        "n_block": 18,
        "n_classes": 512,
        "n_experts": 3,
    }
    return ResNet1DMoE(
        in_channels=1,
        base_filters=model_config["base_filters"],
        kernel_size=model_config["kernel_size"],
        stride=model_config["stride"],
        groups=model_config["groups"],
        n_block=model_config["n_block"],
        n_classes=model_config["n_classes"],
        n_experts=model_config["n_experts"],
    )


def _coerce_1d_signal(x: object) -> np.ndarray:
    if isinstance(x, np.ndarray):
        signal = x.astype(np.float32, copy=False)
    elif torch.is_tensor(x):
        signal = x.detach().cpu().numpy().astype(np.float32, copy=False)
    elif isinstance(x, (list, tuple)):
        signal = np.asarray(x, dtype=np.float32)
    elif isinstance(x, str):
        parsed = ast.literal_eval(x)
        signal = np.asarray(parsed, dtype=np.float32)
    else:
        raise TypeError(f"Unsupported signal type: {type(x)}")

    signal = np.ravel(signal)
    if signal.size == 0:
        raise ValueError("Empty signal")
    return signal


def _zscore(signal: np.ndarray) -> np.ndarray:
    std = float(np.std(signal))
    if std == 0.0:
        return signal - float(np.mean(signal))
    return (signal - float(np.mean(signal))) / std


def _resample_1d(signal: np.ndarray, source_fs: int, target_fs: int) -> np.ndarray:
    if source_fs == target_fs:
        return signal
    new_length = int(round(len(signal) * (target_fs / source_fs)))
    if new_length < 1:
        raise ValueError("Resampled signal length became < 1")
    return np.interp(
        np.linspace(0, len(signal) - 1, new_length),
        np.arange(len(signal)),
        signal,
    ).astype(np.float32, copy=False)


def _pad_or_center_crop(signal: np.ndarray, target_length: int) -> np.ndarray:
    length = len(signal)
    if length == target_length:
        return signal
    if length < target_length:
        padding = target_length - length
        pad_left = padding // 2
        pad_right = padding - pad_left
        return np.pad(signal, (pad_left, pad_right), mode="constant")

    center = length // 2
    start = center - (target_length // 2)
    end = start + target_length
    return signal[start:end]


def _iter_batches(array: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(array), batch_size):
        yield array[start : start + batch_size]


def prepare_signals(
    df: pd.DataFrame,
    signal_col: str,
    source_fs: int = 125,
    target_fs: int = 125,
    target_length: int = 1250,
    normalize: bool = True,
    drop_invalid: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare DataFrame signals for PaPaGei-S.

    Returns:
        processed_signals: shape [N_valid, target_length]
        valid_index: original row indices that produced valid signals
    """
    if signal_col not in df.columns:
        raise KeyError(f"Column '{signal_col}' not found in dataframe")

    processed: list[np.ndarray] = []
    valid_index: list[object] = []

    for idx, raw_signal in df[signal_col].items():
        try:
            signal = _coerce_1d_signal(raw_signal)
            if normalize:
                signal = _zscore(signal)
            signal = _resample_1d(signal, source_fs=source_fs, target_fs=target_fs)
            signal = _pad_or_center_crop(signal, target_length=target_length)
            processed.append(signal.astype(np.float32, copy=False))
            valid_index.append(idx)
        except Exception:
            if not drop_invalid:
                raise

    if not processed:
        raise ValueError("No valid signals were produced from the input dataframe")

    return np.stack(processed, axis=0), np.asarray(valid_index)


def extract_papagei_s_embeddings(
    df: pd.DataFrame,
    signal_col: str,
    model_path: str | Path | None = None,
    source_fs: int = 125,
    target_fs: int = 125,
    target_length: int = 1250,
    normalize: bool = True,
    drop_invalid: bool = True,
    batch_size: int = 256,
    device: str | torch.device | None = None,
    show_progress: bool = True,
    return_index: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Extract PaPaGei-S embeddings from a DataFrame column of PPG signals.

    Args:
        df: Input dataframe.
        signal_col: Column containing 1D PPG signals (array/list/tuple/torch tensor/stringified list).
        model_path: Path to PaPaGei-S weights. Default: ./weights/papagei_s.pt
        source_fs: Input signal sampling frequency.
        target_fs: Model input sampling frequency.
        target_length: Model input length.
        normalize: Apply z-score per signal before inference.
        drop_invalid: Skip rows that fail parsing/processing (otherwise raise).
        batch_size: Inference batch size.
        device: "cuda", "cpu", etc. Auto-selects CUDA if available when None.
        show_progress: Show tqdm progress bar during model inference.
        return_index: If True, return `(embeddings, valid_index)`.

    Returns:
        embeddings: np.ndarray of shape [N_valid, 512]
        valid_index (optional): row indices corresponding to embeddings.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    model_path = Path(model_path) if model_path is not None else _default_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    processed_signals, valid_index = prepare_signals(
        df=df,
        signal_col=signal_col,
        source_fs=source_fs,
        target_fs=target_fs,
        target_length=target_length,
        normalize=normalize,
        drop_invalid=drop_invalid,
    )

    model = _build_papagei_s_model()
    model = _load_model_without_module_prefix(model, model_path)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.to(device)
    model.eval()

    all_embeddings: list[np.ndarray] = []

    iterator = _iter_batches(processed_signals, batch_size=batch_size)
    if show_progress:
        total = int(np.ceil(len(processed_signals) / batch_size))
        iterator = tqdm(iterator, total=total, desc="Extracting PaPaGei-S embeddings")

    with torch.inference_mode():
        for batch in iterator:
            batch_tensor = torch.from_numpy(batch).unsqueeze(1).to(device)
            outputs = model(batch_tensor)
            emb = outputs[0].detach().cpu().numpy()
            all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings)

    if return_index:
        return embeddings, valid_index
    return embeddings


def append_papagei_s_embeddings(
    df: pd.DataFrame,
    signal_col: str,
    model_path: str | Path | None = None,
    source_fs: int = 125,
    target_fs: int = 125,
    target_length: int = 1250,
    normalize: bool = True,
    drop_invalid: bool = True,
    batch_size: int = 256,
    device: str | torch.device | None = None,
    show_progress: bool = True,
    embedding_prefix: str = "emb_",
) -> pd.DataFrame:
    """Return original DataFrame with PaPaGei-S embeddings appended as columns.

    Notes:
        - Rows that fail signal parsing/processing are kept, and their embedding
          columns are left as NaN when ``drop_invalid=True``.
        - If ``drop_invalid=False``, invalid rows raise during extraction.
    """
    embeddings, valid_index = extract_papagei_s_embeddings(
        df=df,
        signal_col=signal_col,
        model_path=model_path,
        source_fs=source_fs,
        target_fs=target_fs,
        target_length=target_length,
        normalize=normalize,
        drop_invalid=drop_invalid,
        batch_size=batch_size,
        device=device,
        show_progress=show_progress,
        return_index=True,
    )

    embedding_cols = [f"{embedding_prefix}{i}" for i in range(embeddings.shape[1])]
    embedding_df = pd.DataFrame(embeddings, index=valid_index, columns=embedding_cols)

    # Join on index so embeddings line up with the original row positions.
    return df.join(embedding_df, how="left")


def _read_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(
        f"Unsupported file type: {suffix}. Use .csv, .parquet/.pq, or .pkl/.pickle."
    )


def _write_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
        return
    if suffix in {".pkl", ".pickle"}:
        df.to_pickle(path)
        return
    raise ValueError(
        f"Unsupported output type: {suffix}. Use .csv, .parquet/.pq, or .pkl/.pickle."
    )


if __name__ == "__main__":
    # Fill this in and run:
    # python papagei_s_embedding_extractor.py
    INPUT_FILE = "../biobank_rap/data/UKB_prevalentMACCE_RAP_withpyppg.parquet"

    # Optional. Leave as None to auto-detect from common names.
    SIGNAL_COL: str = "ppg_resampled"

    # Optional. If None, output is '<input_stem>_with_embeddings.parquet'
    OUTPUT_FILE: str | None = None

    if INPUT_FILE.startswith("REPLACE_WITH_"):
        raise ValueError(
            "Set INPUT_FILE at the bottom of papagei_s_embedding_extractor.py"
        )

    input_path = Path(INPUT_FILE)
    df_in = _read_dataframe(input_path)

    if OUTPUT_FILE is None:
        OUTPUT_FILE = str(
            input_path.with_name(f"{input_path.stem}_with_embeddings.parquet")
        )

    df_out = append_papagei_s_embeddings(
        df=df_in,
        signal_col=SIGNAL_COL,
    )
    _write_dataframe(df_out, OUTPUT_FILE)
    print(f"Saved DataFrame with embeddings: {OUTPUT_FILE}")
