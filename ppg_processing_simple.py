import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from sklearn.model_selection import train_test_split
import joblib
import os

from tqdm import tqdm


def parse_ppg_signal(x):
    """
    Convert the 'ppg' column entry into a usable Python object.
    Handle:
      - None
      - float('nan') or np.nan
      - string that might represent a list (use ast.literal_eval)
      - already a list or numpy array
    """
    # If the entire entry is None:
    if x is None:
        return None

    # If it's a floating NaN (e.g. from a numeric column):
    if isinstance(x, float) and np.isnan(x):
        return None

    # If it's already an array or list, just return it:
    if isinstance(x, (list, np.ndarray)):
        return x

    # If it's a string, we try to parse it:
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (SyntaxError, ValueError):
            return None

    # Otherwise, we don't know how to handle it
    return None


def analyze_signals(df):
    """
    1) Convert the 'ppg' column entries with parse_ppg_signal
    2) Filter out rows that remain None
    3) Print stats on distribution of lengths
    """
    # Convert each row's "ppg" column to a python array if possible
    signals = df["ppg"].apply(parse_ppg_signal)

    # Filter out None values
    valid_signals = signals.dropna()

    # Summaries
    print(f"Total rows in DataFrame:         {len(signals)}")
    print(f"Rows with a valid PPG signal:    {len(valid_signals)}")
    print(f"Rows with invalid/None signals:  {len(signals) - len(valid_signals)}")

    # For each valid row, store length
    lengths = valid_signals.apply(lambda arr: len(arr) if arr is not None else 0)

    print("\nSignal Length Statistics (samples at original sampling):")
    print(lengths.describe())

    # Quick histogram
    plt.figure(figsize=(8, 4))
    lengths.hist(bins=50)
    plt.title("Distribution of (PPG) Signal Lengths")
    plt.xlabel("Number of Samples in PPG")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    return lengths, valid_signals


def process_and_pad_signal(ppg, fs_original=250, fs_target=125, target_length=1250):
    """
    Example pipeline:
      - Resample from fs_original to fs_target
      - Z-score normalize
      - Zero-pad or clip to fixed length
    """
    # Convert to numpy array
    ppg = np.array(ppg, dtype=float).squeeze()

    # Example: resample from 250 -> 125
    gcd = np.gcd(fs_original, fs_target)
    up = fs_target // gcd
    down = fs_original // gcd
    if len(ppg) == 0:
        return None

    # Resample
    from scipy.signal import resample_poly

    ppg_resampled = resample_poly(ppg, up, down)

    # Z-score
    mean_val = ppg_resampled.mean()
    std_val = ppg_resampled.std() + 1e-10
    ppg_normalized = (ppg_resampled - mean_val) / std_val

    # Pad or clip
    current_length = len(ppg_normalized)
    if current_length < target_length:
        # pad equally on both sides
        pad_left = (target_length - current_length) // 2
        pad_right = target_length - current_length - pad_left
        ppg_final = np.pad(
            ppg_normalized, (pad_left, pad_right), mode="constant", constant_values=0
        )
    else:
        # if too long, clip center
        start = (current_length - target_length) // 2
        ppg_final = ppg_normalized[start : start + target_length]

    # Return final
    return ppg_final


def visualize_processing_steps(
    signal, fs_original=250, fs_target=125, target_length=1250
):
    """
    Show your pipeline for a single example
    """
    from copy import deepcopy

    signal_copy = deepcopy(signal)

    # raw
    ppg_raw = np.array(signal_copy, dtype=float).squeeze()
    # resample
    gcd = np.gcd(fs_original, fs_target)
    up = fs_target // gcd
    down = fs_original // gcd
    from scipy.signal import resample_poly

    ppg_resampled = resample_poly(ppg_raw, up, down)

    # z-score
    ppg_norm = (ppg_resampled - ppg_resampled.mean()) / (ppg_resampled.std() + 1e-10)

    # pad
    if len(ppg_norm) < target_length:
        pad_left = (target_length - len(ppg_norm)) // 2
        pad_right = target_length - len(ppg_norm) - pad_left
        ppg_padded = np.pad(
            ppg_norm, (pad_left, pad_right), mode="constant", constant_values=0
        )
    else:
        start = (len(ppg_norm) - target_length) // 2
        ppg_padded = ppg_norm[start : start + target_length]

    # Plots
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(10, 8))
    axes[0].plot(ppg_raw)
    axes[0].set_title("Original (250 Hz)")

    axes[1].plot(ppg_resampled)
    axes[1].set_title(f"Resampled -> {fs_target} Hz")

    axes[2].plot(ppg_norm)
    axes[2].set_title("Z-score Normalized")

    axes[3].plot(ppg_padded)
    axes[3].set_title(f"Padded or Clipped -> {target_length} samples")

    plt.tight_layout()
    plt.show()


def preprocess_ppg_data(
    df,
    eid_col="eid",
    ppg_col="ppg",
    label_col="hypertension",
    fs_original=250,
    fs_target=125,
    target_length=1250,
    out_dir="data/ppg-bp",
):
    """
    1) Convert 'ppg' with parse_ppg_signal
    2) Drop invalid
    3) Split
    4) Save signals to .p files
    5) Save metadata .csv
    """

    # Convert "ppg" column
    df = df.copy()
    df[ppg_col] = df[ppg_col].apply(parse_ppg_signal)

    # Filter out None
    df = df.dropna(subset=[ppg_col])

    # train-val-test split
    subjects = df[eid_col].unique()
    train_subj, test_subj = train_test_split(subjects, test_size=0.2, random_state=42)
    train_subj, val_subj = train_test_split(train_subj, test_size=0.2, random_state=42)

    splits = {
        "train": df[df[eid_col].isin(train_subj)].copy(),
        "val": df[df[eid_col].isin(val_subj)].copy(),
        "test": df[df[eid_col].isin(test_subj)].copy(),
    }

    # Make output directory
    os.makedirs(out_dir, exist_ok=True)
    ppg_save_dir = os.path.join(out_dir, "ppg")
    os.makedirs(ppg_save_dir, exist_ok=True)

    # Process each split
    for split_name, split_df in splits.items():
        print(f"\nProcessing {split_name} set with {len(split_df)} rows ...")

        # We'll track rows in a small list
        meta_list = []
        for idx, row in tqdm(split_df.iterrows()):
            # Example user ID as "0001", "0002", ...
            user_id_str = str(row[eid_col]).zfill(4)
            # create sub-directory for this user
            user_dir = os.path.join(ppg_save_dir, user_id_str)
            os.makedirs(user_dir, exist_ok=True)

            # process + pad
            final_ppg = process_and_pad_signal(
                row[ppg_col],
                fs_original=fs_original,
                fs_target=fs_target,
                target_length=target_length,
            )
            # if it is None (e.g. empty), skip
            if final_ppg is None or len(final_ppg) == 0:
                continue

            # We'll name the segment by the 'idx'
            seg_name = f"{idx}.p"
            # Save it
            joblib.dump(final_ppg, os.path.join(user_dir, seg_name))

            # Record metadata
            meta_list.append(
                {
                    "case_id": user_id_str,
                    "segments": seg_name,
                    "hypertension": row[label_col],
                    "fs": fs_target,
                }
            )

        # Convert to DataFrame
        meta_df = pd.DataFrame(meta_list)
        meta_csv_path = os.path.join(out_dir, f"{split_name}.csv")
        meta_df.to_csv(meta_csv_path, index=False)
        print(f"Saved {split_name} metadata -> {meta_csv_path}")

    return splits
