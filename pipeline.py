import pandas as pd
from ppg_processing_simple import (
    analyze_signals,
    visualize_processing_steps,
    preprocess_ppg_data,
)


def analyze_dataset(df, num_examples=3):
    """
    Helper that calls analyze_signals and also visualizes some random examples.
    """
    # Analyze all signals
    lengths, valid_signals = analyze_signals(df)

    # Show processing for a few random ones
    print(
        f"\nShowing processing steps for up to {num_examples} random valid signals ..."
    )

    valid_indices = valid_signals.index.to_list()

    import random

    random.shuffle(valid_indices)

    shown = 0
    for idx in valid_indices:
        if shown >= num_examples:
            break
        signal_arr = valid_signals.loc[idx]
        if signal_arr is not None and len(signal_arr) > 1:
            print(f"Example {shown+1}, original length = {len(signal_arr)}")
            visualize_processing_steps(
                signal_arr, fs_original=250, fs_target=125, target_length=1250
            )
            shown += 1

    return lengths


if __name__ == "__main__":
    # Load your Parquet file
    df = pd.read_parquet("data/215k_pyppg_features_and_conditions.parquet")
    # Has columns: "eid", "ppg", "hypertension"
    for col in df.columns:
        print(col)

    # Step 1: Analyze signals
    print("Analyzing signals...")
    lengths = analyze_dataset(df, num_examples=3)

    # Step 2: If all good, do the real preprocessing
    print("\nPreprocessing signals to disk...")
    splits = preprocess_ppg_data(
        df,
        eid_col="eid",
        ppg_col="ppg",
        label_col="Hypertension",
        fs_original=250,
        fs_target=125,
        target_length=1250,
        out_dir="data/ppg-bp",
    )
    print("Finished!")
