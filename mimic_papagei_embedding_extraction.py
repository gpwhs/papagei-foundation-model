from biobank_experiment_utils import get_embedding_df

if __name__ == "__main__":
    # Example usage of get_embedding_df
    import os
    import pandas as pd

    # Load some example DataFrame (replace with actual data loading)
    data_path = os.getenv("BIOBANK_DATA_PATH")
    if not data_path:
        raise ValueError("BIOBANK_DATA_PATH environment variable is not set")
    df = pd.read_parquet(f"{data_path}/mimic_biomarkers_singlebeat.parquet")
    embedding_df = get_embedding_df(
        df, embeddings_file="mimic_papagei_embeddings.npy", ppg_column="ppg_template"
    )
    print(embedding_df.head())
    print([col for col in embedding_df.columns])
    embedding_df.to_parquet("mimic_papagei_embeddings.parquet")
