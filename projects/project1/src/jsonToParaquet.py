import pandas as pd
import json
import os


def stream_json_to_parquet(json_path, parquet_path, batch_size=100):
    """
    Converts a very large JSON file to Parquet format without memory overflow.

    Parameters:
    - json_path: Path to input JSON file.
    - parquet_path: Path to output Parquet file.
    - batch_size: Number of rows to process at a time (reduce for low-memory machines).
    """
    if not os.path.exists(json_path):
        print(f"[ERROR] File not found: {json_path}")
        return

    print(f"[INFO] Streaming {json_path} to {parquet_path}...")

    data_chunks = []
    with open(json_path, "r") as f:
        data = json.load(f)  # Stream JSON instead of fully loading

        for i, (key, value) in enumerate(data.items()):
            data_chunks.append({"node": key, "path": value})  # Store paths as lists

            if (i + 1) % batch_size == 0:  # Save in small chunks
                df = pd.DataFrame(data_chunks)

                # If file exists, load existing data & append
                if os.path.exists(parquet_path):
                    existing_df = pd.read_parquet(parquet_path)
                    df = pd.concat([existing_df, df])

                df.to_parquet(parquet_path, compression="snappy", index=False)
                data_chunks = []  # Clear memory

    # Save remaining data
    if data_chunks:
        df = pd.DataFrame(data_chunks)

        if os.path.exists(parquet_path):
            existing_df = pd.read_parquet(parquet_path)
            df = pd.concat([existing_df, df])

        df.to_parquet(parquet_path, compression="snappy", index=False)

    print(f"[INFO] Successfully saved: {parquet_path}")


def main():
    """
    Converts large transit_node_paths.json to Parquet format using streaming.
    """
    json_path = "preprocessing_output/transit_node_paths.json"
    parquet_path = "preprocessing_output/transit_node_paths.parquet"

    stream_json_to_parquet(json_path, parquet_path, batch_size=100)


if __name__ == "__main__":
    main()
