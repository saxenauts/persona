"""
Download PersonaMem 32k dataset from HuggingFace

This script downloads the PersonaMem dataset (32k variant) from HuggingFace
and saves it locally for evaluation.
"""

import json
import os
import shutil
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def download_personamem_32k(output_dir: str = "evals/data/personamem"):
    """
    Download PersonaMem 32k variant from HuggingFace

    Args:
        output_dir: Directory to save the downloaded data
    """
    print("Downloading PersonaMem 32k dataset from HuggingFace...")

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load the dataset from HuggingFace
        # The PersonaMem dataset is at bowen-upenn/PersonaMem
        # First, let's check what configs are available
        dataset = load_dataset("bowen-upenn/PersonaMem", "benchmark")

        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")

        # Save the questions as CSV
        for split_name, split_data in dataset.items():
            output_file = output_path / f"questions_{split_name}_32k.csv"
            split_data.to_csv(str(output_file), index=False)
            print(f"Saved {split_name} split to {output_file}")
            print(f"  - {len(split_data)} questions")

        # Also save as JSON for easier programmatic access
        for split_name, split_data in dataset.items():
            output_file = output_path / f"questions_{split_name}_32k.json"
            with open(output_file, 'w') as f:
                json.dump(split_data.to_list(), f, indent=2)
            print(f"Saved {split_name} split to {output_file} (JSON format)")

        # Save metadata
        metadata = {
            "dataset_name": "PersonaMem",
            "variant": "32k",
            "source": "bowen-upenn/PersonaMem",
            "splits": {
                split_name: {
                    "num_questions": len(split_data),
                    "columns": list(split_data.column_names)
                }
                for split_name, split_data in dataset.items()
            }
        }

        metadata_file = output_path / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\nMetadata saved to {metadata_file}")

        # Download shared contexts file
        contexts_filename = "shared_contexts_32k.jsonl"
        contexts_path = hf_hub_download(
            repo_id="bowen-upenn/PersonaMem",
            filename=contexts_filename,
            repo_type="dataset"
        )
        dest_path = output_path / contexts_filename
        shutil.copy(contexts_path, dest_path)
        print(f"Saved shared contexts to {dest_path}")

        print("\n✓ PersonaMem 32k dataset downloaded successfully!")

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nIf the dataset is not publicly available, you may need to:")
        print("1. Request access on HuggingFace")
        print("2. Authenticate with `huggingface-cli login`")
        raise


if __name__ == "__main__":
    download_personamem_32k()
