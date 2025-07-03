import pycurl
import json
import os
from pathlib import Path
from tqdm import tqdm
from .config import DATA_DIR

# Dataset URLs are defined here as they are only used for fetching
DATASET_URLS = {
    "oracle": "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_oracle",
    "s": "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s", 
    "m": "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_m"
}

def ensure_data_dir():
    """Create data directory if it doesn't exist"""
    data_path = Path(DATA_DIR)
    data_path.mkdir(parents=True, exist_ok=True)
    return data_path

def download_dataset(dataset_type="oracle"):
    """
    Ensures the full LongMemEval dataset is available locally, downloading it
    from HuggingFace if necessary using the robust pycurl library.
    """
    if dataset_type not in DATASET_URLS:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Must be one of {list(DATASET_URLS.keys())}")
    
    data_path = ensure_data_dir()
    url = DATASET_URLS[dataset_type]
    filename = f"longmemeval_{dataset_type}.json"
    file_path = data_path / filename
    
    if file_path.exists() and os.path.getsize(file_path) > 0:
        print(f"✅ Full dataset already exists locally: {file_path}")
        return file_path
    
    print(f"Downloading full '{dataset_type}' dataset from HuggingFace...")
    print(f"URL: {url}")

    try:
        with open(file_path, 'wb') as f, tqdm(
            unit='B', unit_scale=True, desc=f"📥 Downloading {dataset_type}"
        ) as pbar:
            c = pycurl.Curl()
            c.setopt(c.URL, url)
            c.setopt(c.WRITEDATA, f)
            c.setopt(c.FOLLOWLOCATION, True) # Follow redirects
            c.setopt(c.NOPROGRESS, False)
            
            # Custom progress function for tqdm
            def progress(download_t, download_d, upload_t, upload_d):
                pbar.total = download_t
                pbar.n = download_d
                pbar.refresh()

            c.setopt(c.XFERINFOFUNCTION, progress)
            
            c.perform()
            c.close()

        print(f"✅ Download complete. Dataset saved to: {file_path}")
        return file_path
        
    except pycurl.error as e:
        print(f"❌ A pycurl error occurred: {e}")
        if file_path.exists():
            os.remove(file_path)
        raise
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        if file_path.exists():
            os.remove(file_path)
        raise

def load_dataset(file_path):
    """Load dataset from file"""
    with open(file_path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download LongMemEval dataset")
    parser.add_argument("--dataset", choices=["oracle", "s", "m"], default="oracle", 
                       help="Dataset type to download")
    
    args = parser.parse_args()
    
    file_path = download_dataset(args.dataset)
    print(f"Dataset is ready at: {file_path}") 