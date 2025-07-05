API_BASE_URL = "http://localhost:8000/api/v1"

# Evaluation settings
EVAL_MODEL = "gpt-4o-mini"
DEFAULT_SUBSET_SIZE = 3  # For initial testing
MAX_CONCURRENCY = 50  # Process full batches at once for single progress bar
TEMPERATURE = 0

# File paths - resolved relative to project root
import os
from pathlib import Path

# Get the project root (assuming this file is in evals/longmemeval/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "evals" / "data"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
