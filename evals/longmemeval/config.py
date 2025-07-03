API_BASE_URL = "http://localhost:8000/api/v1"

# Evaluation settings
EVAL_MODEL = "gpt-4o-mini"
DEFAULT_SUBSET_SIZE = 3  # For initial testing
MAX_CONCURRENCY = 50  # Process full batches at once for single progress bar
TEMPERATURE = 0

# File paths
DATA_DIR = "evals/data"
RESULTS_DIR = "evals/results"
