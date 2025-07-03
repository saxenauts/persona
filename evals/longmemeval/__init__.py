"""
LongMemEval evaluation pipeline for Persona system
==================================================

This module provides a complete evaluation pipeline for the LongMemEval benchmark,
designed to test long-term memory capabilities of conversational AI systems.

Main components:
- fetch_data: Downloads datasets from HuggingFace
- loader: Loads and parses LongMemEval data structures  
- ingest: Ingests conversation data into Persona system
- answer: Generates answers using different strategies
- evaluate_qa: Evaluates answers using official LongMemEval methodology
- pipeline: Orchestrates the complete end-to-end evaluation

Usage:
    python -m evals.longmemeval.pipeline --dataset oracle --strategy hybrid --limit 3
"""

from .config import (
    API_BASE_URL,
    EVAL_MODEL, 
    DEFAULT_SUBSET_SIZE,
    DATA_DIR,
    RESULTS_DIR
)

from .fetch_data import download_dataset, load_dataset
from .loader import LongMemInstance, Turn, yield_instances, load_instances
from .pipeline import LongMemEvalPipeline

__version__ = "1.0.0"
__all__ = [
    "LongMemEvalPipeline",
    "download_dataset", 
    "load_dataset",
    "LongMemInstance",
    "Turn",
    "yield_instances",
    "load_instances"
] 