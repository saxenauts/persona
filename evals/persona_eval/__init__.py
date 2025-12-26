"""
PersonaEval: Simple, extensible evaluation analysis for memory systems.

This module provides:
- SQLite provenance database for querying failures
- RAGAS-style retrieval metrics (precision, recall)
- Failure taxonomy classification
- Analysis utilities for error analysis

Philosophy: Just write Python. Keep it simple. Focus on root cause analysis.
"""

from .database import EvalDatabase
from .metrics import RetrievalMetrics
from .failure_classifier import FailureClassifier
from .analyzer import EvalAnalyzer

__all__ = [
    "EvalDatabase",
    "RetrievalMetrics",
    "FailureClassifier",
    "EvalAnalyzer",
]
