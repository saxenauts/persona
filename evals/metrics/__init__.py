"""
Metrics module for evaluation.

Provides composable metrics following research best practices:
- Binary pass/fail preferred over numeric scales
- Single responsibility per metric
- Async by default for LLM-as-judge support
"""

from .base import BaseMetric, AllOf, AnyOf, ThresholdGate
from .exact_match import BinaryExactMatch, OptionExtractor, ContainsAnswer
from .retrieval import ContextPrecision, ContextRecall
from .llm_judge import LLMBinaryJudge, AbstentionAccuracy, SemanticSimilarity

__all__ = [
    # Base
    "BaseMetric",
    "AllOf",
    "AnyOf",
    "ThresholdGate",
    # Exact match
    "BinaryExactMatch",
    "OptionExtractor",
    "ContainsAnswer",
    # Retrieval
    "ContextPrecision",
    "ContextRecall",
    # LLM Judge
    "LLMBinaryJudge",
    "AbstentionAccuracy",
    "SemanticSimilarity",
]
