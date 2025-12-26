"""
Core data models and interfaces for the evaluation framework.

This module provides the foundational types used throughout the evals system:
- Data models: TestCase, QueryResult, MetricResult, EvalResult
- Interfaces: MemorySystemAdapter, Metric, Benchmark protocols
- Compatibility: LegacyAdapterWrapper for existing adapters
"""

from .models import (
    Session,
    TestCase,
    RetrievedItem,
    Usage,
    Latency,
    QueryResult,
    MetricResult,
    EvalResult,
    RunSpec,
    ScoreType,
    MetricKind,
)

from .interfaces import (
    AdapterCapabilities,
    MemorySystemAdapter,
    Metric,
    Benchmark,
    Event,
    EventStore,
)

from .compat import LegacyAdapterWrapper, wrap_legacy_adapter

__all__ = [
    # Models
    "Session",
    "TestCase",
    "RetrievedItem",
    "Usage",
    "Latency",
    "QueryResult",
    "MetricResult",
    "EvalResult",
    "RunSpec",
    "ScoreType",
    "MetricKind",
    # Interfaces
    "AdapterCapabilities",
    "MemorySystemAdapter",
    "Metric",
    "Benchmark",
    "Event",
    "EventStore",
    # Compat
    "LegacyAdapterWrapper",
    "wrap_legacy_adapter",
]
