"""
Protocol definitions for the evaluation framework.

Design Principles:
- Use Protocol for structural subtyping (duck typing with type hints)
- Adapters return structured QueryResult, not raw strings
- Metrics are async by default (LLM-as-judge needs async)
- Benchmarks are pure data loaders (no execution logic)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from .models import (
    Session,
    TestCase,
    QueryResult,
    MetricResult,
    MetricKind,
    ScoreType,
)


@dataclass(frozen=True)
class AdapterCapabilities:
    """
    Declares what features an adapter supports.

    Design: Explicit capabilities allow graceful degradation.
    Engine can skip retrieval metrics if adapter doesn't support them.
    """

    supports_async: bool = False
    supports_bulk_ingest: bool = False
    supports_retrieval_items: bool = False  # Can return RetrievedItem list
    supports_context_text: bool = False  # Can return raw context string
    supports_reset: bool = True
    supports_user_namespace: bool = True  # Isolates users


@runtime_checkable
class MemorySystemAdapter(Protocol):
    """
    Protocol for memory system adapters.

    Design Principles:
    - Returns structured QueryResult (no last_query_stats hack)
    - Sync methods are required, async methods optional
    - Capabilities declare what the adapter supports

    Implementations:
    - PersonaAdapter (native, full featured)
    - GraphitiAdapter (Zep/Graphiti backend)
    - Mem0Adapter (Mem0 backend)
    - LegacyAdapterWrapper (wraps old-style adapters)
    """

    @property
    def name(self) -> str:
        """Unique identifier for this adapter."""
        ...

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Declare supported features."""
        ...

    # === Required sync methods ===

    def reset(self, user_id: str) -> None:
        """Clear all memory for a user. Required for clean benchmark isolation."""
        ...

    def add_sessions(self, user_id: str, sessions: Sequence[Session]) -> None:
        """Ingest sessions into memory. Adapters may batch or parallelize."""
        ...

    def query(self, user_id: str, query: str, *, trace: bool = True) -> QueryResult:
        """
        Query the memory system.

        Args:
            user_id: User namespace for isolation
            query: The question/query string
            trace: If True, populate retrieval info in QueryResult

        Returns:
            QueryResult with answer and optional retrieval/timing info
        """
        ...

    # === Optional async methods (for high throughput) ===

    async def areset(self, user_id: str) -> None:
        """Async version of reset."""
        ...

    async def aadd_sessions(self, user_id: str, sessions: Sequence[Session]) -> None:
        """Async version of add_sessions."""
        ...

    async def aquery(
        self, user_id: str, query: str, *, trace: bool = True
    ) -> QueryResult:
        """Async version of query."""
        ...


@runtime_checkable
class Metric(Protocol):
    """
    Protocol for evaluation metrics.

    Design Principles (from Hamel/Eugene research):
    - Single responsibility: one metric = one aspect
    - Async by default (LLM-as-judge needs it)
    - Returns binary pass/fail with reasoning
    - Composable via AllOf/AnyOf wrappers

    Examples:
    - BinaryExactMatch: string comparison
    - ContextPrecision: retrieval quality
    - LLMBinaryJudge: LLM-based evaluation
    """

    @property
    def name(self) -> str:
        """Unique metric name."""
        ...

    @property
    def kind(self) -> MetricKind:
        """Category: retrieval, generation, end_to_end, etc."""
        ...

    @property
    def score_type(self) -> ScoreType:
        """binary (0/1) or continuous (0.0-1.0)."""
        ...

    def required_capabilities(self) -> AdapterCapabilities:
        """
        Minimum adapter capabilities needed.

        Engine uses this to skip metrics that can't run
        (e.g., skip ContextPrecision if adapter doesn't return retrieved items).
        """
        ...

    async def evaluate(
        self,
        test_case: TestCase,
        query_result: QueryResult,
        *,
        resources: Mapping[str, Any],
    ) -> MetricResult:
        """
        Evaluate the test case.

        Args:
            test_case: The input (question, reference answer, labels)
            query_result: The adapter's output
            resources: Shared resources (LLM clients, embeddings, etc.)

        Returns:
            MetricResult with score, pass/fail, and reasoning
        """
        ...


@runtime_checkable
class Benchmark(Protocol):
    """
    Protocol for benchmark definitions.

    Design Principles:
    - Benchmarks are pure data loaders
    - No adapter calls, no evaluation logic
    - Return TestCase list, Engine handles execution
    - One official implementation per benchmark (OpenBench principle)
    """

    @property
    def name(self) -> str:
        """Benchmark identifier (e.g., 'personamem', 'longmemeval')."""
        ...

    @property
    def version(self) -> str:
        """Benchmark version for reproducibility."""
        ...

    def load(self, *, variant: Optional[str] = None) -> Sequence[TestCase]:
        """
        Load all test cases from the benchmark.

        Args:
            variant: Optional variant (e.g., '32k', '128k' for PersonaMem)

        Returns:
            Sequence of TestCase objects
        """
        ...

    def default_metrics(self) -> Sequence[str]:
        """
        Return metric names to use by default for this benchmark.

        Design: Benchmarks know which metrics make sense for them.
        """
        ...

    def sample(
        self,
        sizes: Mapping[str, int],
        *,
        seed: Optional[int] = None,
        variant: Optional[str] = None,
    ) -> Sequence[TestCase]:
        """
        Stratified sampling for quick evaluation runs.

        Args:
            sizes: {question_type: count} mapping
            seed: Random seed for reproducibility
            variant: Benchmark variant

        Returns:
            Sampled TestCase list
        """
        ...


@dataclass(frozen=True)
class Event:
    """
    Event for append-only storage.

    Design: Event-sourced storage enables flexible view materialization
    without coupling storage to evaluation logic.
    """

    type: str  # "run_started", "case_finished", etc.
    occurred_at: datetime
    run_id: str
    payload: Mapping[str, Any]
    payload_version: int = 1


class EventStore(Protocol):
    """
    Protocol for event storage.

    Design: Append-only events + materialized views.
    Keeps storage decoupled from evaluation logic.
    """

    @property
    def schema_version(self) -> int:
        """Current schema version for migrations."""
        ...

    def append(self, event: Event) -> None:
        """Append an event to the store."""
        ...

    def iter_events(
        self,
        *,
        run_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> Sequence[Event]:
        """Iterate events with optional filters."""
        ...

    def materialize_views(self, *, run_id: str) -> None:
        """Materialize derived views (summaries, aggregates) for a run."""
        ...
