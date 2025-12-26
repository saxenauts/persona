"""
Core data models for the evaluation framework.

Design Principles (from research):
- Immutable dataclasses for thread safety and debugging
- Clear separation: TestCase (input) -> QueryResult (output) -> MetricResult (score)
- Favor composition over inheritance
- Binary pass/fail over uncalibrated 1-5 scales (Hamel Husain)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence, Literal, Union

# Type aliases for clarity
ScoreType = Literal["binary", "continuous"]
MetricKind = Literal[
    "retrieval", "generation", "end_to_end", "safety", "latency", "cost"
]


@dataclass(frozen=True)
class Session:
    """
    A single session/conversation to be ingested into the memory system.

    Design: Minimal required fields, extensible via metadata.
    """

    content: str
    date: Optional[str] = None  # "YYYY-MM-DD" format when temporal context matters
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TestCase:
    """
    An atomic unit of evaluation - one question/query to test.

    Design Principles:
    - Benchmarks produce TestCases, Engine consumes them
    - No benchmark-specific logic embedded here
    - reference dict holds labels for metric computation (e.g., relevant_item_ids)
    - tags enable filtering/stratification without subclassing
    """

    id: str
    benchmark: str
    user_id: str
    query: str
    sessions: Sequence[Session]  # What gets ingested before query

    # Ground truth
    reference_answer: Optional[str] = None  # Gold answer string
    reference: Mapping[str, Any] = field(default_factory=dict)  # Labels, doc_ids, etc.

    # Categorization
    question_type: Optional[str] = None  # e.g., "temporal_reasoning", "multi_session"
    tags: Sequence[str] = field(default_factory=tuple)

    # Additional context
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # For multiple choice questions (PersonaMem style)
    options: Optional[Mapping[str, str]] = (
        None  # {"a": "Option A", "b": "Option B", ...}
    )
    correct_option: Optional[str] = None  # "a", "b", "c", or "d"


@dataclass(frozen=True)
class RetrievedItem:
    """
    A single item retrieved by the memory system.

    Design: Captures enough info for retrieval metrics (precision, recall)
    without being coupled to any specific backend.
    """

    id: str
    text: str
    score: Optional[float] = None  # Similarity/relevance score
    rank: Optional[int] = None  # Position in retrieval results
    source: Optional[str] = None  # "vector", "graph", "memory", "sql"
    node_type: Optional[str] = None  # "episode", "fact", "preference", etc.
    timestamp: Optional[str] = None  # When this memory was created
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Usage:
    """
    Token usage and cost tracking.

    Design: Provider-agnostic, captures common metrics.
    """

    provider: Optional[str] = None  # "openai", "azure_openai", "anthropic"
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost_usd: Optional[float] = None  # If available


@dataclass(frozen=True)
class Latency:
    """
    Timing breakdown for the query pipeline.

    Design: Separates ingestion, retrieval, and generation for diagnosis.
    """

    ingestion_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    total_ms: Optional[float] = None


@dataclass(frozen=True)
class QueryResult:
    """
    Structured response from a memory system query.

    Design Principles:
    - Replaces the `last_query_stats` hack with explicit structure
    - Everything needed for metrics is here, not in side channels
    - Immutable for thread safety across parallel evaluation
    """

    answer: str

    # Retrieval info (for retrieval metrics)
    retrieved: Sequence[RetrievedItem] = field(default_factory=tuple)
    context_text: Optional[str] = None  # Raw context passed to generator

    # Performance tracking
    usage: Usage = field(default_factory=Usage)
    latency: Latency = field(default_factory=Latency)

    # Debugging
    telemetry: Mapping[str, Any] = field(
        default_factory=dict
    )  # Backend-specific traces
    error: Optional[str] = None  # Error message if query failed

    # Timestamps
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


@dataclass(frozen=True)
class MetricResult:
    """
    Result of evaluating a single metric on a test case.

    Design Principles (from Hamel/Eugene research):
    - Binary pass/fail is preferred over numeric scales
    - Continuous scores (0-1) with threshold → binary decision
    - Always include reasoning for debuggability
    - Artifacts store raw judge outputs, intermediate data
    """

    metric: str  # Metric name
    kind: MetricKind
    score_type: ScoreType

    score: float  # 0/1 for binary, 0.0-1.0 for continuous
    passed: bool  # The decision (score >= threshold)

    reason: str = ""  # Human-readable explanation
    threshold: Optional[float] = None  # Threshold used for pass/fail

    # For debugging and auditing
    artifacts: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalResult:
    """
    Complete evaluation result for a single test case.

    Design: Combines QueryResult + all MetricResults + pass/fail decision.
    """

    run_id: str
    adapter: str
    benchmark: str
    test_case_id: str
    question_type: Optional[str] = None

    # Results
    query_result: QueryResult = field(default_factory=lambda: QueryResult(answer=""))
    metric_results: Sequence[MetricResult] = field(default_factory=tuple)

    # Overall pass/fail (typically all metrics must pass)
    passed: bool = False

    # Timing
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    # Additional data
    artifacts: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunSpec:
    """
    Specification for an evaluation run.

    Design: Captures all parameters needed to reproduce a run.
    """

    run_id: str
    created_at: datetime

    adapters: Sequence[str]
    benchmarks: Sequence[str]
    metric_names: Sequence[str]

    # Configuration
    concurrency: int = 10
    rate_limit_rpm: Optional[int] = None
    random_seed: Optional[int] = None

    parameters: Mapping[str, Any] = field(default_factory=dict)
