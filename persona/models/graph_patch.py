"""
GraphPatch Schema and IntegrationJob Model for Persona's Integration Agent.

The integration agent runs asynchronously to:
- Link related memories across sessions
- Merge duplicate entities
- Derive higher-order insights from patterns
- Detect and resolve conflicts

GraphPatch is the atomic unit of graph modification - a batch of operations
that can be applied transactionally to the memory graph.
"""

from datetime import datetime
from typing import Dict, Any, List, Literal, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


# ============================================================================
# GraphPatch Operations
# ============================================================================


class LinkOp(BaseModel):
    """Create a relationship between two existing memories.

    Links are the edges in the memory graph - they represent semantic,
    temporal, or causal relationships discovered during integration.
    """

    source_id: UUID = Field(..., description="UUID of the source memory node")
    target_id: UUID = Field(..., description="UUID of the target memory node")
    relation: str = Field(
        ...,
        description="Relationship type: LED_TO, CAUSED_BY, MENTIONS, RELATES_TO, etc.",
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional edge properties (weight, confidence, evidence)",
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in this link"
    )


class MergeOp(BaseModel):
    """Merge a duplicate entity/memory into a canonical one.

    When integration detects duplicates (e.g., 'Sarah' and 'Sarah Smith'),
    merge combines them: all links from duplicate redirect to canonical,
    then duplicate is marked as merged.
    """

    canonical_id: UUID = Field(..., description="UUID of the memory to keep")
    duplicate_id: UUID = Field(
        ..., description="UUID of the memory to merge into canonical"
    )
    reason: str = Field(
        default="",
        description="Why these were identified as duplicates",
    )


class DerivedOp(BaseModel):
    """Create a new memory derived from existing ones.

    Derived memories are higher-order insights:
    - Pattern detection: "User exercises every Monday"
    - Entity consolidation: Summary of what we know about 'Sarah'
    - Psyche evolution: Updated trait from multiple observations
    """

    content: str = Field(..., description="The derived memory content")
    memory_type: Literal["episode", "psyche", "entity", "note"] = Field(
        ...,
        description="Type of derived memory",
    )
    source_ids: List[UUID] = Field(
        ...,
        description="UUIDs of memories this was derived from",
    )
    title: str = Field(default="", description="Optional title for the derived memory")
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional properties for the derived memory",
    )


class ConflictOp(BaseModel):
    """Flag a conflict between two memories for resolution.

    Conflicts arise when memories contain contradictory information:
    - "Sarah's birthday is June 5" vs "Sarah's birthday is June 15"
    - "User prefers morning workouts" vs recent pattern of evening workouts

    Resolution options:
    - keep_both: Both are valid (different contexts)
    - supersede: memory_b replaces memory_a (temporal update)
    - invalidate: memory_a is wrong, remove it
    """

    memory_a: UUID = Field(..., description="UUID of first conflicting memory")
    memory_b: UUID = Field(..., description="UUID of second conflicting memory")
    reason: str = Field(..., description="Description of the conflict")
    resolution: Optional[Literal["keep_both", "supersede", "invalidate"]] = Field(
        default=None,
        description="Resolution strategy (None means flagged for human review)",
    )
    auto_resolved: bool = Field(
        default=False,
        description="Whether resolution was applied automatically",
    )


# ============================================================================
# GraphPatch: Atomic Batch of Operations
# ============================================================================


class GraphPatch(BaseModel):
    """A batch of graph operations to apply atomically.

    GraphPatch is the output of the integration agent - a declarative
    specification of what changes to make to the memory graph.

    Design principles:
    - Idempotent: Applying the same patch twice has no additional effect
    - Atomic: All operations succeed or none do
    - Auditable: run_id provides provenance for all changes
    """

    run_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique ID for this integration run (for provenance)",
    )
    user_id: str = Field(..., description="User ID this patch applies to")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    links: List[LinkOp] = Field(
        default_factory=list,
        description="New relationships to create between memories",
    )
    merges: List[MergeOp] = Field(
        default_factory=list,
        description="Duplicate memories to merge",
    )
    derived: List[DerivedOp] = Field(
        default_factory=list,
        description="New memories to create from existing ones",
    )
    conflicts: List[ConflictOp] = Field(
        default_factory=list,
        description="Conflicts to flag or resolve",
    )
    processed_ids: List[UUID] = Field(
        default_factory=list,
        description="Memory IDs to mark as integrated (won't be processed again)",
    )

    def is_empty(self) -> bool:
        """Check if this patch has any operations."""
        return (
            not self.links
            and not self.merges
            and not self.derived
            and not self.conflicts
            and not self.processed_ids
        )

    def operation_count(self) -> int:
        """Total number of operations in this patch."""
        return (
            len(self.links) + len(self.merges) + len(self.derived) + len(self.conflicts)
        )


# ============================================================================
# GraphPatchResult: Outcome of Applying a Patch
# ============================================================================


class OperationResult(BaseModel):
    """Result of a single operation."""

    operation_type: str = Field(..., description="link, merge, derived, or conflict")
    success: bool = Field(..., description="Whether the operation succeeded")
    details: Optional[str] = Field(default=None, description="Additional info")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class GraphPatchResult(BaseModel):
    """Result of applying a GraphPatch to the memory graph.

    Provides detailed feedback on what was applied and any errors.
    """

    success: bool = Field(..., description="True if patch applied without errors")
    run_id: str = Field(..., description="The run_id from the applied patch")
    user_id: str = Field(..., description="User ID this patch was applied to")
    applied_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the patch was applied",
    )

    links_created: int = Field(default=0, description="Number of links created")
    merges_applied: int = Field(default=0, description="Number of merges completed")
    derived_created: int = Field(
        default=0, description="Number of derived memories created"
    )
    conflicts_flagged: int = Field(
        default=0, description="Number of conflicts recorded"
    )
    memories_marked_integrated: int = Field(
        default=0,
        description="Number of memories marked as integrated",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Error messages for failed operations",
    )
    operation_results: List[OperationResult] = Field(
        default_factory=list, description="Detailed results per operation"
    )

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def total_operations(self) -> int:
        return (
            self.links_created
            + self.merges_applied
            + self.derived_created
            + self.conflicts_flagged
        )


# ============================================================================
# IntegrationJob: Job Queue Record
# ============================================================================


class IntegrationScope(BaseModel):
    """Defines what memories to process in an integration job.

    Exactly one of these should be set to define the scope:
    - session_id: Process all memories from a specific session
    - date_range: Process memories within a time window
    - memory_ids: Process specific memories by ID
    """

    session_id: Optional[str] = Field(
        default=None,
        description="Process all memories from this session",
    )
    date_start: Optional[datetime] = Field(
        default=None,
        description="Start of date range (inclusive)",
    )
    date_end: Optional[datetime] = Field(
        default=None,
        description="End of date range (exclusive)",
    )
    memory_ids: List[UUID] = Field(
        default_factory=list,
        description="Specific memory IDs to process",
    )

    def is_valid(self) -> bool:
        """Check that exactly one scope dimension is set."""
        has_session = self.session_id is not None
        has_date_range = self.date_start is not None or self.date_end is not None
        has_memory_ids = len(self.memory_ids) > 0
        return sum([has_session, has_date_range, has_memory_ids]) == 1


class IntegrationJob(BaseModel):
    """A queued integration job for async processing.

    Integration jobs are created after ingestion and processed by a
    background worker. They support:
    - Delayed execution (run_after)
    - Checkpointing for resumability
    - Status tracking for observability
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique job identifier",
    )
    user_id: str = Field(..., description="User whose memories to integrate")
    scope: IntegrationScope = Field(
        default_factory=IntegrationScope,
        description="What memories to process",
    )
    run_after: datetime = Field(
        default_factory=datetime.utcnow,
        description="Don't start before this time (for delayed execution)",
    )
    checkpoint: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Cursor for resuming interrupted jobs",
    )
    status: Literal["pending", "running", "complete", "failed"] = Field(
        default="pending",
        description="Current job status",
    )
    run_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique ID for this run (links to GraphPatch.run_id)",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the job was created",
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="When the job started running",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When the job finished (success or failure)",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if status is 'failed'",
    )
    result: Optional[GraphPatchResult] = Field(
        default=None,
        description="Result of the integration run",
    )

    # Retry configuration
    attempt: int = Field(default=1, description="Current attempt number")
    max_attempts: int = Field(default=3, description="Maximum retry attempts")

    def is_ready(self, now: Optional[datetime] = None) -> bool:
        """Check if job is ready to run."""
        now = now or datetime.utcnow()
        return self.status == "pending" and self.run_after <= now

    def can_retry(self) -> bool:
        """Check if job can be retried after failure."""
        return self.status == "failed" and self.attempt < self.max_attempts

    def mark_running(self) -> None:
        """Transition job to running state."""
        self.status = "running"
        self.started_at = datetime.utcnow()

    def mark_complete(self, result: GraphPatchResult) -> None:
        """Transition job to complete state."""
        self.status = "complete"
        self.completed_at = datetime.utcnow()
        self.result = result

    def mark_failed(self, error: str) -> None:
        """Transition job to failed state."""
        self.status = "failed"
        self.completed_at = datetime.utcnow()
        self.error = error
