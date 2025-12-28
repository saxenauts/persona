"""
Memory Model for Persona v2 Identity Architecture.

A unified memory type that can represent:
- Episodes (narrative memory units, what happened)
- Psyche (traits, preferences, values, beliefs)
- Notes (tasks, projects, todos, reminders, facts, lists, contacts, ideas)

All memories are stored the same way, differentiated by `type`.
Links connect memories to each other.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union, Annotated, Literal
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class BaseMemory(BaseModel):
    """Base fields for all memory units."""

    id: UUID = Field(default_factory=uuid4)
    type: str = Field(
        ..., description="The discriminator field (episode, psyche, note)"
    )

    # Generic content
    title: str = Field(default="", description="Short title for display")
    content: str = Field(..., description="The memory content in natural language")

    # Temporal anchoring
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    day_id: Optional[str] = Field(default=None)

    # Provenance
    source_type: str = Field(default="conversation")
    source_ref: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(
        default=None, description="Source conversation session ID"
    )
    extraction_model: Optional[str] = Field(
        default=None, description="LLM model that extracted this memory"
    )
    extraction_confidence: Optional[float] = Field(
        default=None, description="Extraction confidence 0-1"
    )

    # Retrieval
    embedding: Optional[List[float]] = Field(default=None)

    # User ownership
    user_id: str = Field(...)

    # Retention & Importance
    access_count: int = Field(default=0)
    last_accessed: Optional[datetime] = Field(default=None)
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Memory importance score 0-1. Used for ordering and pruning.",
    )

    # Catch-all for forward compatibility
    properties: Dict[str, Any] = Field(default_factory=dict)


class EpisodeMemory(BaseMemory):
    """Narrative memory of an event."""

    type: Literal["episode"] = Field(default="episode")
    summary: Optional[str] = None


class PsycheMemory(BaseMemory):
    """Identity related memory (trait, preference, value, belief)."""

    type: Literal["psyche"] = Field(default="psyche")
    psyche_type: Optional[str] = Field(
        default=None, description="trait, preference, value, belief"
    )


class NoteMemory(BaseMemory):
    """Structured/unstructured notes: tasks, projects, facts, lists, contacts, ideas, reminders."""

    type: Literal["note"] = Field(default="note")
    note_type: Optional[str] = Field(
        default=None,
        description="task, project, fact, list, contact, reminder, idea, goal, etc.",
    )
    status: str = Field(default="active")
    due_date: Optional[datetime] = None


# The Unified Memory type using Discriminated Union
Memory = Annotated[
    Union[EpisodeMemory, PsycheMemory, NoteMemory], Field(discriminator="type")
]


class UserCard(BaseModel):
    """
    Compact identity anchor for LLM context.

    Updated after every session ingestion via consolidation.
    All fields are prose - no structured lists that assume categories.

    Design principles:
    - LLMs consume text naturally, so we provide prose
    - No category assumptions (roles, values, etc.) - let structure emerge
    - Consolidation rewrites entire prose (emergent, not additive)
    - Simple to update, simple to format
    """

    user_id: str
    timezone: Optional[str] = None

    # === CORE IDENTITY (LLM-generated prose) ===
    identity_prose: str = Field(
        default="",
        description="""
        2-3 sentences: Who this person is, their context, what matters to them.
        Updated on consolidation after each session. Written for LLM consumption.
        Example: "Alex is a software engineer in Austin who recently started 
        a fitness journey. They're focused on work-life balance and maintaining
        connections with family. Currently navigating a career transition."
        """,
    )

    # === METADATA ===
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = Field(default=2, description="Schema version for migrations")

    # === FUTURE: Memory Clusters ===
    # TODO: Phase 2 - Ranked clusters with pointers to linked memories
    # memory_clusters: List[MemoryCluster] = Field(default_factory=list)


class WorkingMemoryConfig(BaseModel):
    """Time tuner for working memory retrieval. Global defaults, overridable per-user."""

    episode_window: timedelta = Field(
        default=timedelta(days=2),
        description="How far back to fetch episodes",
    )
    psyche_window: timedelta = Field(
        default=timedelta(days=2),
        description="How far back to fetch psyche updates",
    )

    max_episodes: int = Field(default=10)
    max_psyche: int = Field(default=5)
    max_active_notes: int = Field(default=5)


DEFAULT_WORKING_MEMORY_CONFIG = WorkingMemoryConfig()


class MemoryLink(BaseModel):
    """
    A link (edge) between two memories.

    Links can represent:
    - Temporal: PREVIOUS/NEXT for chronological chains
    - Causal: caused_by, led_to, inspired
    - Reference: source_of, evidence_for, part_of
    """

    source_id: UUID
    target_id: UUID
    relation: str = Field(
        ..., description="Relationship type: PREVIOUS, NEXT, caused_by, etc."
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Additional edge properties"
    )


# ============================================================================
# Request/Response Models
# ============================================================================


class MemoryCreateRequest(BaseModel):
    """Request for creating a memory from raw input."""

    raw_content: str
    timestamp: Optional[datetime] = None
    session_id: Optional[str] = None
    source_type: str = "conversation"
    source_ref: Optional[str] = None


class MemoryQueryResponse(BaseModel):
    """Response for memory queries."""

    memories: List[Memory]
    total_count: int


# ============================================================================
# LLM Extraction Output Models
# ============================================================================


class EpisodeOutput(BaseModel):
    """LLM extraction output for episodes."""

    title: str
    content: str


class PsycheOutput(BaseModel):
    """LLM extraction output for psyche items."""

    type: str = Field(default="trait")
    content: str


class NoteOutput(BaseModel):
    """LLM extraction output for notes (tasks, facts, lists, etc.)."""

    type: str = Field(default="task")
    title: str
    content: str = Field(default="")
    status: str = Field(default="active")


class IngestionOutput(BaseModel):
    """Complete ingestion output from LLM."""

    episode: EpisodeOutput
    psyche: List[PsycheOutput] = Field(default_factory=list)
    notes: List[NoteOutput] = Field(default_factory=list)
