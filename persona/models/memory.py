"""
Memory Model for Persona v2 Identity Architecture.

A unified memory type that can represent:
- Episodes (narrative memory units, what happened)
- Psyche (traits, preferences, values, beliefs)
- Goals (tasks, projects, todos, reminders)

All memories are stored the same way, differentiated by `type`.
Links connect memories to each other.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class Memory(BaseModel):
    """
    A unified memory unit.
    
    The `type` field determines what kind of memory this is:
    - episode: narrative memory (what happened)
    - psyche: trait, preference, value, belief
    - goal: task, project, todo, reminder
    
    All memories can link to other memories via edges.
    """
    
    id: UUID = Field(default_factory=uuid4)
    
    # Type determines the memory category
    type: str = Field(
        ...,
        description="Memory type: episode, psyche, goal, etc."
    )
    
    # Content
    title: str = Field(
        default="",
        description="Short title for display"
    )
    content: str = Field(
        ...,
        description="The memory content in natural language"
    )
    
    # Temporal anchoring
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this memory refers to"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this record was created in the system"
    )
    day_id: Optional[str] = Field(
        default=None,
        description="YYYY-MM-DD format for day-level queries"
    )
    
    # Optional fields (used by specific types)
    status: Optional[str] = Field(
        default=None,
        description="For goals: active, completed, blocked, etc."
    )
    due_date: Optional[datetime] = Field(
        default=None,
        description="For goals: deadline"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="For episodes: groups memories within a session"
    )
    
    # Provenance
    source_type: str = Field(
        default="conversation",
        description="Origin: conversation, instagram, spotify, etc."
    )
    source_ref: Optional[str] = Field(
        default=None,
        description="Original message ID, URL, etc."
    )
    
    # Retrieval
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding for similarity search"
    )
    
    # Decay mechanics
    access_count: int = Field(default=0)
    last_accessed: Optional[datetime] = Field(default=None)
    
    # User ownership
    user_id: str = Field(..., description="User this memory belongs to")
    
    # Additional properties (flexible)
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional type-specific properties"
    )


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
        ...,
        description="Relationship type: PREVIOUS, NEXT, caused_by, etc."
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional edge properties"
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


class GoalOutput(BaseModel):
    """LLM extraction output for goals."""
    type: str = Field(default="task")
    title: str
    content: str = Field(default="")
    status: str = Field(default="active")


class IngestionOutput(BaseModel):
    """Complete ingestion output from LLM."""
    episode: EpisodeOutput
    psyche: List[PsycheOutput] = Field(default_factory=list)
    goals: List[GoalOutput] = Field(default_factory=list)
