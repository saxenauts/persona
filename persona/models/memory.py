"""
Memory Models for Persona v2 Identity Architecture.

The three layers of the identity graph:
- Episode: Narrative memory units (what happened)
- Psyche: Traits, preferences, values, beliefs (who you are)
- Goal: Tasks, projects, todos, reminders (what you're doing)
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


# ============================================================================
# Layer 1: Episodes (The Narrative Foundation)
# ============================================================================

class Episode(BaseModel):
    """
    A narrative memory unit - not a triplet, not a single fact, but a coherent
    story chunk that preserves context.
    """
    
    id: UUID = Field(default_factory=uuid4)
    
    # Content
    title: str = Field(
        ..., 
        description="Concise title (2-10 words)"
    )
    content: str = Field(
        ...,
        description="Narrative content preserving context and nuance"
    )
    
    # Temporal anchoring
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this episode happened"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this record was created in the system"
    )
    day_id: str = Field(
        ...,
        description="YYYY-MM-DD format for day-level queries"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Groups episodes within a conversation session"
    )
    
    # Provenance
    source_type: str = Field(
        default="conversation",
        description="Origin type: conversation, instagram, spotify, etc."
    )
    source_ref: Optional[str] = Field(
        default=None,
        description="Original message ID, URL, file path, etc."
    )
    
    # Retrieval
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding for similarity search"
    )
    
    # Decay mechanics
    access_count: int = Field(default=0)
    last_accessed: Optional[datetime] = Field(default=None)
    
    # Temporal chain
    previous_episode_id: Optional[UUID] = Field(default=None)
    next_episode_id: Optional[UUID] = Field(default=None)
    
    # User ownership
    user_id: str = Field(..., description="User this episode belongs to")


# ============================================================================
# Layer 2: Psyche (Traits, Preferences, Values, Beliefs)
# ============================================================================

class Psyche(BaseModel):
    """
    A trait, belief, preference, or value - the fragments of identity.
    """
    
    id: UUID = Field(default_factory=uuid4)
    
    # Type is fluid
    type: str = Field(
        default="trait",
        description="Fluid type: trait, preference, value, belief, arc, etc."
    )
    
    # Content
    content: str = Field(
        ...,
        description="The trait/preference/belief in natural language"
    )
    
    # Temporal anchoring
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Origin tracking
    source_episode_id: Optional[UUID] = Field(
        default=None,
        description="Episode that led to this psyche item"
    )
    
    # Retrieval
    embedding: Optional[List[float]] = Field(default=None)
    
    # User ownership
    user_id: str = Field(...)


# ============================================================================
# Layer 3: Goals (Tasks, Projects, Todos)
# ============================================================================

class Goal(BaseModel):
    """
    A project, task, todo, or action item.
    """
    
    id: UUID = Field(default_factory=uuid4)
    
    # Type is fluid
    type: str = Field(
        default="task",
        description="Fluid type: task, project, reminder, todo, routine, etc."
    )
    
    # Content
    title: str = Field(..., description="Short display title")
    content: str = Field(default="", description="Description or context")
    
    # Status
    status: str = Field(
        default="active",
        description="Fluid status: active, completed, archived, blocked, etc."
    )
    
    # Temporal anchoring
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    due_date: Optional[datetime] = Field(default=None)
    
    # Origin tracking
    source_episode_id: Optional[UUID] = Field(default=None)
    
    # Retrieval
    embedding: Optional[List[float]] = Field(default=None)
    
    # User ownership
    user_id: str = Field(...)


# ============================================================================
# Request/Response Models
# ============================================================================

class EpisodeCreateRequest(BaseModel):
    """Request model for creating an episode from raw input."""
    raw_content: str
    timestamp: Optional[datetime] = None
    session_id: Optional[str] = None
    source_type: str = "conversation"
    source_ref: Optional[str] = None


class EpisodeChainResponse(BaseModel):
    """Response model for temporal chain queries."""
    episodes: List[Episode]
    start_date: str
    end_date: str
    total_count: int


# ============================================================================
# LLM Extraction Input Models
# ============================================================================

class PsycheCreateInput(BaseModel):
    """Input structure for creating psyche items from LLM extraction."""
    type: str = Field(default="trait")
    content: str


class GoalCreateInput(BaseModel):
    """Input structure for creating goals from LLM extraction."""
    type: str = Field(default="task")
    title: str
    content: str = Field(default="")
    status: str = Field(default="active")
    due_date: Optional[str] = Field(default=None)
