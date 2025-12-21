"""
Episode Layer Models for Persona v2 Identity Architecture.

This module defines the Episode model - the foundational narrative memory units
that replace triplets with contextual, temporally-linked story chunks.
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class Episode(BaseModel):
    """
    A narrative memory unit - not a triplet, not a single fact, but a coherent
    story chunk that preserves context. LLMs reason beautifully over stories.
    
    Episodes form the foundation layer (Layer 1) of the Persona identity graph,
    feeding both the Personality layer (evidence) and Goals layer (artifacts).
    """
    
    id: UUID = Field(default_factory=uuid4)
    
    # Content (LLM generates this as natural prose)
    title: str = Field(
        ..., 
        description="Concise title (2-10 words), e.g., 'Morning standup about API refactor'"
    )
    content: str = Field(
        ...,
        description="Fluid narrative - could be a sentence or session summary. "
                    "Includes who, what, when, why, emotional context, implications."
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
        description="YYYY-MM-DD format for day-level queries, e.g., '2024-12-19'"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Groups episodes within a conversation session"
    )
    
    # Provenance chain (trust + debugging)
    source_type: str = Field(
        default="conversation",
        description="Origin type: 'conversation', 'instagram', 'spotify', 'linear', etc."
    )
    source_ref: Optional[str] = Field(
        default=None,
        description="Original message ID, URL, file path, etc."
    )
    
    # Retrieval signals
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding for similarity search"
    )
    
    # Decay mechanics (will be intelligent, not just statistical)
    access_count: int = Field(default=0)
    last_accessed: Optional[datetime] = Field(default=None)
    
    # Causality chain position (temporal linking)
    previous_episode_id: Optional[UUID] = Field(
        default=None,
        description="Link to temporally previous episode (PREV in chain)"
    )
    next_episode_id: Optional[UUID] = Field(
        default=None,
        description="Link to temporally next episode (NEXT in chain)"
    )
    
    # User ownership
    user_id: str = Field(
        ...,
        description="User this episode belongs to"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "Coffee conversation about startup funding",
                "content": "Met with Sam for coffee at Blue Bottle. He's stressed about his Series A - "
                           "investors are pushing back on valuation. I offered to intro him to Maya "
                           "who helped me navigate similar talks. The conversation reminded me of my "
                           "own fundraising anxiety last year.",
                "timestamp": "2024-12-19T10:30:00Z",
                "day_id": "2024-12-19",
                "session_id": "conv_abc123",
                "source_type": "conversation",
                "source_ref": "msg_12345",
                "user_id": "user_001"
            }
        }


class EpisodeCreateRequest(BaseModel):
    """Request model for creating an episode from raw input."""
    
    raw_content: str = Field(
        ...,
        description="The raw input text to convert into an episode"
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="When this happened (defaults to now)"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session grouping"
    )
    source_type: str = Field(
        default="conversation",
        description="Origin type of the content"
    )
    source_ref: Optional[str] = Field(
        default=None,
        description="Reference to original source"
    )


class EpisodeResponse(BaseModel):
    """Response model for episode queries."""
    
    episode: Episode
    previous_title: Optional[str] = Field(
        default=None,
        description="Title of previous episode in chain (for context)"
    )
    next_title: Optional[str] = Field(
        default=None,
        description="Title of next episode in chain (for context)"
    )


class EpisodeChainResponse(BaseModel):
    """Response model for temporal chain queries."""
    
    episodes: List[Episode]
    start_date: str
    end_date: str
    total_count: int
