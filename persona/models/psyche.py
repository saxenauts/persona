"""
Psyche Layer Models for Persona v2 Identity Architecture.

Layer 2: Traits, preferences, values, beliefs, emotional truths.
The "who you are becoming" layer that evolves from episodes.
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class Psyche(BaseModel):
    """
    A trait, belief, preference, or value - the fragments of identity.
    
    These are fluid interpretations of who the user is, derived from
    episodes or stated directly. Think of them as qualia or memes.
    
    Examples:
    - "Prefers remote work"
    - "Risk-averse since the burnout in 2023"
    - "Values deep human connection over networking"
    """
    
    id: UUID = Field(default_factory=uuid4)
    
    # Type is fluid, not a strict enum
    type: str = Field(
        default="trait",
        description="Fluid type: 'trait', 'preference', 'value', 'belief', 'arc', etc."
    )
    
    # Content
    content: str = Field(
        ...,
        description="The trait/preference/belief in natural language"
    )
    
    # Temporal anchoring
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this was recognized"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this record was created in the system"
    )
    
    # Origin tracking
    source_episode_id: Optional[UUID] = Field(
        default=None,
        description="Episode that led to this psyche item (evidence link)"
    )
    
    # Retrieval
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding for similarity search"
    )
    
    # User ownership
    user_id: str = Field(
        ...,
        description="User this psyche item belongs to"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440001",
                "type": "preference",
                "content": "Prefers deep work sessions of 3+ hours without interruption",
                "timestamp": "2024-12-19T10:30:00Z",
                "user_id": "user_001"
            }
        }


class PsycheCreateInput(BaseModel):
    """Input structure for creating psyche items from LLM extraction."""
    type: str = Field(default="trait")
    content: str
