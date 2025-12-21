"""
Goals Layer Models for Persona v2 Identity Architecture.

Layer 3: Projects, tasks, todos, reminders, action items.
The "what you are doing" layer - practical and actionable.
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class Goal(BaseModel):
    """
    A project, task, todo, or action item.
    
    Goals are actionable items that connect to the user's life.
    They can be as simple as "buy groceries" or as complex as
    multi-month projects with sub-items.
    
    Examples:
    - "Launch MVP by end of Q1"
    - "Intro Sam to Maya for fundraising advice"
    - "Buy milk"
    """
    
    id: UUID = Field(default_factory=uuid4)
    
    # Type is fluid
    type: str = Field(
        default="task",
        description="Fluid type: 'task', 'project', 'reminder', 'todo', 'routine', etc."
    )
    
    # Content
    title: str = Field(
        ...,
        description="Short display title for the goal"
    )
    content: str = Field(
        default="",
        description="Description or context for the goal"
    )
    
    # Status
    status: str = Field(
        default="active",
        description="Fluid status: 'active', 'completed', 'archived', 'blocked', etc."
    )
    
    # Temporal anchoring
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this goal was created/recognized"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this record was created in the system"
    )
    due_date: Optional[datetime] = Field(
        default=None,
        description="Optional deadline"
    )
    
    # Origin tracking
    source_episode_id: Optional[UUID] = Field(
        default=None,
        description="Episode that led to this goal"
    )
    
    # Retrieval
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding for similarity search"
    )
    
    # User ownership
    user_id: str = Field(
        ...,
        description="User this goal belongs to"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440002",
                "type": "task",
                "title": "Intro Sam to Maya",
                "content": "Connect Sam with Maya for fundraising advice",
                "status": "active",
                "timestamp": "2024-12-19T10:30:00Z",
                "user_id": "user_001"
            }
        }


class GoalCreateInput(BaseModel):
    """Input structure for creating goals from LLM extraction."""
    type: str = Field(default="task")
    title: str
    content: str = Field(default="")
    status: str = Field(default="active")
    due_date: Optional[str] = Field(default=None)
