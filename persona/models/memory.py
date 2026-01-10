"""
Memory Model for Persona Identity Architecture.

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
    # Note: `type` is defined in subclasses as Literal for discriminated union

    # Generic content
    title: str = Field(default="", description="Short title for display")
    content: str = Field(..., description="The memory content in natural language")

    # Temporal anchoring
    event_time: datetime = Field(default_factory=datetime.utcnow)
    observed_at: datetime = Field(default_factory=datetime.utcnow)
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

    # Integration tracking
    integrated_at: Optional[datetime] = Field(default=None)
    integrated_by: Optional[str] = Field(default=None)

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
    """Agent commitments: tasks, goals, reminders, ideas, lists, projects (tracking status).

    Notes are created when there's INTENTION or TRIGGER:
    - "remind me", "I need to", due dates, imperatives
    - State machine: active → done/cancelled

    NOT for facts (those are Entity attributes).
    """

    type: Literal["note"] = Field(default="note")
    note_type: Optional[str] = Field(
        default=None,
        description="task, goal, reminder, idea, list, project (for tracking status)",
    )
    status: str = Field(default="active")
    due_date: Optional[datetime] = None
    entity_refs: List[UUID] = Field(
        default_factory=list,
        description="Entity IDs this note relates to (e.g., reminder about a person)",
    )


# ============================================================================
# Entity Memory System (4th Pillar)
# ============================================================================


class EntityAttribute(BaseModel):
    """A structured attribute of an entity with evidence provenance.

    Attributes are facts about entities (birthday, job title, favorite color).
    Each attribute tracks where the knowledge came from for conflict resolution.
    """

    key: str = Field(..., description="Attribute name: birthday, job_title, location")
    value: str = Field(..., description="Attribute value")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_id: Optional[UUID] = Field(
        default=None, description="Episode/Note ID that evidences this attribute"
    )
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class EntityRelation(BaseModel):
    """A relationship from this entity to another entity.

    Relations are typed connections: works_at, married_to, located_in, part_of.
    """

    target_entity_id: UUID
    relation_type: str = Field(..., description="works_at, married_to, friend_of, etc.")
    properties: Dict[str, Any] = Field(default_factory=dict)
    evidence_id: Optional[UUID] = Field(default=None)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class EntityMemory(BaseMemory):
    """Semantic memory about world referents (people, places, things, concepts).

    Entities answer "What/Who is X?" - they represent things that EXIST in the world.
    Facts about entities are stored as attributes, not as separate Notes.

    Key distinction from Notes:
    - Entity = Things that EXIST (nouns: Sarah, Paris, Project Alpha)
    - Note = Things I INTEND to do (verbs: call Sarah, finish project)

    Update semantics: Upsert with conflict handling (attributes can be updated).
    """

    type: Literal["entity"] = Field(default="entity")
    entity_type: str = Field(
        ..., description="person, place, organization, project, tool, concept"
    )

    # Identity
    canonical_name: str = Field(..., description="Primary name: 'Sarah Smith'")
    aliases: List[str] = Field(
        default_factory=list,
        description="Alternative names: ['Sarah', 'my girlfriend']",
    )
    description: str = Field(
        default="", description="LLM-consolidated summary of what we know"
    )

    # Structured attributes (facts about this entity)
    attributes: List[EntityAttribute] = Field(
        default_factory=list,
        description="Structured facts: [{key: 'birthday', value: 'June 5', ...}]",
    )

    # Relationships to other entities
    relationships: List[EntityRelation] = Field(
        default_factory=list,
        description="Links to other entities: [{target_id, relation_type: 'works_at'}]",
    )

    # Back-references (which memories mention this entity)
    mentioned_in: List[UUID] = Field(
        default_factory=list,
        description="Episode/Note IDs that mention this entity",
    )


# The Unified Memory type using Discriminated Union
Memory = Annotated[
    Union[EpisodeMemory, PsycheMemory, NoteMemory, EntityMemory],
    Field(discriminator="type"),
]


class UserCard(BaseModel):
    """
    Compact identity anchor for LLM context.

    #TODO: Updated after every session ingestion via consolidation.
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
        a fitness journey. He is focused on work-life balance and maintaining
        connections with family. He is currently navigating a career transition.
        He wants to start a new content business along the way."
        """,
    )

    # TODO: IMPORTANT: Need a better prose.

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

    # TODO for later evaluation, find the right balance for context window.
    max_episodes: int = Field(default=10)
    max_psyche: int = Field(default=10)
    max_active_notes: int = Field(default=10)


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
    """LLM extraction output for notes (tasks, goals, reminders, ideas, lists)."""

    type: str = Field(default="task")
    title: str
    content: str = Field(default="")
    status: str = Field(default="active")
    entity_refs: List[str] = Field(
        default_factory=list,
        description="Names of entities this note relates to (for linking)",
    )


class EntityAttributeOutput(BaseModel):
    """LLM extraction output for entity attributes."""

    key: str
    value: str


class EntityOutput(BaseModel):
    """LLM extraction output for entities (people, places, things, concepts)."""

    entity_type: str = Field(
        ..., description="person, place, organization, project, tool, concept"
    )
    canonical_name: str
    aliases: List[str] = Field(default_factory=list)
    description: str = Field(default="")
    attributes: List[EntityAttributeOutput] = Field(default_factory=list)


class IngestionOutput(BaseModel):
    """Complete ingestion output from LLM."""

    event_time: Optional[str] = Field(
        default=None,
        description="ISO 8601 timestamp of when the event actually happened",
    )
    episode: EpisodeOutput
    psyche: List[PsycheOutput] = Field(default_factory=list)
    notes: List[NoteOutput] = Field(default_factory=list)
    entities: List[EntityOutput] = Field(default_factory=list)


# ============================================================================
# Memeplex: Per-User Memory Index
# ============================================================================


class ActiveMemory(BaseModel):
    """A currently active memory (note, entity, or topic cluster).

    Represents what's "in focus" for this user right now.
    """

    memory_id: UUID
    memory_type: str = Field(..., description="note, entity, or topic")
    title: str
    keywords: List[str] = Field(default_factory=list)
    last_touched: datetime = Field(default_factory=datetime.utcnow)
    context_snippet: str = Field(
        default="", description="1 sentence of why it's active"
    )


class MemoryStats(BaseModel):
    """Aggregate stats about user's memory for overview."""

    total_memories: int = 0
    total_episodes: int = 0
    total_psyche: int = 0
    total_notes: int = 0
    total_entities: int = 0
    active_notes: int = 0
    earliest_memory: Optional[datetime] = None
    latest_memory: Optional[datetime] = None
    session_count: int = 0


class TemporalContext(BaseModel):
    """Temporal grounding for the user's memory.

    Provides week/month summaries that update with consolidation.
    Helps LLM understand temporal context without needing to query.
    """

    current_date: str = Field(default="", description="Today's date for reference")

    # This week's context
    week_summary: str = Field(
        default="",
        description="2-3 sentences: What's been happening this week. Key events, themes, people.",
    )
    week_start: str = Field(default="", description="ISO date of week start (Monday)")

    # This month's context
    month_summary: str = Field(
        default="", description="2-3 sentences: Major themes and events this month."
    )
    month_name: str = Field(default="", description="e.g., 'January 2026'")

    # Key dates the user cares about (extracted from memories)
    upcoming: List[str] = Field(
        default_factory=list,
        description="Up to 3 upcoming events/deadlines mentioned in memories",
    )


class Memeplex(BaseModel):
    """Per-user memory index. Stored as single JSON node in Neo4j.

    The Memeplex is the "table of contents" for a user's memory:
    - Topics/themes they care about (universal - everyone has these)
    - Known entities (people, projects, places, concepts - optional)
    - Time windows (what's been active recently)
    - Temporal context (this week, this month)

    Lives in system prompt for immediate LLM access.
    Updated after ingestion during integration phase.

    Design: Topics are universal. Entities are optional.
    A "loner" might only have topics. A social person has both.
    """

    user_id: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # === TOPICS (Universal - everyone has these) ===
    topics: List[str] = Field(
        default_factory=list,
        description="Main topics/themes user cares about. Max 15. e.g. ['fitness', 'AI research', 'cooking']",
    )

    # === ENTITIES (Optional - depends on user's world) ===
    people: List[str] = Field(
        default_factory=list,
        description="Known people with relationship. Max 20. e.g. ['Sarah (wife)', 'Max (colleague)']",
    )
    projects: List[str] = Field(
        default_factory=list,
        description="Active or notable projects. Max 15. e.g. ['Persona', 'Home Renovation']",
    )
    places: List[str] = Field(
        default_factory=list,
        description="Significant places. Max 10. e.g. ['SF (home)', 'Tokyo (2024 trip)']",
    )
    concepts: List[str] = Field(
        default_factory=list,
        description="Abstract concepts/philosophies. Max 10. e.g. ['stoicism', 'minimalism']",
    )

    # === TIME WINDOWS (What's been active) ===
    last_week_topics: List[str] = Field(
        default_factory=list,
        description="Topics active in last 7 days. e.g. ['Persona v1', 'fitness']",
    )
    last_month_topics: List[str] = Field(
        default_factory=list,
        description="Topics active in last 30 days.",
    )
    recent_focus: str = Field(
        default="",
        description="1-2 sentences on current focus. e.g. 'Building Memeplex for Persona v1 release'",
    )

    # === TEMPORAL GROUNDING ===
    temporal_context: Optional[TemporalContext] = Field(default=None)

    # === MEMORY STATS ===
    memory_stats: MemoryStats = Field(default_factory=MemoryStats)

    # === TIMELINE ===
    timeline_summary: str = Field(
        default="",
        description="e.g. 'Active since Nov 2024, 47 sessions, 312 memories'",
    )

    def to_system_prompt(self) -> str:
        lines = ["## Your Knowledge of This User", ""]

        if self.topics:
            lines.append(f"**Topics**: {', '.join(self.topics)}")

        if self.people:
            lines.append(f"**People**: {', '.join(self.people)}")

        if self.projects:
            lines.append(f"**Projects**: {', '.join(self.projects)}")

        if self.places:
            lines.append(f"**Places**: {', '.join(self.places)}")

        if self.concepts:
            lines.append(f"**Concepts**: {', '.join(self.concepts)}")

        lines.append("")

        if self.last_week_topics:
            lines.append(f"**Last week**: {', '.join(self.last_week_topics)}")

        if self.last_month_topics:
            lines.append(f"**Last month**: {', '.join(self.last_month_topics)}")

        if self.recent_focus:
            lines.append(f"**Current focus**: {self.recent_focus}")

        if self.temporal_context:
            tc = self.temporal_context
            if tc.current_date:
                lines.append(f"**Today**: {tc.current_date}")
            if tc.upcoming:
                lines.append(f"**Upcoming**: {', '.join(tc.upcoming[:3])}")

        lines.append("")
        stats = self.memory_stats
        lines.append(
            f"*{stats.total_memories} memories | "
            f"{stats.total_entities} entities | {stats.active_notes} active notes*"
        )

        if self.timeline_summary:
            lines.append(f"*{self.timeline_summary}*")

        return "\n".join(lines)
