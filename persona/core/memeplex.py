"""Memeplex: The Memory Index Layer.

A memeplex (from memetics, Dawkins/Blackmore) is a group of memes that
reinforce and propagate together. In Persona, the Memeplex is the index
layer that clusters and routes queries to related memories.

The Memeplex provides fast, structured lookups into the memory graph
without full embedding search. It serves as a navigation layer - pointing
to where memories are, not storing the memories themselves.

The hippocampus in the brain doesn't store memories; it indexes them.
The Memeplex does the same for Persona's memory graph.

Three Index Dimensions:
- Entity Registry: WHO/WHAT - people, places, things, concepts
- Temporal Timeline: WHEN - time ranges, sequences, durations
- Topic Cluster: ABOUT - themes, categories, contexts

Usage:
    mplex = Memeplex(graph_ops, store)

    # Fast candidate resolution
    candidates = await mplex.resolve(
        entities=["Sarah", "project"],
        time_range=(date(2024, 12, 1), date(2024, 12, 31)),
    )

    # Then vector search only within candidates
    results = await vector_search(query, filter_ids=candidates)

This module defines the interface. Full implementation requires:
- Entity extraction during ingestion (Phase: Ingestion)
- Coreference resolution (Phase: Ingestion)
- Topic clustering (Phase: Consolidation)

Current status: Interface only. Implementation follows ingestion work.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional, Set, Dict, Any, Protocol
from uuid import UUID


@dataclass
class Entity:
    """An entity extracted from memories (person, place, thing, concept)."""

    id: UUID
    name: str
    entity_type: str  # person, place, organization, project, concept
    aliases: List[str] = field(default_factory=list)
    user_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    def matches(self, mention: str) -> bool:
        """Check if a mention refers to this entity."""
        mention_lower = mention.lower()
        if self.name.lower() == mention_lower:
            return True
        return any(alias.lower() == mention_lower for alias in self.aliases)


@dataclass
class EntityMention:
    """A mention of an entity in a specific memory."""

    entity_id: UUID
    memory_id: UUID
    mention_text: str  # The actual text that mentioned the entity
    context_snippet: str = ""  # Surrounding text for disambiguation


class EntityRegistry(Protocol):
    """Protocol for entity indexing and lookup."""

    async def register_entity(
        self,
        name: str,
        entity_type: str,
        user_id: str,
        aliases: Optional[List[str]] = None,
    ) -> Entity:
        """Create or get an entity."""
        ...

    async def link_to_memory(
        self,
        entity_id: UUID,
        memory_id: UUID,
        mention_text: str,
        context: str = "",
    ) -> None:
        """Link an entity mention to a memory."""
        ...

    async def resolve_mention(
        self,
        mention: str,
        user_id: str,
        context: Optional[str] = None,
    ) -> Optional[Entity]:
        """Resolve a mention to an entity (coreference resolution)."""
        ...

    async def memories_for_entity(
        self,
        entity_id: UUID,
        limit: int = 100,
    ) -> List[UUID]:
        """Get all memory IDs mentioning this entity."""
        ...

    async def search_entities(
        self,
        query: str,
        user_id: str,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Entity]:
        """Search entities by name/alias."""
        ...


class TemporalTimeline(Protocol):
    """Protocol for time-based memory indexing."""

    async def memories_in_range(
        self,
        user_id: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
        memory_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[UUID]:
        """Get memories with timestamps in the given range."""
        ...

    async def memories_before(
        self,
        anchor_id: UUID,
        limit: int = 10,
    ) -> List[UUID]:
        """Get memories that occurred before the anchor (via PREVIOUS chain)."""
        ...

    async def memories_after(
        self,
        anchor_id: UUID,
        limit: int = 10,
    ) -> List[UUID]:
        """Get memories that occurred after the anchor (via NEXT chain)."""
        ...

    async def order_chronologically(
        self,
        memory_ids: List[UUID],
    ) -> List[UUID]:
        """Sort memory IDs by timestamp, oldest first."""
        ...

    async def get_sequence(
        self,
        start_id: UUID,
        direction: str = "forward",  # "forward" or "backward"
        limit: int = 10,
    ) -> List[UUID]:
        """Follow NEXT/PREVIOUS chain from a starting point."""
        ...


class TopicCluster(Protocol):
    """Protocol for topic/theme-based memory grouping."""

    async def memories_for_topic(
        self,
        topic: str,
        user_id: str,
        limit: int = 100,
    ) -> List[UUID]:
        """Get memories tagged with or clustered into a topic."""
        ...

    async def topics_for_memory(
        self,
        memory_id: UUID,
    ) -> List[str]:
        """Get topics associated with a memory."""
        ...

    async def suggest_topics(
        self,
        user_id: str,
        limit: int = 20,
    ) -> List[str]:
        """Get the most common/recent topics for a user."""
        ...


@dataclass
class MemeplexQuery:
    """A structured query for the Memeplex index."""

    entities: List[str] = field(default_factory=list)
    time_start: Optional[date] = None
    time_end: Optional[date] = None
    topics: List[str] = field(default_factory=list)
    memory_types: List[str] = field(default_factory=list)  # episode, psyche, note

    @property
    def has_filters(self) -> bool:
        return bool(
            self.entities
            or self.time_start
            or self.time_end
            or self.topics
            or self.memory_types
        )


@dataclass
class MemeplexResult:
    """Result of a Loci index lookup."""

    candidate_ids: Set[UUID]
    entity_matches: Dict[str, List[UUID]] = field(default_factory=dict)
    time_matches: int = 0
    topic_matches: Dict[str, List[UUID]] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.candidate_ids)

    @property
    def is_empty(self) -> bool:
        return len(self.candidate_ids) == 0


class Memeplex(ABC):
    """
    The Memory Index Layer.

    Routes queries to memory candidates via structured lookups,
    reducing the search space for subsequent vector search.

    Implements intersection semantics: if multiple filters are
    provided, the result is their intersection (AND logic).
    """

    @abstractmethod
    async def resolve(
        self,
        query: MemeplexQuery,
        user_id: str,
    ) -> MemeplexResult:
        """
        Resolve a structured query to candidate memory IDs.

        This is the main entry point. Given filters (entities, time range,
        topics), returns the set of memory IDs that match ALL filters.
        """
        ...

    @abstractmethod
    async def resolve_simple(
        self,
        user_id: str,
        entities: Optional[List[str]] = None,
        time_start: Optional[date] = None,
        time_end: Optional[date] = None,
        topics: Optional[List[str]] = None,
        memory_types: Optional[List[str]] = None,
    ) -> Set[UUID]:
        """
        Simplified resolve that returns just the candidate IDs.

        Convenience method for common use cases.
        """
        ...

    @abstractmethod
    async def expand_from(
        self,
        memory_id: UUID,
        relationship_types: Optional[List[str]] = None,
        depth: int = 1,
    ) -> Set[UUID]:
        """
        Expand from a memory to find connected memories via graph traversal.

        This enables associative recall - one memory activating related ones.
        """
        ...


# =============================================================================
# Future: Concrete Implementation (after Ingestion phase)
# =============================================================================

# class Neo4jMemeplex(Memeplex):
#     """Neo4j-backed Memeplex implementation."""
#
#     def __init__(
#         self,
#         graph_ops: "GraphOps",
#         entity_registry: EntityRegistry,
#         timeline: TemporalTimeline,
#         topics: TopicCluster,
#     ):
#         self.graph_ops = graph_ops
#         self.entities = entity_registry
#         self.timeline = timeline
#         self.topics = topics
#
#     async def resolve(self, query: MemeplexQuery, user_id: str) -> MemeplexResult:
#         # Implementation follows after ingestion work
#         pass
