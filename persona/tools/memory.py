"""Memory Tools: Dialectic memory access for AI agents."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID
import re

from persona.core.graph_ops import GraphOps
from persona.core.memory_store import MemoryStore
from persona.services.ingestion_service import MemoryIngestionService
from server.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryHit:
    id: str
    type: str
    title: str
    snippet: str
    timestamp: str
    score: float = 0.0


@dataclass
class RecallResult:
    items: List[MemoryHit] = field(default_factory=list)
    count: int = 0


@dataclass
class RecordResult:
    stored: List[Dict[str, str]] = field(default_factory=list)


def _parse_time_cues(query: str) -> Optional[tuple[datetime, datetime]]:
    query_lower = query.lower()
    now = datetime.utcnow()

    if "today" in query_lower:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, now
    if "yesterday" in query_lower:
        start = (now - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, end
    if "last week" in query_lower or "this week" in query_lower:
        return now - timedelta(days=7), now
    if "last month" in query_lower or "this month" in query_lower:
        return now - timedelta(days=30), now

    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", query)
    if date_match:
        try:
            date = datetime.fromisoformat(date_match.group(1))
            return date, date + timedelta(days=1)
        except ValueError:
            pass

    return None


def _parse_type_cues(query: str) -> Optional[List[str]]:
    query_lower = query.lower()

    if any(
        w in query_lower
        for w in ["preference", "like", "dislike", "personality", "trait", "value"]
    ):
        return ["psyche"]
    if any(
        w in query_lower for w in ["task", "todo", "note", "list", "goal", "remind"]
    ):
        return ["note"]
    if any(
        w in query_lower
        for w in ["happened", "did", "went", "event", "meeting", "conversation"]
    ):
        return ["episode"]

    return None


def _memory_to_hit(memory: Any, score: float = 0.0) -> MemoryHit:
    content = getattr(memory, "content", "") or ""
    title = getattr(memory, "title", "") or ""
    snippet = content[:200] + "..." if len(content) > 200 else content

    ts = getattr(memory, "timestamp", None)
    timestamp_str = ts.isoformat() if ts else ""

    return MemoryHit(
        id=str(memory.id),
        type=memory.type,
        title=title,
        snippet=snippet,
        timestamp=timestamp_str,
        score=score,
    )


async def recall(
    user_id: str,
    query: str,
    graph_ops: GraphOps,
    store: MemoryStore,
) -> RecallResult:
    time_range = _parse_time_cues(query)
    type_filter = _parse_type_cues(query)

    date_range = None
    if time_range:
        date_range = (time_range[0].date(), time_range[1].date())

    try:
        results = await graph_ops.text_similarity_search(
            query=query,
            user_id=user_id,
            limit=15,
            date_range=date_range,
        )
    except Exception as e:
        logger.error(f"Recall failed: {e}")
        return RecallResult()

    items = []

    for r in results.get("results", []):
        node_id = r.get("nodeName")
        score = r.get("score", 0.0)

        try:
            mem = await store.get(UUID(node_id), user_id)
            if mem:
                if type_filter and mem.type not in type_filter:
                    continue
                items.append(_memory_to_hit(mem, score=score))
                if len(items) >= 10:
                    break
        except Exception as e:
            logger.debug(f"Could not retrieve memory {node_id}: {e}")

    return RecallResult(items=items, count=len(items))


@dataclass
class ExpandResult:
    """Result from graph expansion."""

    center_id: str
    neighbors: List[MemoryHit] = field(default_factory=list)
    relationships: List[Dict[str, str]] = field(default_factory=list)
    count: int = 0


async def expand_neighbors(
    memory_id: str,
    user_id: str,
    store: MemoryStore,
    relationship_types: Optional[List[str]] = None,
    max_depth: int = 1,
) -> ExpandResult:
    """
    Expand from a memory node to find connected memories.

    Use after recall() to explore interesting connections from a specific memory.
    Returns the memory's neighbors via graph relationships.

    Args:
        memory_id: UUID of the memory to expand from
        user_id: User ID for access control
        store: MemoryStore instance
        relationship_types: Filter by relationship types (e.g., ["LED_TO", "CAUSED_BY"]).
                           If None, returns all relationships.
        max_depth: How many hops to traverse (currently supports 1)

    Returns:
        ExpandResult with neighbors and relationship info
    """
    try:
        source_uuid = UUID(memory_id)
    except ValueError:
        logger.warning(f"Invalid memory_id format: {memory_id}")
        return ExpandResult(center_id=memory_id)

    connections = await store.get_connected_batch([source_uuid], user_id)
    neighbor_tuples = connections.get(source_uuid, [])

    if relationship_types:
        relationship_types_lower = [r.lower() for r in relationship_types]
        neighbor_tuples = [
            (target_id, rel_type)
            for target_id, rel_type in neighbor_tuples
            if rel_type.lower() in relationship_types_lower
        ]

    if not neighbor_tuples:
        return ExpandResult(center_id=memory_id)

    neighbor_ids = [target_id for target_id, _ in neighbor_tuples]
    neighbor_memories = await store.get_memories_by_ids(neighbor_ids, user_id)

    neighbors = []
    relationships = []
    id_to_memory = {mem.id: mem for mem in neighbor_memories}

    for target_id, rel_type in neighbor_tuples:
        mem = id_to_memory.get(target_id)
        if mem:
            neighbors.append(_memory_to_hit(mem))
            relationships.append(
                {
                    "source": memory_id,
                    "target": str(target_id),
                    "relation": rel_type,
                }
            )

    return ExpandResult(
        center_id=memory_id,
        neighbors=neighbors,
        relationships=relationships,
        count=len(neighbors),
    )


async def follow_relationship(
    source_id: str,
    relation_type: str,
    user_id: str,
    store: MemoryStore,
    limit: int = 5,
) -> RecallResult:
    """
    Follow a specific relationship type from a memory.

    Use to trace causal chains or thematic connections.
    More targeted than expand_neighbors() - focuses on one relationship type.

    Args:
        source_id: UUID of the starting memory
        relation_type: The relationship type to follow (e.g., "LED_TO", "CAUSED_BY", "NEXT")
        user_id: User ID for access control
        store: MemoryStore instance
        limit: Maximum number of connected memories to return

    Returns:
        RecallResult with memories connected by this relationship
    """
    try:
        source_uuid = UUID(source_id)
    except ValueError:
        logger.warning(f"Invalid source_id format: {source_id}")
        return RecallResult()

    connections = await store.get_connected_batch(
        [source_uuid], user_id, relation=relation_type
    )
    neighbor_tuples = connections.get(source_uuid, [])

    if not neighbor_tuples:
        return RecallResult()

    neighbor_tuples = neighbor_tuples[:limit]
    neighbor_ids = [target_id for target_id, _ in neighbor_tuples]
    neighbor_memories = await store.get_memories_by_ids(neighbor_ids, user_id)

    items = [_memory_to_hit(mem) for mem in neighbor_memories]

    return RecallResult(items=items, count=len(items))


async def record(
    user_id: str,
    text: str,
    session_id: Optional[str] = None,
) -> RecordResult:
    ingestion_service = MemoryIngestionService()
    stored = []

    if not text.strip():
        return RecordResult()

    try:
        result = await ingestion_service.ingest(
            raw_content=text,
            user_id=user_id,
            session_id=session_id,
            source_type="tool",
            source_ref="record",
        )

        if result.success:
            for mem in result.memories:
                stored.append(
                    {
                        "id": str(mem.id),
                        "type": mem.type,
                    }
                )
        else:
            logger.warning(f"Record failed: {result.error}")

    except Exception as e:
        logger.error(f"Record failed: {e}")

    return RecordResult(stored=stored)
