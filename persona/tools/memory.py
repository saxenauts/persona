"""Memory Tools: Dialectic memory access for AI agents.

Handlers accept ToolContext (injected at runtime) + structured parameters
(provided by LLM). No natural language parsing - LLM provides dates/types directly.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from uuid import UUID

from persona.tools.context import ToolContext
from server.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryHit:
    id: str
    type: str
    title: str
    snippet: str
    event_time: str
    score: float = 0.0


@dataclass
class RecallResult:
    items: List[MemoryHit] = field(default_factory=list)
    count: int = 0


@dataclass
class RecordResult:
    stored: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ExpandResult:
    center_id: str
    neighbors: List[MemoryHit] = field(default_factory=list)
    relationships: List[Dict[str, str]] = field(default_factory=list)
    count: int = 0


def _memory_to_hit(memory: Any, score: float = 0.0) -> MemoryHit:
    title = getattr(memory, "title", "") or ""
    ts = getattr(memory, "event_time", None)
    timestamp_str = ts.isoformat() if ts else ""

    if memory.type == "entity":
        parts = [
            f"{getattr(memory, 'entity_type', 'entity')}: {getattr(memory, 'canonical_name', title)}"
        ]
        aliases = getattr(memory, "aliases", [])
        if aliases:
            parts.append(f"(aliases: {', '.join(aliases)})")
        desc = getattr(memory, "description", "")
        if desc:
            parts.append(desc)
        attrs = getattr(memory, "attributes", [])
        if attrs:
            attr_strs = [f"{a.key}: {a.value}" for a in attrs[:10]]
            parts.append("Facts: " + "; ".join(attr_strs))
        snippet = " | ".join(parts)
    else:
        content = getattr(memory, "content", "") or ""
        snippet = content[:200] + "..." if len(content) > 200 else content

    return MemoryHit(
        id=str(memory.id),
        type=memory.type,
        title=title,
        snippet=snippet,
        event_time=timestamp_str,
        score=score,
    )


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str).date()
    except ValueError:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            logger.warning(f"Invalid date format: {date_str}")
            return None


def _compute_recency_score(
    similarity: float, event_time: Optional[datetime], now: datetime
) -> float:
    """Return pure similarity score - let LLM reason about recency from timestamps."""
    return similarity


async def recall_handler(
    ctx: ToolContext,
    query: str,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    memory_types: Optional[List[str]] = None,
    limit: int = 10,
    exclude_transcripts: bool = True,
    order: Optional[str] = None,
) -> RecallResult:
    start_date = _parse_date(date_start)
    end_date = _parse_date(date_end)
    date_range = (start_date, end_date) if start_date or end_date else None

    try:
        results = await ctx.graph_ops.text_similarity_search(
            query=query,
            user_id=ctx.user_id,
            limit=limit * 3,
            date_range=date_range,
        )
        logger.info(
            f"Vector search returned {len(results.get('results', []))} results for query: {query[:50]}..."
        )
    except Exception as e:
        logger.error(f"Recall failed: {e}")
        return RecallResult()

    now = datetime.now()
    candidates = []

    for r in results.get("results", []):
        node_id = r.get("nodeName")
        similarity_score = r.get("score", 0.0)

        try:
            mem = await ctx.store.get(UUID(node_id), ctx.user_id)
            if mem:
                logger.info(
                    f"Processing memory {node_id}: type={mem.type}, title={getattr(mem, 'title', '?')}"
                )
                if (
                    exclude_transcripts
                    and getattr(mem, "source_type", None) == "transcript"
                ):
                    logger.info(f"Skipping transcript: {node_id}")
                    continue
                if memory_types and mem.type not in memory_types:
                    logger.info(f"Skipping type filter: {node_id} ({mem.type})")
                    continue

                event_time = getattr(mem, "event_time", None)
                recency_score = _compute_recency_score(
                    similarity_score, event_time, now
                )
                hit = _memory_to_hit(mem, score=recency_score)
                candidates.append((hit, event_time, recency_score))
            else:
                logger.warning(f"Memory not found: {node_id}")
        except Exception as e:
            logger.warning(f"Could not retrieve memory {node_id}: {e}")

    if order == "asc":
        candidates.sort(key=lambda x: x[1] or datetime.min)
    elif order == "desc":
        candidates.sort(key=lambda x: x[1] or datetime.min, reverse=True)
    else:
        candidates.sort(key=lambda x: x[2], reverse=True)

    candidates = candidates[:limit]
    items = [hit for hit, _, _ in candidates]

    return RecallResult(items=items, count=len(items))


async def record_handler(
    ctx: ToolContext,
    text: str,
) -> RecordResult:
    """Store new information to memory via full ingestion pipeline."""
    stored = []

    if not text.strip():
        return RecordResult()

    try:
        result = await ctx.adapter.ingest(
            content=text,
            source_type="tool",
            session_id=ctx.session_id,
            persist=True,
        )

        if result.success:
            for mem in result.memories:
                stored.append({"id": str(mem.id), "type": mem.type})
        else:
            logger.warning(f"Record failed: {result.error}")

    except Exception as e:
        logger.error(f"Record failed: {e}")

    return RecordResult(stored=stored)


async def expand_neighbors_handler(
    ctx: ToolContext,
    memory_id: str,
    relationship_types: Optional[List[str]] = None,
    limit: int = 10,
) -> ExpandResult:
    """Expand from a memory node to find connected memories."""
    try:
        source_uuid = UUID(memory_id)
    except ValueError:
        logger.warning(f"Invalid memory_id format: {memory_id}")
        return ExpandResult(center_id=memory_id)

    connections = await ctx.store.get_connected_batch([source_uuid], ctx.user_id)
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

    neighbor_tuples = neighbor_tuples[:limit]
    neighbor_ids = [target_id for target_id, _ in neighbor_tuples]
    neighbor_memories = await ctx.store.get_memories_by_ids(neighbor_ids, ctx.user_id)

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


async def follow_relationship_handler(
    ctx: ToolContext,
    source_id: str,
    relation_type: str,
    limit: int = 5,
) -> RecallResult:
    """Follow a specific relationship type from a memory."""
    try:
        source_uuid = UUID(source_id)
    except ValueError:
        logger.warning(f"Invalid source_id format: {source_id}")
        return RecallResult()

    connections = await ctx.store.get_connected_batch(
        [source_uuid], ctx.user_id, relation=relation_type
    )
    neighbor_tuples = connections.get(source_uuid, [])

    if not neighbor_tuples:
        return RecallResult()

    neighbor_tuples = neighbor_tuples[:limit]
    neighbor_ids = [target_id for target_id, _ in neighbor_tuples]
    neighbor_memories = await ctx.store.get_memories_by_ids(neighbor_ids, ctx.user_id)

    items = [_memory_to_hit(mem) for mem in neighbor_memories]
    return RecallResult(items=items, count=len(items))


@dataclass
class BrowseResult:
    items: List[MemoryHit] = field(default_factory=list)
    count: int = 0


@dataclass
class GetMemoryResult:
    found: bool = False
    memory: Optional[Dict[str, Any]] = None


@dataclass
class UpdateMemoryResult:
    success: bool = False
    memory_id: Optional[str] = None
    updated_fields: List[str] = field(default_factory=list)


async def browse_handler(
    ctx: ToolContext,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    memory_types: Optional[List[str]] = None,
    limit: int = 20,
    order: str = "desc",
) -> BrowseResult:
    """Time-ordered listing of memories. Unlike recall, results are sorted by event_time, not similarity."""
    start_date = _parse_date(date_start)
    end_date = _parse_date(date_end)

    all_memories = []
    types_to_fetch = memory_types or ["episode", "psyche", "note", "entity"]

    for mem_type in types_to_fetch:
        memories = await ctx.store.get_by_type(mem_type, ctx.user_id, limit=limit * 2)
        all_memories.extend(memories)

    if start_date:
        all_memories = [
            m
            for m in all_memories
            if m.event_time and m.event_time.date() >= start_date
        ]
    if end_date:
        all_memories = [
            m for m in all_memories if m.event_time and m.event_time.date() <= end_date
        ]

    reverse = order.lower() == "desc"
    all_memories.sort(key=lambda m: m.event_time or datetime.min, reverse=reverse)
    all_memories = all_memories[:limit]

    items = [_memory_to_hit(mem) for mem in all_memories]
    return BrowseResult(items=items, count=len(items))


async def get_memory_handler(
    ctx: ToolContext,
    memory_id: str,
) -> GetMemoryResult:
    """Fetch a single memory by ID. Returns full content, not just a snippet."""
    try:
        memory_uuid = UUID(memory_id)
    except ValueError:
        logger.warning(f"Invalid memory_id format: {memory_id}")
        return GetMemoryResult(found=False)

    memory = await ctx.store.get(memory_uuid, ctx.user_id)
    if not memory:
        return GetMemoryResult(found=False)

    memory_dict = {
        "id": str(memory.id),
        "type": memory.type,
        "title": getattr(memory, "title", ""),
        "content": getattr(memory, "content", ""),
        "event_time": memory.event_time.isoformat() if memory.event_time else None,
        "observed_at": memory.observed_at.isoformat() if memory.observed_at else None,
    }

    if memory.type == "note":
        memory_dict["status"] = getattr(memory, "status", "active")
        memory_dict["due_date"] = (
            getattr(memory, "due_date").isoformat()
            if getattr(memory, "due_date", None)
            else None
        )
        memory_dict["note_type"] = getattr(memory, "note_type", None)
    elif memory.type == "entity":
        memory_dict["entity_type"] = getattr(memory, "entity_type", None)
        memory_dict["canonical_name"] = getattr(memory, "canonical_name", "")
        memory_dict["aliases"] = getattr(memory, "aliases", [])
        memory_dict["description"] = getattr(memory, "description", "")
        attrs = getattr(memory, "attributes", [])
        memory_dict["attributes"] = [{"key": a.key, "value": a.value} for a in attrs]
    elif memory.type == "psyche":
        memory_dict["psyche_type"] = getattr(memory, "psyche_type", None)

    return GetMemoryResult(found=True, memory=memory_dict)


async def update_memory_handler(
    ctx: ToolContext,
    memory_id: str,
    updates: Dict[str, Any],
) -> UpdateMemoryResult:
    """Update fields on an existing memory. Returns success status and updated fields."""
    try:
        memory_uuid = UUID(memory_id)
    except ValueError:
        logger.warning(f"Invalid memory_id format: {memory_id}")
        return UpdateMemoryResult(success=False)

    memory = await ctx.store.get(memory_uuid, ctx.user_id)
    if not memory:
        logger.warning(f"Memory not found: {memory_id}")
        return UpdateMemoryResult(success=False)

    allowed_fields = {"title", "content", "status", "due_date", "importance"}
    filtered_updates = {k: v for k, v in updates.items() if k in allowed_fields}

    if not filtered_updates:
        return UpdateMemoryResult(success=False, memory_id=memory_id, updated_fields=[])

    try:
        await ctx.store.update(memory_uuid, ctx.user_id, filtered_updates)
        return UpdateMemoryResult(
            success=True,
            memory_id=memory_id,
            updated_fields=list(filtered_updates.keys()),
        )
    except Exception as e:
        logger.error(f"Update failed for {memory_id}: {e}")
        return UpdateMemoryResult(success=False, memory_id=memory_id)


TOOL_HANDLERS = {
    "recall": recall_handler,
    "record": record_handler,
    "expand_neighbors": expand_neighbors_handler,
    "follow_relationship": follow_relationship_handler,
    "browse": browse_handler,
    "get_memory": get_memory_handler,
    "update_memory": update_memory_handler,
}
