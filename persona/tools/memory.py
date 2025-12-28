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


def _should_expand(query: str) -> bool:
    query_lower = query.lower()
    return any(
        w in query_lower
        for w in [
            "related",
            "connected",
            "everything about",
            "all about",
            "what else",
            "more about",
            "associated with",
        ]
    )


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
    should_expand = _should_expand(query)

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
    seed_ids = []

    for r in results.get("results", []):
        node_id = r.get("nodeName")
        score = r.get("score", 0.0)

        try:
            mem = await store.get(UUID(node_id), user_id)
            if mem:
                if type_filter and mem.type not in type_filter:
                    continue
                items.append(_memory_to_hit(mem, score=score))
                seed_ids.append(mem.id)
                if len(items) >= 10:
                    break
        except Exception as e:
            logger.debug(f"Could not retrieve memory {node_id}: {e}")

    if should_expand and seed_ids:
        visited = set(seed_ids)
        for seed_id in seed_ids[:3]:
            try:
                connected = await store.get_connected(seed_id, user_id)
                for cm in connected:
                    if cm.id not in visited and len(items) < 15:
                        visited.add(cm.id)
                        items.append(_memory_to_hit(cm))
            except Exception as e:
                logger.debug(f"Failed to expand {seed_id}: {e}")

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
