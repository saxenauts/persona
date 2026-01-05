"""Consolidation Service: Periodic synthesis of user identity from memories.

Runs after integration to:
- Refresh UserCard.identity_prose from recent Psyche/Notes/Episodes
- Persist updated UserCard to graph for caching
- Update entity descriptions from accumulated mentions

This is the "Distiller" - synthesizing raw memories into refined knowledge.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

from persona.core.graph_ops import GraphOps
from persona.core.memory_store import MemoryStore
from persona.models.memory import UserCard
from persona.services.user_service import UserCardService
from server.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ConsolidationResult:
    success: bool
    user_card_updated: bool = False
    entities_updated: int = 0
    duration_ms: float = 0.0
    errors: List[str] = field(default_factory=list)


async def run_consolidation(
    user_id: str,
    graph_ops: Optional[GraphOps] = None,
    force: bool = False,
) -> ConsolidationResult:
    """Run consolidation to refresh user identity synthesis.

    Args:
        user_id: User to consolidate
        graph_ops: Optional GraphOps (creates one if not provided)
        force: Force refresh even if recently updated

    Returns:
        ConsolidationResult with statistics
    """
    start_time = time.time()
    result = ConsolidationResult(success=True)

    own_graph_ops = graph_ops is None
    if own_graph_ops:
        graph_ops = GraphOps()
        await graph_ops.__aenter__()

    try:
        store = MemoryStore(graph_ops.graph_db)
        user_card_service = UserCardService(store, graph_ops)

        existing_card = await _get_existing_usercard(graph_ops, user_id)

        if not force and existing_card:
            age_hours = _card_age_hours(existing_card)
            if age_hours < 1.0:
                logger.info(
                    f"UserCard for {user_id} is recent ({age_hours:.1f}h), skipping"
                )
                result.duration_ms = (time.time() - start_time) * 1000
                return result

        new_card = await user_card_service.generate(user_id)

        if new_card.identity_prose:
            await _persist_usercard(graph_ops, user_id, new_card)
            result.user_card_updated = True
            logger.info(
                f"Updated UserCard for {user_id}: {new_card.identity_prose[:50]}..."
            )

        result.duration_ms = (time.time() - start_time) * 1000

    except Exception as e:
        logger.error(f"Consolidation failed for {user_id}: {e}")
        result.success = False
        result.errors.append(str(e))

    finally:
        if own_graph_ops and graph_ops:
            await graph_ops.__aexit__(None, None, None)

    return result


async def _get_existing_usercard(
    graph_ops: GraphOps, user_id: str
) -> Optional[UserCard]:
    """Fetch existing UserCard from graph if present."""
    try:
        nodes = await graph_ops.graph_db.get_all_nodes(user_id)
        for node in nodes:
            if node.get("type") == "usercard":
                return UserCard(
                    user_id=user_id,
                    identity_prose=node.get("identity_prose", ""),
                    updated_at=datetime.fromisoformat(node["updated_at"])
                    if node.get("updated_at")
                    else None,
                )
    except Exception as e:
        logger.warning(f"Failed to fetch existing UserCard: {e}")
    return None


async def _persist_usercard(graph_ops: GraphOps, user_id: str, card: UserCard) -> None:
    """Persist UserCard to graph for caching."""
    await graph_ops.graph_db.create_nodes(
        [
            {
                "name": f"usercard_{user_id}",
                "type": "usercard",
                "identity_prose": card.identity_prose,
                "updated_at": datetime.utcnow().isoformat(),
                "timezone": card.timezone or "UTC",
            }
        ],
        user_id,
    )


def _card_age_hours(card: UserCard) -> float:
    """Calculate age of UserCard in hours."""
    if not card.updated_at:
        return float("inf")
    delta = datetime.utcnow() - card.updated_at
    return delta.total_seconds() / 3600


async def maybe_run_consolidation(
    user_id: str,
    graph_ops: Optional[GraphOps] = None,
    min_memories_changed: int = 3,
) -> Optional[ConsolidationResult]:
    """Conditionally run consolidation based on change threshold.

    Call this after integration. Only runs if enough memories were processed.

    Args:
        user_id: User to consolidate
        graph_ops: Optional GraphOps to reuse
        min_memories_changed: Minimum memories changed before consolidating

    Returns:
        ConsolidationResult if run, None if skipped
    """
    logger.info(f"Triggering consolidation for {user_id}")
    return await run_consolidation(user_id, graph_ops)
