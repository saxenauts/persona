"""Consolidation Service: Distill user identity from memories.

Minimal v1: Single LLM call to synthesize UserCard.identity_prose from recent memories.
No intermediate theme extraction - "current threads" are folded into the UserCard prompt.

Runs after integration to refresh the cached UserCard.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List

from persona.core.graph_ops import GraphOps
from persona.core.memory_store import MemoryStore
from persona.models.memory import UserCard
from persona.services.user_service import UserCardService
from server.logging_config import get_logger

logger = get_logger(__name__)

USERCARD_TTL_HOURS = 1.0


@dataclass
class ConsolidationResult:
    success: bool
    user_card_updated: bool = False
    duration_ms: float = 0.0
    reason: str = ""
    errors: List[str] = field(default_factory=list)


async def run_consolidation(
    user_id: str,
    graph_ops: Optional[GraphOps] = None,
    force: bool = False,
) -> ConsolidationResult:
    """Distill user identity into cached UserCard.

    Pipeline:
    1. Check if cached UserCard is fresh (skip if < TTL hours old)
    2. Generate new UserCard via single LLM call
    3. Persist to graph for caching

    Args:
        user_id: User to consolidate
        graph_ops: Optional GraphOps (creates one if not provided)
        force: Force refresh even if recently updated
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

        if not force:
            existing_card = await get_cached_usercard(graph_ops, user_id)
            if existing_card:
                age_hours = _card_age_hours(existing_card)
                if age_hours < USERCARD_TTL_HOURS:
                    result.reason = (
                        f"cached (age={age_hours:.1f}h < {USERCARD_TTL_HOURS}h)"
                    )
                    result.duration_ms = (time.time() - start_time) * 1000
                    logger.info(f"Consolidation skipped for {user_id}: {result.reason}")
                    return result

        new_card = await user_card_service.generate(user_id)

        if new_card.identity_prose:
            await persist_usercard(graph_ops, user_id, new_card)
            result.user_card_updated = True
            result.reason = "generated"
            logger.info(
                f"Consolidated UserCard for {user_id}: {new_card.identity_prose[:50]}..."
            )
        else:
            result.reason = "no_memories"

        result.duration_ms = (time.time() - start_time) * 1000

    except Exception as e:
        logger.error(f"Consolidation failed for {user_id}: {e}")
        result.success = False
        result.errors.append(str(e))

    finally:
        if own_graph_ops and graph_ops:
            await graph_ops.__aexit__(None, None, None)

    return result


async def get_cached_usercard(graph_ops: GraphOps, user_id: str) -> Optional[UserCard]:
    """Read cached UserCard from graph."""
    try:
        nodes = await graph_ops.graph_db.get_all_nodes(user_id)
        for node in nodes:
            if node.get("type") == "usercard":
                updated_at_str = node.get("updated_at", "")
                if updated_at_str:
                    if updated_at_str.endswith("Z"):
                        updated_at_str = updated_at_str[:-1] + "+00:00"
                    updated_at = datetime.fromisoformat(updated_at_str)
                else:
                    updated_at = datetime.now(timezone.utc)

                return UserCard(
                    user_id=user_id,
                    identity_prose=node.get("identity_prose", ""),
                    timezone=node.get("timezone"),
                    updated_at=updated_at,
                )
    except Exception as e:
        logger.warning(f"Failed to read cached UserCard: {e}")
    return None


async def persist_usercard(graph_ops: GraphOps, user_id: str, card: UserCard) -> None:
    """Write UserCard to graph for caching."""
    now = datetime.now(timezone.utc)
    await graph_ops.graph_db.create_nodes(
        [
            {
                "name": f"usercard_{user_id}",
                "type": "usercard",
                "identity_prose": card.identity_prose,
                "updated_at": now.isoformat(),
                "timezone": card.timezone or "UTC",
            }
        ],
        user_id,
    )


def _card_age_hours(card: UserCard) -> float:
    """Calculate age of UserCard in hours."""
    if not card.updated_at:
        return float("inf")

    now = datetime.now(timezone.utc)
    card_time = card.updated_at
    if card_time.tzinfo is None:
        card_time = card_time.replace(tzinfo=timezone.utc)

    delta = now - card_time
    return delta.total_seconds() / 3600


async def maybe_run_consolidation(
    user_id: str,
    graph_ops: Optional[GraphOps] = None,
) -> Optional[ConsolidationResult]:
    """Trigger consolidation after integration."""
    logger.info(f"Triggering consolidation for {user_id}")
    return await run_consolidation(user_id, graph_ops)


async def get_or_generate_usercard(
    user_id: str,
    graph_ops: GraphOps,
    user_timezone: str = "UTC",
    ttl_hours: float = USERCARD_TTL_HOURS,
) -> Optional[UserCard]:
    """Single entry point for UserCard access.

    1. Try to read cached UserCard from graph
    2. If fresh enough, return it
    3. Otherwise, generate new one, cache it, return it

    Used by PersonaService and RAGInterface.
    """
    cached = await get_cached_usercard(graph_ops, user_id)

    if cached:
        age = _card_age_hours(cached)
        if age < ttl_hours:
            logger.debug(f"Using cached UserCard for {user_id} (age={age:.1f}h)")
            return cached

    store = MemoryStore(graph_ops.graph_db)
    service = UserCardService(store, graph_ops)

    try:
        new_card = await service.generate(user_id, timezone=user_timezone)
        if new_card.identity_prose:
            await persist_usercard(graph_ops, user_id, new_card)
            logger.info(f"Generated and cached UserCard for {user_id}")
        return new_card
    except Exception as e:
        logger.warning(f"UserCard generation failed: {e}")
        return cached
