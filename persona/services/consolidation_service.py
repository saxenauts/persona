"""Consolidation Service: Distill user identity from memories.

Minimal v1: Single LLM call to synthesize UserCard.identity_prose from recent memories.
No intermediate theme extraction - "current threads" are folded into the UserCard prompt.

Runs after integration to refresh the cached UserCard and Memeplex.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, List

from persona.core.graph_ops import GraphOps
from persona.core.memory_store import MemoryStore
from persona.models.memory import (
    UserCard,
    TemporalContext,
    Memory,
    Memeplex,
    MemoryStats,
    make_short_id,
)
from persona.llm.client_factory import get_chat_client
from persona.llm.providers.base import ChatMessage
from server.logging_config import get_logger

logger = get_logger(__name__)

USERCARD_SYSTEM_PROMPT = """Synthesize a 5-8 sentence identity summary from the user's memories.

Write in third person, present tense. Include:
- Who they are (role, context, background)
- What they do (hobbies, creative pursuits, tools they use)
- What matters to them (values, priorities, preferences)
- Current life threads (what they're actively working on or thinking about)

Preserve specific details: tool names, activity types, genres, techniques.
Be natural prose, not lists. Every detail might matter for future personalization."""


class UserCardService:
    def __init__(self, store: MemoryStore, graph_ops: Optional[GraphOps] = None):
        self.store = store
        self.graph_ops = graph_ops
        self.chat_client = get_chat_client()

    async def generate(
        self,
        user_id: str,
        timezone: Optional[str] = None,
    ) -> UserCard:
        psyche = await self.store.get_by_type("psyche", user_id, limit=20)
        notes = await self.store.get_by_type("note", user_id, limit=10)
        episodes = await self.store.get_by_type("episode", user_id, limit=15)

        active_notes = [
            n for n in notes if getattr(n, "status", "active") != "COMPLETED"
        ]

        if not psyche and not active_notes and not episodes:
            logger.info(f"No memories for user {user_id}, returning empty UserCard")
            return UserCard(user_id=user_id, timezone=timezone)

        try:
            identity_prose = await self._synthesize_prose(
                psyche, active_notes, episodes
            )
            return UserCard(
                user_id=user_id,
                timezone=timezone,
                identity_prose=identity_prose,
                updated_at=datetime.utcnow(),
                version=2,
            )
        except Exception as e:
            logger.warning(f"UserCard synthesis failed: {e}, returning fallback")
            return self._fallback_card(user_id, timezone, psyche, active_notes)

    async def _synthesize_prose(
        self,
        psyche: List[Memory],
        notes: List[Memory],
        episodes: List[Memory],
    ) -> str:
        memory_text = self._format_memories(psyche, notes, episodes)

        messages = [
            ChatMessage(role="system", content=USERCARD_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=f"Memories:\n{memory_text}\n\nWrite identity summary:",
            ),
        ]

        response = await self.chat_client.chat(messages)
        return (response.content or "").strip()

    def _format_memories(
        self,
        psyche: List[Memory],
        notes: List[Memory],
        episodes: List[Memory],
    ) -> str:
        lines = []

        for m in psyche[:15]:
            ptype = getattr(m, "psyche_type", "trait")
            lines.append(f"[{ptype}] {m.content}")

        for n in notes[:5]:
            ntype = getattr(n, "note_type", "task")
            lines.append(f"[{ntype}] {n.title}: {n.content}"[:200])

        for e in episodes[:8]:
            ts = e.event_time.strftime("%Y-%m-%d") if e.event_time else ""
            lines.append(f"[{ts}] {e.content}"[:200])

        return "\n".join(lines)

    def _fallback_card(
        self,
        user_id: str,
        timezone: Optional[str],
        psyche: List[Memory],
        notes: List[Memory],
    ) -> UserCard:
        parts = []

        traits = [
            m.content for m in psyche if getattr(m, "psyche_type", "") == "trait"
        ][:2]
        if traits:
            parts.append(f"Traits: {', '.join(traits)}.")

        focus = [n.title for n in notes][:2]
        if focus:
            parts.append(f"Current focus: {', '.join(focus)}.")

        return UserCard(
            user_id=user_id,
            timezone=timezone,
            identity_prose=" ".join(parts) if parts else "",
            updated_at=datetime.utcnow(),
            version=2,
        )


USERCARD_TTL_HOURS = 1.0


TEMPORAL_CONTEXT_PROMPT = """Summarize recent activity from these memories into temporal context.

Return JSON with:
- week_summary: 2-3 sentences on what's happening THIS WEEK (key events, themes, people)
- month_summary: 2-3 sentences on THIS MONTH's major themes
- upcoming: array of up to 3 upcoming events/deadlines mentioned

Be concise. Focus on what's most relevant for context.

Memories:
{memories}

Return only valid JSON."""


MEMEPLEX_REFRESH_PROMPT = """Build a memory index for this user's graph. This index helps you navigate and retrieve memories efficiently.

## CURRENT INDEX
{current_memeplex}

## MEMORIES (with short IDs)
{memories}

## ENTITIES (with short IDs)  
{entities}

## PSYCHE SIGNALS
{psyche}

## ACTIVE NOTES
{notes}

## YOUR TASK

Write a free-form index that captures:
1. **Key entities** with their short IDs - people [e:xxxx], places, projects, concepts
2. **Recent episodes** with IDs [ep:xxxx] - what's been happening
3. **Psyche signals** with IDs [p:xxxx] - traits, values, preferences  
4. **Active notes** with IDs [n:xxxx] - open tasks, goals, reminders
5. **Open threads** - ongoing narratives, unfinished business
6. **Rich memory areas** - topics with deep coverage

Include short IDs like [ep:a3f2] so you can reference specific memories later.
Format for YOUR use - what would help you recall and navigate this person's life?

Return JSON with single field:
{{"index": "your free-form index text here"}}

Keep it concise but comprehensive. This is your map of their memory graph."""


@dataclass
class ConsolidationResult:
    success: bool
    user_card_updated: bool = False
    temporal_context_updated: bool = False
    memeplex_updated: bool = False
    duration_ms: float = 0.0
    reason: str = ""
    errors: List[str] = field(default_factory=list)


async def run_consolidation(
    user_id: str,
    graph_ops: Optional[GraphOps] = None,
    force: bool = False,
) -> ConsolidationResult:
    """Distill user identity into cached UserCard and temporal context.

    Pipeline:
    1. Check if cached UserCard is fresh (skip if < TTL hours old)
    2. Generate new UserCard via single LLM call
    3. Generate temporal context (week/month summaries)
    4. Persist both to graph for caching
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

        temporal_ctx = await generate_temporal_context(user_id, graph_ops)
        if temporal_ctx and (temporal_ctx.week_summary or temporal_ctx.month_summary):
            await persist_temporal_context(graph_ops, user_id, temporal_ctx)
            result.temporal_context_updated = True
            logger.info(f"Updated temporal context for {user_id}")

        memeplex = await refresh_memeplex(user_id, graph_ops, store)
        if memeplex and memeplex.index:
            result.memeplex_updated = True
            logger.info(f"Updated memeplex for {user_id}: {len(memeplex.index)} chars")

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


async def persist_temporal_context(
    graph_ops: GraphOps, user_id: str, ctx: TemporalContext
) -> None:
    now = datetime.now(timezone.utc)
    await graph_ops.graph_db.create_nodes(
        [
            {
                "name": f"temporal_context_{user_id}",
                "type": "temporal_context",
                "current_date": ctx.current_date,
                "week_summary": ctx.week_summary,
                "week_start": ctx.week_start,
                "month_summary": ctx.month_summary,
                "month_name": ctx.month_name,
                "upcoming": json.dumps(ctx.upcoming),
                "updated_at": now.isoformat(),
            }
        ],
        user_id,
    )


async def get_cached_temporal_context(
    graph_ops: GraphOps, user_id: str
) -> Optional[TemporalContext]:
    try:
        nodes = await graph_ops.graph_db.get_all_nodes(user_id)
        for node in nodes:
            if node.get("type") == "temporal_context":
                upcoming_raw = node.get("upcoming", "[]")
                try:
                    upcoming = json.loads(upcoming_raw) if upcoming_raw else []
                except json.JSONDecodeError:
                    upcoming = []

                return TemporalContext(
                    current_date=node.get("current_date", ""),
                    week_summary=node.get("week_summary", ""),
                    week_start=node.get("week_start", ""),
                    month_summary=node.get("month_summary", ""),
                    month_name=node.get("month_name", ""),
                    upcoming=upcoming,
                )
    except Exception as e:
        logger.warning(f"Failed to read cached temporal context: {e}")
    return None


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


async def generate_temporal_context(
    user_id: str,
    graph_ops: GraphOps,
) -> Optional[TemporalContext]:
    """Generate week/month summaries from recent memories."""
    store = MemoryStore(graph_ops.graph_db)
    now = datetime.now(timezone.utc)

    week_start = now - timedelta(days=now.weekday())
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    episodes = await store.get_by_type("episode", user_id, limit=20)

    week_episodes = [
        e
        for e in episodes
        if e.event_time and e.event_time >= week_start.replace(tzinfo=None)
    ]
    month_episodes = [
        e
        for e in episodes
        if e.event_time and e.event_time >= month_start.replace(tzinfo=None)
    ]

    if not month_episodes:
        return TemporalContext(
            current_date=now.strftime("%A, %B %d, %Y"),
            week_start=week_start.strftime("%Y-%m-%d"),
            month_name=now.strftime("%B %Y"),
        )

    memory_text = "\n".join(
        [
            f"[{e.event_time.strftime('%Y-%m-%d') if e.event_time else ''}] {e.content[:150]}"
            for e in month_episodes[:15]
        ]
    )

    try:
        chat_client = get_chat_client()
        response = await chat_client.chat(
            messages=[
                ChatMessage(
                    role="user",
                    content=TEMPORAL_CONTEXT_PROMPT.format(memories=memory_text),
                )
            ],
            response_format={"type": "json_object"},
        )

        data = json.loads(response.content or "{}")

        return TemporalContext(
            current_date=now.strftime("%A, %B %d, %Y"),
            week_summary=data.get("week_summary", ""),
            week_start=week_start.strftime("%Y-%m-%d"),
            month_summary=data.get("month_summary", ""),
            month_name=now.strftime("%B %Y"),
            upcoming=data.get("upcoming", [])[:3],
        )
    except Exception as e:
        logger.warning(f"Temporal context generation failed: {e}")
        return TemporalContext(
            current_date=now.strftime("%A, %B %d, %Y"),
            week_start=week_start.strftime("%Y-%m-%d"),
            month_name=now.strftime("%B %Y"),
        )


async def refresh_memeplex(
    user_id: str,
    graph_ops: GraphOps,
    store: MemoryStore,
) -> Optional[Memeplex]:
    now = datetime.now(timezone.utc)
    month_ago = now - timedelta(days=30)

    current_memeplex = await store.get_memeplex(user_id)

    episodes = await store.get_by_type("episode", user_id, limit=30)
    entities = await store.get_by_type("entity", user_id, limit=50)
    psyche = await store.get_by_type("psyche", user_id, limit=20)
    notes = await store.get_by_type("note", user_id, limit=20)

    month_episodes = [
        e
        for e in episodes
        if e.event_time and e.event_time >= month_ago.replace(tzinfo=None)
    ]

    active_notes = [n for n in notes if getattr(n, "status", "active") != "COMPLETED"]

    if not month_episodes and not entities and not psyche:
        return current_memeplex

    memory_text = "\n".join(
        [
            f"[{make_short_id(e.id, 'episode')}] {e.event_time.strftime('%Y-%m-%d') if e.event_time else ''}: {e.title} - {e.content[:150]}"
            for e in month_episodes[:20]
        ]
    )

    entity_text = "\n".join(
        [
            f"[{make_short_id(e.id, 'entity')}] {getattr(e, 'canonical_name', 'Unknown')} ({getattr(e, 'entity_type', 'entity')}): {getattr(e, 'description', '')[:80]}"
            for e in entities[:30]
            if hasattr(e, "canonical_name")
        ]
    )

    psyche_text = "\n".join(
        [
            f"[{make_short_id(p.id, 'psyche')}] {getattr(p, 'psyche_type', 'trait')}: {p.content[:100]}"
            for p in psyche[:15]
        ]
    )

    notes_text = "\n".join(
        [
            f"[{make_short_id(n.id, 'note')}] {getattr(n, 'note_type', 'task')}: {n.title}"
            for n in active_notes[:10]
        ]
    )

    current_text = current_memeplex.index if current_memeplex else "None yet"

    try:
        chat_client = get_chat_client()
        response = await chat_client.chat(
            messages=[
                ChatMessage(
                    role="user",
                    content=MEMEPLEX_REFRESH_PROMPT.format(
                        current_memeplex=current_text,
                        memories=memory_text or "No recent memories",
                        entities=entity_text or "No entities yet",
                        psyche=psyche_text or "No psyche signals",
                        notes=notes_text or "No active notes",
                    ),
                )
            ],
            response_format={"type": "json_object"},
        )

        data = json.loads(response.content or "{}")

        stats = await store.compute_memory_stats(user_id)

        memeplex = Memeplex(
            user_id=user_id,
            updated_at=now,
            index=data.get("index", ""),
            memory_stats=stats,
        )

        await store.save_memeplex(memeplex)
        return memeplex

    except Exception as e:
        logger.warning(f"Memeplex refresh failed: {e}")
        return current_memeplex


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

    Used by PersonaService.
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
