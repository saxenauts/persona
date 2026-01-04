"""Retrieval Layer: Time-windowed working memory with prose formatting.

No query expansion. No vector search for base context.
Just time-windowed fetch of recent memories with links, formatted as prose.
"""

from datetime import datetime
from typing import Optional, Dict, Any, Sequence
from uuid import UUID

from persona.core.graph_ops import GraphOps
from persona.core.memory_store import MemoryStore
from persona.core.context import format_working_memory_prose
from persona.models.memory import (
    Memory,
    MemoryLink,
    EpisodeMemory,
    PsycheMemory,
    NoteMemory,
    UserCard,
    WorkingMemoryConfig,
    DEFAULT_WORKING_MEMORY_CONFIG,
)
from server.logging_config import get_logger

logger = get_logger(__name__)


class Retriever:
    """Retrieves working memory for LLM via time-windowed fetch + prose formatting."""

    def __init__(self, user_id: str, store: MemoryStore, graph_ops: GraphOps):
        self.user_id = user_id
        self.store = store
        self.graph_ops = graph_ops

    async def get_working_memory(
        self,
        config: Optional[WorkingMemoryConfig] = None,
        user_card: Optional[UserCard] = None,
        user_timezone: str = "UTC",
        collect_stats: bool = False,
        **kwargs,
    ) -> str | tuple[str, Dict[str, Any]]:
        """Get prose-formatted working memory for a dialog turn."""
        cfg = config or DEFAULT_WORKING_MEMORY_CONFIG
        now = datetime.utcnow()
        stats: Dict[str, Any] = {}

        episodes = await self._get_recent_episodes(now, cfg)
        psyche = await self._get_recent_psyche(now, cfg)
        active_notes = await self._get_active_notes(cfg)

        all_memories: Sequence[Memory] = [*episodes, *psyche, *active_notes]
        memory_ids = [m.id for m in all_memories]
        links = await self._get_links_for_memories(memory_ids)

        working_memory = format_working_memory_prose(
            user_card=user_card,
            episodes=episodes,
            psyche=psyche,
            active_notes=active_notes,
            links=links,
        )

        if collect_stats:
            stats = {
                "episode_count": len(episodes),
                "psyche_count": len(psyche),
                "note_count": len(active_notes),
                "link_count": len(links),
                "working_memory_chars": len(working_memory),
                "config": {
                    "episode_window_days": cfg.episode_window.days,
                    "psyche_window_days": cfg.psyche_window.days,
                },
            }
            logger.info(f"Retriever stats: {stats}")
            return working_memory, stats

        logger.info(
            f"Retriever: {len(all_memories)} memories, {len(working_memory)} chars"
        )
        return working_memory

    async def get_working_memory_with_stats(
        self,
        config: Optional[WorkingMemoryConfig] = None,
        user_card: Optional[UserCard] = None,
        user_timezone: str = "UTC",
    ) -> tuple[str, Dict[str, Any]]:
        result = await self.get_working_memory(
            config=config,
            user_card=user_card,
            user_timezone=user_timezone,
            collect_stats=True,
        )
        return result  # type: ignore

    async def _get_recent_episodes(
        self, now: datetime, cfg: WorkingMemoryConfig
    ) -> list[EpisodeMemory]:
        since = now - cfg.episode_window
        try:
            memories = await self.store.get_by_type(
                "episode", self.user_id, limit=cfg.max_episodes
            )
            recent = [m for m in memories if m.event_time and m.event_time >= since]
            recent.sort(key=lambda m: m.event_time, reverse=True)
            return [
                m for m in recent[: cfg.max_episodes] if isinstance(m, EpisodeMemory)
            ]
        except Exception as e:
            logger.warning(f"Failed to get episodes: {e}")
            return []

    async def _get_recent_psyche(
        self, now: datetime, cfg: WorkingMemoryConfig
    ) -> list[PsycheMemory]:
        since = now - cfg.psyche_window
        try:
            memories = await self.store.get_by_type(
                "psyche", self.user_id, limit=cfg.max_psyche
            )
            recent = [m for m in memories if m.event_time and m.event_time >= since]
            recent.sort(key=lambda m: m.event_time, reverse=True)
            return [m for m in recent[: cfg.max_psyche] if isinstance(m, PsycheMemory)]
        except Exception as e:
            logger.warning(f"Failed to get psyche: {e}")
            return []

    async def _get_active_notes(self, cfg: WorkingMemoryConfig) -> list[NoteMemory]:
        try:
            notes = await self.store.get_by_type(
                "note", self.user_id, limit=cfg.max_active_notes * 2
            )
            active = [
                n
                for n in notes
                if getattr(n, "status", "active").lower() != "completed"
            ]
            active.sort(key=lambda n: n.event_time, reverse=True)
            return [
                n for n in active[: cfg.max_active_notes] if isinstance(n, NoteMemory)
            ]
        except Exception as e:
            logger.warning(f"Failed to get notes: {e}")
            return []

    async def _get_links_for_memories(self, memory_ids: list[UUID]) -> list[MemoryLink]:
        if not memory_ids:
            return []
        try:
            connections = await self.store.get_connected_batch(memory_ids, self.user_id)
            all_links = []
            for source_id, targets in connections.items():
                for target_id, relation in targets:
                    all_links.append(
                        MemoryLink(
                            source_id=source_id,
                            target_id=target_id,
                            relation=relation,
                        )
                    )
            return all_links
        except Exception as e:
            logger.warning(f"Failed to get links: {e}")
            return []
