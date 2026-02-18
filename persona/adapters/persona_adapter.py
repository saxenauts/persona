"""
PersonaAdapter: The unified ingestion interface for Persona.

This adapter handles the complete lifecycle of ingesting raw content into Persona:
1. Extraction: Converts raw text into typed Memory objects (Episode, Psyche, Entity, Note).
2. Persistence: Saves memories to the graph database via MemoryStore.
3. Linking: Automatically chains episodes in temporal order.

Usage:
    async with PersonaAdapter(user_id, graph_ops) as adapter:
        result = await adapter.ingest("User said: I want to run a 10k...")
"""

from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional, List, Tuple, Any, Dict
from uuid import uuid4
import time
from persona.core.graph_ops import GraphOps
from persona.core.memory_store import MemoryStore
from persona.models.memory import EpisodeMemory, EntityMemory
from persona.services.ingestion_service import MemoryIngestionService, IngestionResult
from persona.services.integration_agent import run_integration_agent
from persona.utils.session import get_session_id
from server.logging_config import get_logger

logger = get_logger(__name__)

CROSS_SESSION_DEDUP_THRESHOLD = 0.9


def _normalize_name(name: str) -> str:
    return name.lower().strip().replace("-", " ").replace("_", " ")


def _fuzzy_match(str1: str, str2: str) -> float:
    return SequenceMatcher(None, str1, str2).ratio()


def _find_existing_entity_match(
    new_entity: EntityMemory,
    existing_entities: List[EntityMemory],
) -> Optional[EntityMemory]:
    new_name = _normalize_name(new_entity.canonical_name)

    for existing in existing_entities:
        existing_name = _normalize_name(existing.canonical_name)

        if new_name == existing_name:
            return existing

        if _fuzzy_match(new_name, existing_name) >= CROSS_SESSION_DEDUP_THRESHOLD:
            return existing

        for alias in existing.aliases or []:
            alias_norm = _normalize_name(alias)
            if new_name == alias_norm:
                return existing
            if _fuzzy_match(new_name, alias_norm) >= CROSS_SESSION_DEDUP_THRESHOLD:
                return existing

    return None


class PersonaAdapter:
    """
    The ONE interface for ingesting any data into Persona.

    Handles: Conversations, Apple Notes, Twitter, Instagram, etc.
    """

    def __init__(self, user_id: str, graph_ops: GraphOps):
        self.user_id = user_id
        self.graph_ops = graph_ops
        self.store = MemoryStore(graph_ops.graph_db, graph_ops.vector_store)
        self.ingestion_service = MemoryIngestionService()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass  # No cleanup needed; graph_ops is managed externally

    async def ingest(
        self,
        content: str,
        source_type: str = "conversation",
        timestamp: Optional[datetime] = None,
        timezone: str = "UTC",
        session_id: Optional[str] = None,
        store_transcript: bool = False,
        persist: bool = True,
        finalize_session: bool = False,
    ) -> IngestionResult:
        timestamp = timestamp or datetime.utcnow()
        session_id = session_id or get_session_id(source_type)

        logger.info(
            f"PersonaAdapter.ingest: user={self.user_id}, source={source_type}, session={session_id}, persist={persist}"
        )

        start_total = time.time()

        # Store transcript Episode if requested (before extraction)
        if store_transcript and persist:
            transcript_episode = EpisodeMemory(
                id=uuid4(),
                title=f"Transcript: {source_type}",
                content=content,
                event_time=timestamp,
                observed_at=datetime.utcnow(),
                day_id=timestamp.strftime("%Y-%m-%d"),
                session_id=session_id,
                source_type="transcript",
                user_id=self.user_id,
            )
            await self.store.create(transcript_episode, links=[])
            logger.info(f"Stored transcript Episode for session {session_id}")

        result = await self.ingestion_service.ingest(
            raw_content=content,
            user_id=self.user_id,
            session_id=session_id,
            timestamp=timestamp,
            timezone=timezone,
            source_type=source_type,
        )

        if not result.success:
            logger.error(f"Extraction failed: {result.error}")
            return result

        logger.info(
            f"Extracted {len(result.memories)} memories, {len(result.links)} links"
        )

        persist_time_ms = 0.0
        cross_session_dedup_metrics: Dict[str, Any] = {
            "entities_checked": 0,
            "entities_merged": 0,
            "entities_created": 0,
        }

        if persist:
            persist_start = time.time()

            existing_entities = await self.store.get_all_entities(self.user_id)
            episode_id = None

            for memory in result.memories:
                memory_links = [l for l in result.links if l.source_id == memory.id]

                if memory.type == "episode":
                    episode_id = memory.id
                    await self.store.create(memory, links=memory_links)

                elif memory.type == "entity":
                    entity_mem: EntityMemory = memory  # type: ignore
                    cross_session_dedup_metrics["entities_checked"] += 1

                    existing_match = _find_existing_entity_match(
                        entity_mem, existing_entities
                    )

                    if existing_match:
                        await self.store.merge_entity_attributes(
                            existing_match, entity_mem, evidence_id=episode_id
                        )
                        cross_session_dedup_metrics["entities_merged"] += 1
                    else:
                        await self.store.create_entity(entity_mem)
                        existing_entities.append(entity_mem)
                        cross_session_dedup_metrics["entities_created"] += 1
                else:
                    await self.store.create(memory, links=memory_links)

            episode = next((m for m in result.memories if m.type == "episode"), None)
            if episode and episode.event_time:
                predecessor = await self.store.get_temporal_predecessor(
                    self.user_id, episode.event_time
                )
                if predecessor and predecessor.id != episode.id:
                    await self.store.link_temporal_chain(episode, predecessor)
                    logger.info(
                        f"Linked episode '{episode.title}' -> '{predecessor.title}'"
                    )

            persist_time_ms = (time.time() - persist_start) * 1000

            if cross_session_dedup_metrics["entities_merged"] > 0:
                logger.info(
                    f"Cross-session entity dedup: {cross_session_dedup_metrics['entities_checked']} checked, "
                    f"{cross_session_dedup_metrics['entities_merged']} merged, "
                    f"{cross_session_dedup_metrics['entities_created']} created"
                )

        result.persist_time_ms = persist_time_ms
        result.total_time_ms = (time.time() - start_total) * 1000

        if finalize_session and persist and result.memories:
            await self.close_session(session_id)

        return result

    async def close_session(self, session_id: str) -> None:
        logger.info(f"Closing session {session_id} for user {self.user_id}")
        await run_integration_agent(
            user_id=self.user_id,
            trigger_ids=[],
            session_id=session_id,
            graph_ops=self.graph_ops,
        )

    async def ingest_batch(
        self, items: list[dict], persist: bool = True
    ) -> list[IngestionResult]:
        """
        Ingest multiple content items with PARALLEL extraction and SEQUENTIAL persist.

        Phase 1: Extract all sessions in parallel (LLM calls)
        Phase 2: Persist in order with correct temporal linking

        Args:
            items: List of dicts with keys: content, source_type, timestamp (optional).
            persist: If False, only extract.

        Returns:
            List of IngestionResult, one per item.
        """
        import asyncio
        import os

        max_concurrent = int(os.getenv("INGEST_SESSION_CONCURRENCY", "5"))
        sem = asyncio.Semaphore(max_concurrent)

        async def extract_one(idx: int, item: dict):
            """Extract memories for one session (no DB writes)."""
            async with sem:
                timestamp = item.get("timestamp") or datetime.utcnow()
                source_type = item.get("source_type", "conversation")
                session_id = item.get("session_id") or get_session_id(
                    source_type, f"batch_{timestamp.strftime('%Y%m%d_%H%M%S')}_{idx}"
                )

                logger.info(f"[Parallel] Extracting session {idx + 1}/{len(items)}")

                result = await self.ingestion_service.ingest(
                    raw_content=item.get("content", ""),
                    user_id=self.user_id,
                    timestamp=timestamp,
                    session_id=session_id,
                    source_type=item.get("source_type", "conversation"),
                )

                return (idx, result, timestamp, session_id)

        # Phase 1: Parallel extraction (LLM + embeddings, no DB)
        logger.info(
            f"Phase 1: Parallel extraction of {len(items)} sessions (max_concurrent={max_concurrent})"
        )
        tasks = [
            asyncio.create_task(extract_one(i, item)) for i, item in enumerate(items)
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        indexed_results: List[Tuple[int, IngestionResult, Any, Any]] = []
        for i, res in enumerate(raw_results):
            if isinstance(res, Exception):
                logger.error(f"Session {i} extraction failed: {res}")
                indexed_results.append(
                    (i, IngestionResult(success=False, error=str(res)), None, None)
                )
            else:
                indexed_results.append(res)  # type: ignore[arg-type]

        # Phase 2: Sequential persist in correct order
        logger.info(f"Phase 2: Sequential persist of {len(items)} sessions")
        sorted_results = sorted(indexed_results, key=lambda x: x[0])
        final_results = []

        previous_episode = None
        if persist and sorted_results:
            first_timestamp = sorted_results[0][2]
            if first_timestamp:
                previous_episode = await self.store.get_temporal_predecessor(
                    self.user_id, first_timestamp
                )

        for idx, result, timestamp, session_id in sorted_results:
            if not result.success:
                final_results.append(result)
                continue

            logger.info(
                f"Persisting session {idx + 1}/{len(items)}: {len(result.memories)} memories"
            )

            if persist:
                persist_start = time.time()

                existing_entities = await self.store.get_all_entities(self.user_id)
                episode_id = None

                non_entity_memories = []
                entity_memories = []
                for m in result.memories:
                    if m.type == "entity":
                        entity_memories.append(m)
                    else:
                        non_entity_memories.append(m)
                        if m.type == "episode":
                            episode_id = m.id

                non_entity_links = [
                    l
                    for l in result.links
                    if l.source_id not in {e.id for e in entity_memories}
                ]
                await self.store.create_many(
                    non_entity_memories, non_entity_links, self.user_id
                )

                for entity_mem in entity_memories:
                    existing_match = _find_existing_entity_match(
                        entity_mem,
                        existing_entities,  # type: ignore
                    )
                    if existing_match:
                        await self.store.merge_entity_attributes(
                            existing_match,
                            entity_mem,
                            evidence_id=episode_id,  # type: ignore
                        )
                    else:
                        await self.store.create_entity(entity_mem)  # type: ignore
                        existing_entities.append(entity_mem)  # type: ignore

                episode = next(
                    (m for m in result.memories if m.type == "episode"), None
                )
                if episode and previous_episode and previous_episode.id != episode.id:
                    await self.store.link_temporal_chain(episode, previous_episode)
                    logger.debug(
                        f"Linked episode '{episode.title}' -> '{previous_episode.title}'"
                    )

                if episode:
                    previous_episode = episode

                result.persist_time_ms = (time.time() - persist_start) * 1000
            else:
                result.persist_time_ms = 0.0

            if result.extract_time_ms is None:
                result.extract_time_ms = 0.0
            if result.embed_time_ms is None:
                result.embed_time_ms = 0.0
            result.total_time_ms = (
                (result.extract_time_ms or 0.0)
                + (result.embed_time_ms or 0.0)
                + (result.persist_time_ms or 0.0)
            )

            final_results.append(result)

        for res in final_results:
            res.extract_time_ms = res.extract_time_ms or 0.0
            res.embed_time_ms = res.embed_time_ms or 0.0
            res.persist_time_ms = res.persist_time_ms or 0.0
            if res.total_time_ms is None:
                res.total_time_ms = (
                    res.extract_time_ms + res.embed_time_ms + res.persist_time_ms
                )

        logger.info(f"Batch ingestion complete: {len(final_results)} results")

        return final_results
