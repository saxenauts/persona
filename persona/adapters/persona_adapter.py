"""
PersonaAdapter: The unified ingestion interface for Persona.

This adapter handles the complete lifecycle of ingesting raw content into Persona:
1. Extraction: Converts raw text into typed Memory objects (Episode, Psyche, Goal).
2. Persistence: Saves memories to the graph database via MemoryStore.
3. Linking: Automatically chains episodes in temporal order.

Usage:
    async with PersonaAdapter(user_id, graph_ops) as adapter:
        result = await adapter.ingest("User said: I want to run a 10k...")
"""

from datetime import datetime
from typing import Optional
import time
from persona.core.graph_ops import GraphOps
from persona.core.memory_store import MemoryStore
from persona.services.ingestion_service import MemoryIngestionService, IngestionResult
from server.logging_config import get_logger

logger = get_logger(__name__)


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
        session_id: Optional[str] = None,
        persist: bool = True
    ) -> IngestionResult:
        """
        Complete ingestion: Extract + Persist + Link.
        
        Args:
            content: Raw text content (e.g., a conversation transcript).
            source_type: Descriptor for extraction hints ("conversation", "notes", etc.).
            timestamp: When this content was created. Defaults to now.
            session_id: Optional session identifier for grouping.
            persist: If False, only extract (for dry-run/preview).
        
        Returns:
            IngestionResult containing the extracted memories and links.
        """
        timestamp = timestamp or datetime.utcnow()
        session_id = session_id or f"session_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"PersonaAdapter.ingest: user={self.user_id}, source={source_type}, persist={persist}")

        start_total = time.time()

        # Step 1: Extract memories (in-memory)
        result = await self.ingestion_service.ingest(
            raw_content=content,
            user_id=self.user_id,
            session_id=session_id,
            timestamp=timestamp,
            source_type=source_type
        )
        
        if not result.success:
            logger.error(f"Extraction failed: {result.error}")
            return result
        
        logger.info(f"Extracted {len(result.memories)} memories, {len(result.links)} links")
        
        # Step 2: Persist (unless dry-run)
        persist_time_ms = 0.0
        if persist:
            persist_start = time.time()
            # Get previous episode BEFORE creating new ones (for temporal chain)
            previous_episode = await self.store.get_most_recent_episode(self.user_id)
            
            # Persist all memories
            for memory in result.memories:
                memory_links = [l for l in result.links if l.source_id == memory.id]
                await self.store.create(memory, links=memory_links)
            
            # Step 3: Link episodes in temporal chain
            episode = next((m for m in result.memories if m.type == "episode"), None)
            if episode and previous_episode and previous_episode.id != episode.id:
                await self.store.link_temporal_chain(episode, previous_episode)
                logger.info(f"Linked episode '{episode.title}' -> '{previous_episode.title}'")

            persist_time_ms = (time.time() - persist_start) * 1000

        result.persist_time_ms = persist_time_ms
        result.total_time_ms = (time.time() - start_total) * 1000
        
        return result
    
    async def ingest_batch(
        self,
        items: list[dict],
        persist: bool = True
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
                session_id = item.get("session_id") or f"session_{timestamp.strftime('%Y%m%d_%H%M%S')}_{idx}"
                
                logger.info(f"[Parallel] Extracting session {idx+1}/{len(items)}")
                
                result = await self.ingestion_service.ingest(
                    raw_content=item.get("content", ""),
                    user_id=self.user_id,
                    timestamp=timestamp,
                    session_id=session_id,
                    source_type=item.get("source_type", "conversation"),
                )
                
                return (idx, result, timestamp, session_id)
        
        # Phase 1: Parallel extraction (LLM + embeddings, no DB)
        logger.info(f"Phase 1: Parallel extraction of {len(items)} sessions (max_concurrent={max_concurrent})")
        tasks = [asyncio.create_task(extract_one(i, item)) for i, item in enumerate(items)]
        indexed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        for i, res in enumerate(indexed_results):
            if isinstance(res, Exception):
                logger.error(f"Session {i} extraction failed: {res}")
                indexed_results[i] = (i, IngestionResult(success=False, error=str(res)), None, None)
        
        # Phase 2: Sequential persist in correct order
        logger.info(f"Phase 2: Sequential persist of {len(items)} sessions")
        sorted_results = sorted(indexed_results, key=lambda x: x[0])
        final_results = []
        
        if persist:
            # Get previous episode BEFORE all new writes
            previous_episode = await self.store.get_most_recent_episode(self.user_id)
        
        for idx, result, timestamp, session_id in sorted_results:
            if not result.success:
                final_results.append(result)
                continue
            
            logger.info(f"Persisting session {idx+1}/{len(items)}: {len(result.memories)} memories")
            
            if persist:
                persist_start = time.time()
                await self.store.create_many(result.memories, result.links, self.user_id)
                
                # Link episode to previous in temporal chain
                episode = next((m for m in result.memories if m.type == "episode"), None)
                if episode and previous_episode and previous_episode.id != episode.id:
                    await self.store.link_temporal_chain(episode, previous_episode)
                    logger.debug(f"Linked episode '{episode.title}' -> '{previous_episode.title}'")
                
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

        if final_results:
            for res in final_results:
                if res.extract_time_ms is None:
                    res.extract_time_ms = 0.0
                if res.embed_time_ms is None:
                    res.embed_time_ms = 0.0
                if res.persist_time_ms is None:
                    res.persist_time_ms = 0.0
                if res.total_time_ms is None:
                    res.total_time_ms = (
                        (res.extract_time_ms or 0.0)
                        + (res.embed_time_ms or 0.0)
                        + (res.persist_time_ms or 0.0)
                    )
        
        logger.info(f"Batch ingestion complete: {len(final_results)} results")
        return final_results
