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
        self.store = MemoryStore(graph_ops.graph_db)
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
        if persist:
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
        
        return result
    
    async def ingest_batch(
        self,
        items: list[dict],
        persist: bool = True
    ) -> list[IngestionResult]:
        """
        Ingest multiple content items sequentially with temporal linking.
        
        Args:
            items: List of dicts with keys: content, source_type, timestamp (optional).
            persist: If False, only extract.
        
        Returns:
            List of IngestionResult, one per item.
        """
        results = []
        for i, item in enumerate(items):
            logger.info(f"Ingesting item {i+1}/{len(items)}")
            result = await self.ingest(
                content=item.get("content", ""),
                source_type=item.get("source_type", "conversation"),
                timestamp=item.get("timestamp"),
                session_id=item.get("session_id"),
                persist=persist
            )
            results.append(result)
        return results
