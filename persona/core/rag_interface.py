"""
RAG Interface for Persona.

Provides context retrieval and query answering using the Memory architecture.
"""

from typing import List, Dict, Any, Optional
import time
from persona.core.graph_ops import GraphOps
from persona.core.retrieval import Retriever
from persona.core.memory_store import MemoryStore
from persona.core.backends.neo4j_graph import Neo4jGraphDatabase
from persona.core.context import format_memories_for_llm
from persona.llm.llm_graph import (
    generate_response_with_context,
    generate_response_with_context_with_stats
)
from server.logging_config import get_logger

logger = get_logger(__name__)


class RAGInterface:
    """
    Retrieval-Augmented Generation interface for Persona.
    
    Uses the Retriever (Vector Search + Graph Crawl) for context retrieval.
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.graph_ops = None
        self._memory_store = None
        self._retriever = None
        self._graph_db = None
    
    async def __aenter__(self):
        """Initialize resources."""
        self.graph_ops = await GraphOps().__aenter__()
        
        # Initialize memory store
        self._graph_db = Neo4jGraphDatabase()
        await self._graph_db.initialize()
        self._memory_store = MemoryStore(self._graph_db)
        
        # Initialize retriever
        self._retriever = Retriever(
            user_id=self.user_id,
            store=self._memory_store,
            graph_ops=self.graph_ops
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources."""
        if self.graph_ops:
            await self.graph_ops.__aexit__(exc_type, exc_val, exc_tb)
        if self._graph_db:
            await self._graph_db.close()
    
    async def get_context(
        self, 
        query: str, 
        top_k: int = 5, 
        hop_depth: int = 1,
        include_static: bool = True
    ) -> str:
        """
        Get formatted context for a query.
        
        Uses the new Retriever with Vector Search + Graph Crawl.
        
        Args:
            query: Natural language query.
            top_k: Number of vector search results.
            hop_depth: How many relationship hops to crawl.
            include_static: Whether to include active goals and psyche.
        
        Returns:
            XML-formatted context string.
        """
        if not self._retriever:
            await self.__aenter__()
        
        context = await self._retriever.get_context(
            query=query,
            top_k=top_k,
            hop_depth=hop_depth,
            include_static=include_static
        )
        
        logger.info(f"RAGInterface: got {len(context)} chars context for query: {query[:50]}...")
        return context
    
    async def query(self, query: str, include_stats: bool = False):
        """
        Get a generated response for a query.
        
        Retrieves context and generates a response using LLM.
        """
        if not self._retriever:
            await self.__aenter__()

        retrieval_stats = None
        retrieval_start = time.time()
        if include_stats:
            context, retrieval_stats = await self._retriever.get_context_with_stats(query)
        else:
            context = await self.get_context(query)
        retrieval_ms = (time.time() - retrieval_start) * 1000

        logger.info(f"Context for RAG query: {context[:200]}...")

        if include_stats:
            generation_start = time.time()
            answer, llm_stats = await generate_response_with_context_with_stats(query, context)
            generation_ms = (time.time() - generation_start) * 1000
            retrieval_stats = retrieval_stats or {}
            retrieval_stats["context_preview"] = context[:1000]
            return {
                "answer": answer,
                "model": llm_stats.get("model"),
                "usage": llm_stats.get("usage"),
                "temperature": llm_stats.get("temperature"),
                "prompt_tokens": llm_stats.get("prompt_tokens"),
                "completion_tokens": llm_stats.get("completion_tokens"),
                "context_chars": len(context),
                "retrieval": retrieval_stats,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms
            }

        return await generate_response_with_context(query, context)
    
    async def close(self):
        """Close resources."""
        await self.__aexit__(None, None, None)
    
    # ========== V2 Memory Engine: get_user_context ==========
    
    async def get_user_context(
        self,
        current_conversation: Optional[str] = None,
        include_goals: bool = True,
        include_psyche: bool = True,
        include_previous_episode: bool = True,
        max_episodes: int = 5,
        max_goals: int = 10,
        max_psyche: int = 10
    ) -> str:
        """
        Compose structured context from memory layers using the universal XML formatter.
        """
        if not self._memory_store:
            await self.__aenter__()
        
        all_memories = []
        
        # 1. Previous Episodes
        if include_previous_episode:
            episodes = await self._memory_store.get_recent(
                self.user_id, 
                memory_type="episode", 
                limit=max_episodes
            )
            all_memories.extend(episodes)
        
        # 2. Active Goals
        if include_goals:
            goals = await self._memory_store.get_by_type("goal", self.user_id, limit=max_goals)
            all_memories.extend(goals)
        
        # 3. Psyche (traits, preferences)
        if include_psyche:
            psyche = await self._memory_store.get_by_type("psyche", self.user_id, limit=max_psyche)
            all_memories.extend(psyche)
        
        # Generate XML context
        context = format_memories_for_llm(all_memories)
        
        # 4. Current Conversation (Keep as raw text outside XML context for now)
        if current_conversation:
            context = f"{context}\n\n<current_conversation>\n{current_conversation}\n</current_conversation>"
        
        logger.info(f"Generated universal user context: {len(all_memories)} memories, {len(context)} chars")
        return context
