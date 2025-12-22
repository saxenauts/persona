"""
Retrieval Layer for Persona.

Retrieves context for LLM queries using Vector Search + Graph Crawl.
"""

from typing import List, Optional
from uuid import UUID

from persona.core.graph_ops import GraphOps
from persona.core.memory_store import MemoryStore
from persona.core.context import ContextFormatter, convert_to_memories
from persona.models.memory import Memory
from server.logging_config import get_logger

logger = get_logger(__name__)


class Retriever:
    """
    Retrieves context for LLM queries.
    
    V1 Implementation: Vector Search + Graph Crawl.
    
    Future Signals (TODO):
    - BM25: Keyword match for exact terms, proper nouns.
    - Type Filter: Restrict to specific memory types (goal, psyche).
    - Date Filter: Temporal window queries.
    - Re-Ranker: Cross-encoder for precision (expensive).
    - RRF Fusion: Merge ranked lists from multiple signals.
    - LLM Query Planner: Generate smart queries from schema knowledge.
    """
    
    def __init__(self, user_id: str, store: MemoryStore, graph_ops: GraphOps):
        """
        Initialize retriever.
        
        Args:
            user_id: The user whose memories to retrieve.
            store: MemoryStore for typed memory access.
            graph_ops: GraphOps for vector search and graph queries.
        """
        self.user_id = user_id
        self.store = store
        self.graph_ops = graph_ops
        self.formatter = ContextFormatter()
    
    async def get_context(
        self, 
        query: str, 
        top_k: int = 5, 
        hop_depth: int = 1,
        include_static: bool = True
    ) -> str:
        """
        Get formatted context for a user query.
        
        Args:
            query: Natural language query.
            top_k: Number of vector search results.
            hop_depth: How many relationship hops to crawl.
            include_static: Whether to include static context (active goals, psyche).
        
        Returns:
            XML-formatted context string.
        """
        all_memories: dict[UUID, Memory] = {}
        
        # 1. Static Context (always-on background)
        if include_static:
            static = await self._get_static_context()
            for m in static:
                all_memories[m.id] = m
            logger.debug(f"Static context: {len(static)} memories")
        
        # 2. Vector Search (query-specific)
        seeds = await self._vector_search(query, top_k)
        for m in seeds:
            all_memories[m.id] = m
        logger.debug(f"Vector search: {len(seeds)} seeds")
        
        # 3. Graph Crawl (expand from seeds)
        expanded = await self._expand_graph(seeds, hop_depth)
        for m in expanded:
            all_memories[m.id] = m
        logger.debug(f"Graph expansion: {len(expanded)} total after crawl")
        
        # 4. Format
        memories = list(all_memories.values())
        context = self.formatter.format_context(memories)
        logger.info(f"Retriever: {len(memories)} memories, {len(context)} chars context")
        
        return context
    
    async def _get_static_context(self) -> List[Memory]:
        """
        Get always-included context.
        
        TODO: Implement persistent memory marking (pinned flag).
        
        Currently includes:
        - Active goals (status != "COMPLETED")
        - Core psyche items (first 5)
        """
        memories = []
        
        # Active goals
        try:
            goals = await self.store.get_by_type("goal", self.user_id, limit=10)
            active_goals = [g for g in goals if getattr(g, 'status', 'active') != "COMPLETED"]
            memories.extend(active_goals)
        except Exception as e:
            logger.warning(f"Failed to get goals for static context: {e}")
        
        # Core psyche
        try:
            psyche = await self.store.get_by_type("psyche", self.user_id, limit=5)
            memories.extend(psyche)
        except Exception as e:
            logger.warning(f"Failed to get psyche for static context: {e}")
        
        return memories
    
    async def _vector_search(self, query: str, top_k: int) -> List[Memory]:
        """
        Semantic similarity search.
        
        Embeds the query and finds top-K most similar memories.
        """
        try:
            results = await self.graph_ops.text_similarity_search(
                query=query, 
                user_id=self.user_id, 
                limit=top_k
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
        
        memory_ids = [r['nodeName'] for r in results.get('results', [])]
        memories = []
        
        for mid in memory_ids:
            try:
                mem = await self.store.get(UUID(mid), self.user_id)
                if mem:
                    memories.append(mem)
            except (ValueError, Exception) as e:
                logger.debug(f"Could not retrieve memory {mid}: {e}")
        
        return memories
    
    async def _expand_graph(
        self, 
        seeds: List[Memory], 
        hop_depth: int
    ) -> List[Memory]:
        """
        Crawl graph from seed memories.
        
        Follows relationships to find linked memories up to hop_depth.
        This justifies using a graph database - we get relational context
        that pure vector search would miss.
        """
        if hop_depth <= 0:
            return seeds
        
        all_memories = {m.id: m for m in seeds}
        frontier = list(seeds)
        
        for hop in range(hop_depth):
            next_frontier = []
            for memory in frontier:
                try:
                    linked = await self.store.get_connected(memory.id, self.user_id)
                    for m in linked:
                        if m.id not in all_memories:
                            all_memories[m.id] = m
                            next_frontier.append(m)
                except Exception as e:
                    logger.debug(f"Failed to get connected for {memory.id}: {e}")
            
            frontier = next_frontier
            logger.debug(f"Hop {hop + 1}: found {len(next_frontier)} new memories")
        
        return list(all_memories.values())
