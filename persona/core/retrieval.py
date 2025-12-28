"""Retrieval Layer: Vector Search + Graph Crawl for working memory retrieval.

Simplified retriever focused on:
- Vector similarity search
- Graph traversal from seeds
- Query expansion for temporal/entity parsing

No routing logic - LLM decides what to query via tools.
"""

import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple, Union
from uuid import UUID

from persona.core.graph_ops import GraphOps
from persona.core.memory_store import MemoryStore
from persona.core.context import ContextFormatter
from persona.core.query_expansion import expand_query, QueryExpansion, DateRange
from persona.models.memory import Memory, UserCard
from server.logging_config import get_logger

logger = get_logger(__name__)


class Retriever:
    """Retrieves working memory for LLM via Vector Search + Graph Crawl."""

    def __init__(self, user_id: str, store: MemoryStore, graph_ops: GraphOps):
        self.user_id = user_id
        self.store = store
        self.graph_ops = graph_ops
        self.formatter = ContextFormatter()

    async def get_working_memory(
        self,
        query: str,
        top_k: int = 5,
        hop_depth: int = 1,
        include_static: bool = True,
        user_timezone: str = "UTC",
        use_query_expansion: bool = True,
        user_card: Optional[UserCard] = None,
        collect_stats: bool = False,
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """
        Get formatted working memory for a dialog turn.

        Returns XML-formatted context string, optionally with stats dict.
        """
        all_memories: dict[UUID, Memory] = {}
        stats: Dict[str, Any] = {}
        expansion: Optional[QueryExpansion] = None

        # Query expansion for temporal refs and entities
        if use_query_expansion:
            try:
                expansion_start = time.time()
                expansion = await expand_query(query, user_timezone=user_timezone)
                if collect_stats:
                    stats["query_expansion"] = {
                        "original_query": query,
                        "semantic_query": expansion.semantic_query,
                        "date_range": (
                            {
                                "start": str(expansion.date_range.start),
                                "end": str(expansion.date_range.end),
                            }
                            if expansion.date_range
                            else None
                        ),
                        "entities": expansion.entities,
                        "duration_ms": (time.time() - expansion_start) * 1000,
                    }
                logger.debug(
                    f"Query expansion: date_range={expansion.date_range}, "
                    f"entities={expansion.entities}, semantic_query={expansion.semantic_query}"
                )
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}")

        # Static context: notes and psyche
        if include_static:
            static = await self._get_static_context()
            for m in static:
                all_memories[m.id] = m
            if collect_stats:
                stats["static_count"] = len(static)
            logger.debug(f"Static context: {len(static)} memories")

        # Vector search
        search_query = expansion.semantic_query if expansion else query
        date_range = expansion.date_range if expansion else None

        vector_start = time.time()
        seeds, seed_nodes = await self._vector_search(
            search_query, top_k, date_range=date_range
        )
        for m in seeds:
            all_memories[m.id] = m

        if collect_stats:
            stats["vector_search"] = {
                "top_k": top_k,
                "seeds": seed_nodes,
                "duration_ms": (time.time() - vector_start) * 1000,
            }
        logger.debug(f"Vector search: {len(seeds)} seeds")

        # Graph expansion
        graph_start = time.time()
        expanded, graph_stats = await self._expand_graph(
            seeds, hop_depth, query_expansion=expansion
        )
        for m in expanded:
            all_memories[m.id] = m

        if collect_stats:
            stats["graph_traversal"] = {
                "max_hops": hop_depth,
                "nodes_visited": graph_stats["nodes_visited"],
                "relationships_traversed": graph_stats["relationships_traversed"],
                "final_ranked_nodes": graph_stats["final_ranked_nodes"],
                "duration_ms": (time.time() - graph_start) * 1000,
            }
        logger.debug(f"Graph expansion: {len(expanded)} total after crawl")

        # Format as working memory
        memories = list(all_memories.values())
        working_memory = self.formatter.format_working_memory(
            memories, user_card=user_card
        )

        if collect_stats:
            stats["working_memory_chars"] = len(working_memory)
            logger.info(
                f"Retriever stats: seeds={len(seed_nodes)}, nodes={graph_stats['nodes_visited']}, "
                f"working_memory_chars={len(working_memory)}"
            )
            return working_memory, stats

        logger.info(f"Retriever: {len(memories)} memories, {len(working_memory)} chars")
        return working_memory

    async def get_working_memory_with_stats(
        self,
        query: str,
        top_k: int = 5,
        hop_depth: int = 1,
        include_static: bool = True,
        user_timezone: str = "UTC",
        use_query_expansion: bool = True,
        user_card: Optional[UserCard] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Get formatted working memory plus retrieval stats."""
        result = await self.get_working_memory(
            query=query,
            top_k=top_k,
            hop_depth=hop_depth,
            include_static=include_static,
            user_timezone=user_timezone,
            use_query_expansion=use_query_expansion,
            user_card=user_card,
            collect_stats=True,
        )
        return result  # type: ignore

    async def _get_static_context(
        self,
        note_limit: int = 5,
        psyche_limit: int = 5,
    ) -> List[Memory]:
        """Get static context: active notes and psyche traits."""
        memories = []

        if note_limit > 0:
            try:
                notes = await self.store.get_by_type(
                    "note", self.user_id, limit=note_limit
                )
                active_notes = [
                    n for n in notes if getattr(n, "status", "active") != "COMPLETED"
                ]
                memories.extend(active_notes)
            except Exception as e:
                logger.warning(f"Failed to get notes for static context: {e}")

        if psyche_limit > 0:
            try:
                psyche = await self.store.get_by_type(
                    "psyche", self.user_id, limit=psyche_limit
                )
                memories.extend(psyche)
            except Exception as e:
                logger.warning(f"Failed to get psyche for static context: {e}")

        return memories

    async def _vector_search(
        self, query: str, top_k: int, date_range: Optional[DateRange] = None
    ) -> Tuple[List[Memory], List[Dict[str, Any]]]:
        """Semantic similarity search: embed query and find top-K similar memories."""
        effective_date_range = None
        if date_range:
            effective_date_range = (date_range.start, date_range.end)
        else:
            effective_date_range = self._extract_date_filter(query)

        try:
            results = await self.graph_ops.text_similarity_search(
                query=query,
                user_id=self.user_id,
                limit=top_k,
                date_range=effective_date_range,
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return [], []

        memories = []
        seed_nodes = []

        for r in results.get("results", []):
            mid = r.get("nodeName")
            score = r.get("score")
            try:
                mem = await self.store.get(UUID(mid), self.user_id)
                if mem:
                    memories.append(mem)
                    seed_nodes.append(
                        {
                            "node_id": str(mem.id),
                            "score": score,
                            "node_type": getattr(mem, "type", "unknown"),
                        }
                    )
            except (ValueError, Exception) as e:
                logger.debug(f"Could not retrieve memory {mid}: {e}")

        return memories, seed_nodes

    def _extract_date_filter(self, query: str) -> Optional[tuple]:
        """Extract date filter from query string with format: (date: YYYY-MM-DD)."""
        import re

        match = re.search(r"\(date:\s*(\d{4}-\d{2}-\d{2})\)", query)
        if match:
            date_str = match.group(1)
            try:
                anchor_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                query_lower = query.lower()

                if any(
                    term in query_lower
                    for term in ["last week", "past week", "week ago"]
                ):
                    start = anchor_date - timedelta(days=7)
                    return (start, anchor_date)

                if "yesterday" in query_lower:
                    d = anchor_date - timedelta(days=1)
                    return (d, d)

                return None
            except ValueError:
                pass

        return None

    async def _expand_graph(
        self,
        seeds: List[Memory],
        hop_depth: int,
        max_links_per_node: int = 15,
        max_total_nodes: int = 50,
        query_expansion: Optional[QueryExpansion] = None,
    ) -> Tuple[List[Memory], Dict[str, Any]]:
        """Crawl graph from seeds with query-aware link scoring. Returns (memories, stats)."""
        if hop_depth <= 0:
            return seeds, {
                "nodes_visited": len(seeds),
                "relationships_traversed": 0,
                "final_ranked_nodes": [str(m.id) for m in seeds],
            }

        all_memories = {m.id: m for m in seeds}
        frontier = list(seeds)
        relationships_traversed = 0

        for hop in range(hop_depth):
            if len(all_memories) >= max_total_nodes:
                break

            per_node_limit = max_links_per_node if hop == 0 else max_links_per_node // 2
            next_frontier = []

            for memory in frontier:
                if len(all_memories) >= max_total_nodes:
                    break

                try:
                    linked = await self.store.get_connected(memory.id, self.user_id)
                    relationships_traversed += len(linked)
                    scored = self._score_links(linked, query_expansion)
                    scored.sort(key=lambda x: x[0], reverse=True)
                    top_linked = [m for _, m in scored[:per_node_limit]]

                    for m in top_linked:
                        if m.id not in all_memories:
                            all_memories[m.id] = m
                            next_frontier.append(m)
                            if len(all_memories) >= max_total_nodes:
                                break
                except Exception as e:
                    logger.debug(f"Failed to get connected for {memory.id}: {e}")

            frontier = next_frontier
            logger.debug(
                f"Hop {hop + 1}: {len(next_frontier)} new, total={len(all_memories)}"
            )

        memories = list(all_memories.values())
        stats = {
            "nodes_visited": len(memories),
            "relationships_traversed": relationships_traversed,
            "final_ranked_nodes": [str(m.id) for m in memories],
        }

        return memories, stats

    def _score_links(
        self, linked_memories: List[Memory], expansion: Optional[QueryExpansion]
    ) -> List[Tuple[float, Memory]]:
        """Score linked memories: importance + entity matches + recency bonus."""
        scored = []
        query_entities = set(
            e.lower() for e in (expansion.entities if expansion else [])
        )

        for mem in linked_memories:
            score = getattr(mem, "importance", 0.5)

            if query_entities:
                mem_text = f"{mem.title} {mem.content}".lower()
                entity_matches = sum(1 for e in query_entities if e in mem_text)
                score += entity_matches * 0.2

            if hasattr(mem, "timestamp"):
                days_old = (datetime.utcnow() - mem.timestamp).days
                if days_old < 7:
                    score += 0.3
                elif days_old < 30:
                    score += 0.1

            scored.append((score, mem))

        return scored
