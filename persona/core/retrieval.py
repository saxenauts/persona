"""
Retrieval Layer for Persona.

Retrieves context for LLM queries using Vector Search + Graph Crawl.
"""

import time
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID

from persona.core.graph_ops import GraphOps
from persona.core.memory_store import MemoryStore
from persona.core.context import ContextFormatter, convert_to_memories, ContextView
from persona.core.query_expansion import expand_query, QueryExpansion, DateRange
from persona.models.memory import Memory, UserCard
from server.logging_config import get_logger

logger = get_logger(__name__)


class Retriever:
    """
    Retrieves context for LLM queries.

    V1 Implementation: Vector Search + Graph Crawl.

    TODO:
    - Agentic Retrieval: Implement multi-step retrieval loops to refine context.
    - Reasoning Models: Enable toggle for o1/reasoning-style models for better query planning.
    - Agent Loops: Instrumentation for multi-turn internal reasoning before final response.

    Future Signals (TODO):
    - BM25: Keyword match for exact terms, proper nouns.
    - Type Filter: Restrict to specific memory types (note, psyche).
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
        include_static: bool = True,
        user_timezone: str = "UTC",
        use_query_expansion: bool = True,
        user_card: Optional[UserCard] = None,
    ) -> str:
        """
        Get formatted context for a user query.

        Args:
            query: Natural language query.
            top_k: Number of vector search results.
            hop_depth: How many relationship hops to crawl.
            include_static: Whether to include static context (active goals, psyche).
            user_timezone: User's timezone for temporal query parsing.
            use_query_expansion: Whether to use LLM-enhanced query expansion.
            user_card: Optional UserCard for identity anchor (placed first in context).

        Returns:
            XML-formatted context string.
        """
        all_memories: dict[UUID, Memory] = {}
        expansion: Optional[QueryExpansion] = None

        if use_query_expansion:
            try:
                expansion = await expand_query(query, user_timezone=user_timezone)
                logger.debug(
                    f"Query expansion: date_range={expansion.date_range}, "
                    f"entities={expansion.entities}, semantic_query={expansion.semantic_query}"
                )
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}")

        view = self._route_context_view(query, expansion)

        if include_static:
            static = await self._get_static_context()
            for m in static:
                all_memories[m.id] = m
            logger.debug(f"Static context: {len(static)} memories")

        search_query = expansion.semantic_query if expansion else query
        date_range = expansion.date_range if expansion else None
        seeds, _seed_nodes = await self._vector_search(
            search_query, top_k, date_range=date_range
        )
        for m in seeds:
            all_memories[m.id] = m
        logger.debug(f"Vector search: {len(seeds)} seeds")

        expanded = await self._expand_graph(seeds, hop_depth, query_expansion=expansion)
        for m in expanded:
            all_memories[m.id] = m
        logger.debug(f"Graph expansion: {len(expanded)} total after crawl")

        memories = list(all_memories.values())
        context = self.formatter.format_context(
            memories, user_card=user_card, view=view
        )
        logger.info(
            f"Retriever: {len(memories)} memories, {len(context)} chars, view={view.value}"
        )

        return context

    def _route_context_view(
        self, query: str, expansion: Optional[QueryExpansion]
    ) -> ContextView:
        """
        Route to appropriate context view based on query intent.

        Uses query keywords and expansion signals to determine view:
        - TIMELINE: temporal queries ("what happened", date ranges)
        - TASKS: task/goal queries ("what should I do", "my tasks")
        - PROFILE: identity queries ("who am I", "what do I like")
        - GRAPH_NEIGHBORHOOD: entity-focused ("about X", specific topics)
        """
        query_lower = query.lower()

        if expansion and expansion.date_range:
            return ContextView.TIMELINE

        timeline_signals = [
            "what happened",
            "when did",
            "last week",
            "yesterday",
            "recently",
            "history",
            "timeline",
        ]
        if any(s in query_lower for s in timeline_signals):
            return ContextView.TIMELINE

        task_signals = [
            "what should i do",
            "my tasks",
            "todo",
            "to-do",
            "goals",
            "priorities",
            "what's next",
            "action items",
        ]
        if any(s in query_lower for s in task_signals):
            return ContextView.TASKS

        profile_signals = [
            "who am i",
            "what do i like",
            "my preferences",
            "about me",
            "my values",
            "my personality",
        ]
        if any(s in query_lower for s in profile_signals):
            return ContextView.PROFILE

        if expansion and expansion.entities:
            return ContextView.GRAPH_NEIGHBORHOOD

        return ContextView.PROFILE

    async def get_context_with_stats(
        self,
        query: str,
        top_k: int = 5,
        hop_depth: int = 1,
        include_static: bool = True,
        user_timezone: str = "UTC",
        use_query_expansion: bool = True,
        user_card: Optional[UserCard] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Get formatted context plus retrieval stats for logging."""
        all_memories: dict[UUID, Memory] = {}
        stats: Dict[str, Any] = {}
        expansion: Optional[QueryExpansion] = None

        if use_query_expansion:
            try:
                expansion_start = time.time()
                expansion = await expand_query(query, user_timezone=user_timezone)
                expansion_ms = (time.time() - expansion_start) * 1000
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
                    "duration_ms": expansion_ms,
                }
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}")

        view = self._route_context_view(query, expansion)
        stats["context_view"] = view.value

        if include_static:
            static = await self._get_static_context()
            for m in static:
                all_memories[m.id] = m
            stats["static_count"] = len(static)

        search_query = expansion.semantic_query if expansion else query
        date_range = expansion.date_range if expansion else None
        vector_start = time.time()
        seeds, seed_nodes = await self._vector_search(
            search_query, top_k, date_range=date_range
        )
        vector_ms = (time.time() - vector_start) * 1000
        for m in seeds:
            all_memories[m.id] = m

        graph_start = time.time()
        expanded, graph_stats = await self._expand_graph_with_stats(
            seeds, hop_depth, query_expansion=expansion
        )
        graph_ms = (time.time() - graph_start) * 1000
        for m in expanded:
            all_memories[m.id] = m

        memories = list(all_memories.values())
        context = self.formatter.format_context(
            memories, user_card=user_card, view=view
        )

        stats["vector_search"] = {
            "top_k": top_k,
            "seeds": seed_nodes,
            "duration_ms": vector_ms,
        }
        stats["graph_traversal"] = {
            "max_hops": hop_depth,
            "nodes_visited": graph_stats["nodes_visited"],
            "relationships_traversed": graph_stats["relationships_traversed"],
            "final_ranked_nodes": graph_stats["final_ranked_nodes"],
            "duration_ms": graph_ms,
        }
        stats["context_chars"] = len(context)

        logger.info(
            f"Retriever stats: seeds={len(seed_nodes)}, nodes={graph_stats['nodes_visited']}, "
            f"context_chars={len(context)}, view={view.value}"
        )

        return context, stats

    async def _get_static_context(self) -> List[Memory]:
        """
        Get always-included context.

        TODO: Implement persistent memory marking (pinned flag).

        Currently includes:
        - Active notes (status != "COMPLETED")
        - Core psyche items (first 5)
        """
        memories = []

        # Active notes
        try:
            notes = await self.store.get_by_type("note", self.user_id, limit=10)
            active_notes = [
                n for n in notes if getattr(n, "status", "active") != "COMPLETED"
            ]
            memories.extend(active_notes)
        except Exception as e:
            logger.warning(f"Failed to get notes for static context: {e}")

        # Core psyche
        try:
            psyche = await self.store.get_by_type("psyche", self.user_id, limit=5)
            memories.extend(psyche)
        except Exception as e:
            logger.warning(f"Failed to get psyche for static context: {e}")

        return memories

    async def _vector_search(
        self, query: str, top_k: int, date_range: Optional[DateRange] = None
    ) -> Tuple[List[Memory], List[Dict[str, Any]]]:
        """
        Semantic similarity search.

        Embeds the query and finds top-K most similar memories.

        Args:
            query: The search query (ideally cleaned semantic_query from expansion).
            top_k: Number of results to return.
            date_range: Optional DateRange from query expansion (takes precedence over
                        inline date parsing).
        """
        # Use provided date_range or fall back to inline extraction
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

        memory_ids = [r["nodeName"] for r in results.get("results", [])]
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
        """
        Extract date filter from query string.

        Supports format: "(date: YYYY-MM-DD) ..."
        Returns: (start_date, end_date) tuple or None
        """
        import re
        from datetime import datetime, timedelta

        # Match "(date: 2023-05-29)"
        match = re.search(r"\(date:\s*(\d{4}-\d{2}-\d{2})\)", query)
        if match:
            date_str = match.group(1)
            try:
                # TODO: Implement smarter range logic (e.g. "last week").
                # For now, if a specific date is mentioned, we search a wide window around it?
                # Actually, if the user provides a reference date, they usually mean relative to that.
                # But since the goal is "Date-Based Retrieval" for a specific reference point:
                # Let's interpret the tag as setting the "current" context date,
                # but NOT necessarily a strict filter unless the query asks for "last week".

                # However, for this specific task (fixing "last week" failure):
                # We need to detect "last week" in the query content AND use the date tag as anchor.

                # Simple Heuristic V1:
                # If query contains "last week" or "week ago", window = [date-7, date].
                # If query contains "yesterday", window = [date-1, date-1].

                anchor_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                query_lower = query.lower()

                if (
                    "last week" in query_lower
                    or "past week" in query_lower
                    or "week ago" in query_lower
                ):
                    start = anchor_date - timedelta(days=7)
                    return (start, anchor_date)

                if "yesterday" in query_lower:
                    d = anchor_date - timedelta(days=1)
                    return (d, d)

                # If no temporal keyword, do we filter efficiently?
                # Ideally no filter is safer than wrong filter.
                # BUT, if we want to boost precision for "What happened on X date",
                # we might want a filter.
                # Let's start conservative: Only filter if specific relative terms are found.
                return None

            except ValueError:
                pass

        return None

    async def _expand_graph(
        self,
        seeds: List[Memory],
        hop_depth: int,
        max_links_per_node: int = 15,
        query_expansion: Optional[QueryExpansion] = None,
    ) -> List[Memory]:
        """
        Crawl graph from seed memories with query-aware link scoring.

        Follows relationships to find linked memories up to hop_depth.
        Uses query expansion hints to prioritize relevant link types.
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
                    scored = self._score_links(linked, query_expansion)
                    scored.sort(key=lambda x: x[0], reverse=True)
                    top_linked = [m for _, m in scored[:max_links_per_node]]

                    for m in top_linked:
                        if m.id not in all_memories:
                            all_memories[m.id] = m
                            next_frontier.append(m)
                except Exception as e:
                    logger.debug(f"Failed to get connected for {memory.id}: {e}")

            frontier = next_frontier
            logger.debug(
                f"Hop {hop + 1}: {len(next_frontier)} new, total={len(all_memories)}"
            )

        return list(all_memories.values())

    def _score_links(
        self, linked_memories: List[Memory], expansion: Optional[QueryExpansion]
    ) -> List[Tuple[float, Memory]]:
        """
        Score linked memories based on query relevance.

        Scoring factors:
        - Memory type priority (psyche > note > episode for identity queries)
        - Importance field
        - Entity match (if expansion has entities)
        - Recency bonus for episodes
        """
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
                from datetime import datetime, timedelta

                days_old = (datetime.utcnow() - mem.timestamp).days
                if days_old < 7:
                    score += 0.3
                elif days_old < 30:
                    score += 0.1

            scored.append((score, mem))

        return scored

    async def _expand_graph_with_stats(
        self,
        seeds: List[Memory],
        hop_depth: int,
        max_links_per_node: int = 15,
        query_expansion: Optional[QueryExpansion] = None,
    ) -> Tuple[List[Memory], Dict[str, Any]]:
        """Crawl graph with stats for logging/debugging."""
        if hop_depth <= 0:
            stats = {
                "nodes_visited": len(seeds),
                "relationships_traversed": 0,
                "final_ranked_nodes": [str(m.id) for m in seeds],
            }
            return seeds, stats

        all_memories = {m.id: m for m in seeds}
        frontier = list(seeds)
        relationships_traversed = 0

        for hop in range(hop_depth):
            next_frontier = []
            for memory in frontier:
                try:
                    linked = await self.store.get_connected(memory.id, self.user_id)
                    relationships_traversed += len(linked)
                    scored = self._score_links(linked, query_expansion)
                    scored.sort(key=lambda x: x[0], reverse=True)
                    top_linked = [m for _, m in scored[:max_links_per_node]]

                    for m in top_linked:
                        if m.id not in all_memories:
                            all_memories[m.id] = m
                            next_frontier.append(m)
                except Exception as e:
                    logger.debug(f"Failed to get connected for {memory.id}: {e}")

            frontier = next_frontier

        memories = list(all_memories.values())
        stats = {
            "nodes_visited": len(memories),
            "relationships_traversed": relationships_traversed,
            "final_ranked_nodes": [str(m.id) for m in memories],
        }

        return memories, stats
