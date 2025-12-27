"""Retrieval Layer: Vector Search + Graph Crawl for context retrieval."""

import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple, Union
from uuid import UUID

from persona.core.graph_ops import GraphOps
from persona.core.memory_store import MemoryStore
from persona.core.context import ContextFormatter, ContextView
from persona.core.query_expansion import expand_query, QueryExpansion, DateRange
from persona.core.intent_router import IntentRouter, RetrievalHints, RetrievalMode
from persona.models.memory import Memory, UserCard
from server.logging_config import get_logger

logger = get_logger(__name__)


TIMELINE_SIGNALS = [
    "what happened",
    "when did",
    "last week",
    "yesterday",
    "recently",
    "history",
    "timeline",
]

TASK_SIGNALS = [
    "what should i do",
    "my tasks",
    "todo",
    "to-do",
    "goals",
    "priorities",
    "what's next",
    "action items",
]

PROFILE_SIGNALS = [
    "who am i",
    "what do i like",
    "my preferences",
    "about me",
    "my values",
    "my personality",
]


class Retriever:
    """Retrieves context for LLM queries via Vector Search + Graph Crawl."""

    def __init__(self, user_id: str, store: MemoryStore, graph_ops: GraphOps):
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
        collect_stats: bool = False,
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """Get formatted context for a user query. Returns (context, stats) if collect_stats=True."""
        all_memories: dict[UUID, Memory] = {}
        stats: Dict[str, Any] = {}
        expansion: Optional[QueryExpansion] = None

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

        view = self._route_context_view(query, expansion)
        if collect_stats:
            stats["context_view"] = view.value

        if include_static:
            static = await self._get_static_context()
            for m in static:
                all_memories[m.id] = m
            if collect_stats:
                stats["static_count"] = len(static)
            logger.debug(f"Static context: {len(static)} memories")

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

        memories = list(all_memories.values())
        context = self.formatter.format_context(
            memories, user_card=user_card, view=view
        )

        if collect_stats:
            stats["context_chars"] = len(context)
            logger.info(
                f"Retriever stats: seeds={len(seed_nodes)}, nodes={graph_stats['nodes_visited']}, "
                f"context_chars={len(context)}, view={view.value}"
            )
            return context, stats

        logger.info(
            f"Retriever: {len(memories)} memories, {len(context)} chars, view={view.value}"
        )
        return context

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
        """Get formatted context plus retrieval stats."""
        result = await self.get_context(
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

    def _route_context_view(
        self, query: str, expansion: Optional[QueryExpansion]
    ) -> ContextView:
        """Route to appropriate context view based on query intent."""
        query_lower = query.lower()

        if expansion and expansion.date_range:
            return ContextView.TIMELINE

        if any(s in query_lower for s in TIMELINE_SIGNALS):
            return ContextView.TIMELINE

        if any(s in query_lower for s in TASK_SIGNALS):
            return ContextView.TASKS

        if any(s in query_lower for s in PROFILE_SIGNALS):
            return ContextView.PROFILE

        if expansion and expansion.entities:
            return ContextView.GRAPH_NEIGHBORHOOD

        return ContextView.PROFILE

    async def _get_static_context(self) -> List[Memory]:
        """Get always-included context: active notes + core psyche items."""
        memories = []

        try:
            notes = await self.store.get_by_type("note", self.user_id, limit=10)
            active_notes = [
                n for n in notes if getattr(n, "status", "active") != "COMPLETED"
            ]
            memories.extend(active_notes)
        except Exception as e:
            logger.warning(f"Failed to get notes for static context: {e}")

        try:
            psyche = await self.store.get_by_type("psyche", self.user_id, limit=5)
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

    async def get_context_with_hints(
        self,
        query: str,
        hints: RetrievalHints,
        user_card: Optional[UserCard] = None,
        collect_stats: bool = False,
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """Get context using IntentRouter hints for spray-and-pray retrieval."""
        all_memories: dict[UUID, Memory] = {}
        stats: Dict[str, Any] = {
            "mode": hints.mode.value,
            "confidence": hints.confidence,
        }

        lane_start = time.time()

        seed_memories = await self._fetch_seed_memories(hints.seed_memory_ids)
        for m in seed_memories:
            all_memories[m.id] = m
        if collect_stats:
            stats["seed_lane"] = {
                "count": len(seed_memories),
                "ids": hints.seed_memory_ids[:5],
            }

        vector_results, vector_nodes = await self._vector_search_with_hints(
            hints.resolved_query or query,
            hints.top_k,
            hints.date_range,
            hints.memory_type_boost,
        )
        for m in vector_results:
            all_memories[m.id] = m
        if collect_stats:
            stats["vector_lane"] = {"count": len(vector_results), "seeds": vector_nodes}

        if hints.search_keywords:
            keyword_results = await self._keyword_spray(
                hints.search_keywords,
                hints.memory_type_boost,
                limit=hints.top_k,
            )
            for m in keyword_results:
                all_memories[m.id] = m
            if collect_stats:
                stats["keyword_lane"] = {
                    "count": len(keyword_results),
                    "keywords": hints.search_keywords[:5],
                }

        if hints.hop_depth > 0:
            seeds_for_expansion = list(all_memories.values())
            expanded, graph_stats = await self._expand_graph_with_hints(
                seeds_for_expansion,
                hints.hop_depth,
                hints.link_type_boost,
            )
            for m in expanded:
                all_memories[m.id] = m
            if collect_stats:
                stats["graph_traversal"] = graph_stats

        if collect_stats:
            stats["total_retrieval_ms"] = (time.time() - lane_start) * 1000

        memories = list(all_memories.values())
        view = self._route_context_view_from_hints(hints)
        context = self.formatter.format_context(
            memories, user_card=user_card, view=view
        )

        if collect_stats:
            stats["context_chars"] = len(context)
            stats["context_view"] = view.value
            stats["total_memories"] = len(memories)
            logger.info(
                f"Retriever (hints): mode={hints.mode.value}, memories={len(memories)}, "
                f"context_chars={len(context)}, confidence={hints.confidence:.2f}"
            )
            return context, stats

        logger.info(
            f"Retriever (hints): {len(memories)} memories, {len(context)} chars"
        )
        return context

    async def _fetch_seed_memories(self, memory_ids: List[str]) -> List[Memory]:
        """Fetch memories by ID from UserCard hints."""
        memories = []
        for mid in memory_ids[:20]:
            try:
                mem = await self.store.get(UUID(mid), self.user_id)
                if mem:
                    memories.append(mem)
            except (ValueError, Exception) as e:
                logger.debug(f"Could not fetch seed memory {mid}: {e}")
        return memories

    async def _vector_search_with_hints(
        self,
        query: str,
        top_k: int,
        date_range: Optional[Tuple],
        memory_type_boost: List[str],
    ) -> Tuple[List[Memory], List[Dict[str, Any]]]:
        """Vector search with optional type filtering."""
        try:
            results = await self.graph_ops.text_similarity_search(
                query=query,
                user_id=self.user_id,
                limit=top_k,
                date_range=date_range,
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
                    if (
                        memory_type_boost
                        and getattr(mem, "type", "") in memory_type_boost
                    ):
                        score = (score or 0) + 0.1
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

        if memory_type_boost:
            memories.sort(
                key=lambda m: (
                    getattr(m, "type", "") in memory_type_boost,
                    getattr(m, "importance", 0.5),
                ),
                reverse=True,
            )

        return memories, seed_nodes

    async def _keyword_spray(
        self,
        keywords: List[str],
        memory_type_boost: List[str],
        limit: int = 10,
    ) -> List[Memory]:
        """Spray-and-pray: search for keywords across memory titles/content."""
        all_matches: Dict[UUID, Memory] = {}

        for keyword in keywords[:10]:
            try:
                results = await self.graph_ops.text_similarity_search(
                    query=keyword,
                    user_id=self.user_id,
                    limit=3,
                )
                for r in results.get("results", []):
                    mid = r.get("nodeName")
                    try:
                        mem = await self.store.get(UUID(mid), self.user_id)
                        if mem and mem.id not in all_matches:
                            all_matches[mem.id] = mem
                    except (ValueError, Exception):
                        pass
            except Exception as e:
                logger.debug(f"Keyword spray failed for '{keyword}': {e}")

        memories = list(all_matches.values())

        if memory_type_boost:
            memories.sort(
                key=lambda m: (
                    getattr(m, "type", "") in memory_type_boost,
                    getattr(m, "importance", 0.5),
                ),
                reverse=True,
            )

        return memories[:limit]

    async def _expand_graph_with_hints(
        self,
        seeds: List[Memory],
        hop_depth: int,
        link_type_boost: List[str],
        max_links_per_node: int = 15,
    ) -> Tuple[List[Memory], Dict[str, Any]]:
        """Graph traversal with link type prioritization."""
        if hop_depth <= 0:
            return seeds, {
                "nodes_visited": len(seeds),
                "relationships_traversed": 0,
            }

        all_memories = {m.id: m for m in seeds}
        frontier = list(seeds)
        relationships_traversed = 0

        for hop in range(hop_depth):
            next_frontier = []
            for memory in frontier:
                try:
                    linked = await self.store.get_connected(memory.id, self.user_id)
                    relationships_traversed += len(linked)

                    scored = []
                    for mem in linked:
                        score = getattr(mem, "importance", 0.5)
                        if hasattr(mem, "timestamp"):
                            days_old = (datetime.utcnow() - mem.timestamp).days
                            if days_old < 7:
                                score += 0.3
                            elif days_old < 30:
                                score += 0.1
                        scored.append((score, mem))

                    scored.sort(key=lambda x: x[0], reverse=True)
                    top_linked = [m for _, m in scored[:max_links_per_node]]

                    for m in top_linked:
                        if m.id not in all_memories:
                            all_memories[m.id] = m
                            next_frontier.append(m)
                except Exception as e:
                    logger.debug(f"Failed to get connected for {memory.id}: {e}")

            frontier = next_frontier

        return list(all_memories.values()), {
            "nodes_visited": len(all_memories),
            "relationships_traversed": relationships_traversed,
        }

    def _route_context_view_from_hints(self, hints: RetrievalHints) -> ContextView:
        """Derive context view from RetrievalHints."""
        if hints.date_range:
            return ContextView.TIMELINE

        if "note" in hints.memory_type_boost:
            return ContextView.TASKS

        if "psyche" in hints.memory_type_boost:
            return ContextView.PROFILE

        if hints.seed_memory_ids:
            return ContextView.GRAPH_NEIGHBORHOOD

        return ContextView.PROFILE
