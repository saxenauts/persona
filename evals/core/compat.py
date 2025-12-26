"""
Compatibility layer for legacy adapters.

Design: Wrap old-style adapters (query() -> str with last_query_stats hack)
to work with new interface (query() -> QueryResult).

This enables gradual migration without breaking existing adapters.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

from .models import (
    Session,
    QueryResult,
    RetrievedItem,
    Usage,
    Latency,
)
from .interfaces import AdapterCapabilities


class LegacyAdapterWrapper:
    """
    Wraps a legacy MemorySystem adapter to work with new interface.

    The old interface:
    - add_session(user_id, session_data: str, date: str)
    - add_sessions(user_id, sessions: list[dict])  # {"content": str, "date": str}
    - query(user_id, query: str) -> str
    - reset(user_id)
    - last_query_stats: dict  # Side-channel for telemetry

    The new interface:
    - add_sessions(user_id, sessions: Sequence[Session])
    - query(user_id, query: str) -> QueryResult
    - Telemetry embedded in QueryResult
    """

    def __init__(self, legacy_adapter: Any, name: Optional[str] = None):
        """
        Args:
            legacy_adapter: Old-style MemorySystem instance
            name: Adapter name (defaults to class name)
        """
        self._legacy = legacy_adapter
        self._name = name or legacy_adapter.__class__.__name__

        # Detect capabilities from legacy adapter
        self._capabilities = AdapterCapabilities(
            supports_async=False,
            supports_bulk_ingest=hasattr(legacy_adapter, "add_sessions"),
            supports_retrieval_items=getattr(legacy_adapter, "log_node_content", False),
            supports_context_text=True,  # We can extract from last_query_stats
            supports_reset=hasattr(legacy_adapter, "reset"),
            supports_user_namespace=True,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> AdapterCapabilities:
        return self._capabilities

    def reset(self, user_id: str) -> None:
        """Delegate to legacy reset."""
        if hasattr(self._legacy, "reset"):
            self._legacy.reset(user_id)

    def add_sessions(self, user_id: str, sessions: Sequence[Session]) -> None:
        """Convert Session objects to legacy format and ingest."""
        legacy_sessions = [
            {"content": s.content, "date": s.date or "unknown"} for s in sessions
        ]

        if hasattr(self._legacy, "add_sessions"):
            self._legacy.add_sessions(user_id, legacy_sessions)
        else:
            for s in legacy_sessions:
                self._legacy.add_session(user_id, s["content"], s["date"])

    def query(self, user_id: str, query: str, *, trace: bool = True) -> QueryResult:
        """
        Query legacy adapter and convert response to QueryResult.

        Extracts telemetry from last_query_stats if available.
        """
        started = datetime.utcnow()

        try:
            answer = self._legacy.query(user_id, query)
        except Exception as e:
            finished = datetime.utcnow()
            return QueryResult(
                answer="",
                error=str(e),
                started_at=started,
                finished_at=finished,
            )

        finished = datetime.utcnow()

        # Extract stats from legacy side-channel
        stats = getattr(self._legacy, "last_query_stats", None) or {}
        retrieval_stats = stats.get("retrieval", {}) or {}

        # Build retrieved items if available
        retrieved = self._extract_retrieved_items(stats)

        # Extract context text
        context_text = retrieval_stats.get("context_preview")

        # Build usage info
        usage = Usage(
            model=stats.get("model"),
            prompt_tokens=stats.get("prompt_tokens"),
            completion_tokens=stats.get("completion_tokens"),
            total_tokens=(stats.get("prompt_tokens") or 0)
            + (stats.get("completion_tokens") or 0)
            or None,
        )

        # Build latency info
        latency = Latency(
            retrieval_ms=stats.get("retrieval_ms")
            or retrieval_stats.get("duration_ms"),
            generation_ms=stats.get("generation_ms"),
            total_ms=(finished - started).total_seconds() * 1000,
        )

        return QueryResult(
            answer=answer or "",
            retrieved=tuple(retrieved),
            context_text=context_text,
            usage=usage,
            latency=latency,
            telemetry=stats,
            started_at=started,
            finished_at=finished,
        )

    def _extract_retrieved_items(self, stats: dict) -> list[RetrievedItem]:
        """Extract retrieved items from legacy stats structure."""
        items = []
        retrieval = stats.get("retrieval", {}) or {}

        # Try vector search seeds
        vector_stats = retrieval.get("vector_search", {}) or {}
        for i, seed in enumerate(vector_stats.get("seeds", []) or []):
            if not seed:
                continue
            items.append(
                RetrievedItem(
                    id=seed.get("node_id", f"seed_{i}"),
                    text=seed.get("content")
                    or seed.get("text")
                    or seed.get("fact")
                    or "",
                    score=seed.get("score"),
                    rank=i + 1,
                    source="vector",
                    node_type=seed.get("node_type"),
                    metadata=seed,
                )
            )

        # Try graph traversal nodes
        graph_stats = retrieval.get("graph_traversal", {}) or {}
        node_details = graph_stats.get("node_details", {}) or {}
        for i, node_id in enumerate(graph_stats.get("final_ranked_nodes", []) or []):
            if node_id in {item.id for item in items}:
                continue  # Already added from vector search
            details = node_details.get(node_id, {}) or {}
            items.append(
                RetrievedItem(
                    id=node_id,
                    text=details.get("content") or details.get("fact") or "",
                    rank=len(items) + 1,
                    source="graph",
                    node_type=details.get("type"),
                    metadata=details,
                )
            )

        return items

    # === Async methods (wrap sync with to_thread) ===

    async def areset(self, user_id: str) -> None:
        await asyncio.to_thread(self.reset, user_id)

    async def aadd_sessions(self, user_id: str, sessions: Sequence[Session]) -> None:
        await asyncio.to_thread(self.add_sessions, user_id, sessions)

    async def aquery(
        self, user_id: str, query: str, *, trace: bool = True
    ) -> QueryResult:
        return await asyncio.to_thread(self.query, user_id, query, trace=trace)


def wrap_legacy_adapter(
    legacy_adapter: Any, name: Optional[str] = None
) -> LegacyAdapterWrapper:
    """
    Factory function to wrap a legacy adapter.

    Usage:
        from evals.adapters.persona_adapter import PersonaAdapter
        from evals.core import wrap_legacy_adapter

        legacy = PersonaAdapter()
        adapter = wrap_legacy_adapter(legacy, name="persona")

        result = adapter.query("user_123", "What's my favorite color?")
        print(result.answer)
        print(result.retrieved)  # Now available!
    """
    return LegacyAdapterWrapper(legacy_adapter, name)
