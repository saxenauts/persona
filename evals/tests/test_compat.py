"""
Tests for adapter compatibility layer.
"""

import pytest
from unittest.mock import MagicMock

from evals.core.models import Session, QueryResult, RetrievedItem
from evals.core.compat import LegacyAdapterWrapper, wrap_legacy_adapter
from evals.core.interfaces import AdapterCapabilities


# === Mock Legacy Adapter ===


class MockLegacyAdapter:
    """Mock implementation of legacy MemorySystem interface."""

    log_node_content = False

    def __init__(self):
        self.last_query_stats = None
        self._sessions = []
        self._reset_called = False

    def add_session(self, user_id: str, session_data: str, date: str):
        self._sessions.append(
            {"user_id": user_id, "content": session_data, "date": date}
        )

    def add_sessions(self, user_id: str, sessions: list):
        for s in sessions:
            self._sessions.append(
                {"user_id": user_id, "content": s["content"], "date": s["date"]}
            )

    def query(self, user_id: str, query: str) -> str:
        # Simulate returning answer with stats
        self.last_query_stats = {
            "model": "gpt-4o-mini",
            "prompt_tokens": 150,
            "completion_tokens": 50,
            "retrieval": {
                "duration_ms": 120.5,
                "context_preview": "Some context from memory...",
                "vector_search": {
                    "seeds": [
                        {"node_id": "node_1", "content": "Fact 1", "score": 0.95},
                        {"node_id": "node_2", "content": "Fact 2", "score": 0.87},
                    ]
                },
            },
        }
        return "The answer based on memory"

    def reset(self, user_id: str):
        self._reset_called = True
        self._sessions = []


# === LegacyAdapterWrapper Tests ===


class TestLegacyAdapterWrapper:
    """Tests for LegacyAdapterWrapper."""

    def test_wrap_basic(self):
        """Should wrap legacy adapter with new interface."""
        legacy = MockLegacyAdapter()
        wrapper = LegacyAdapterWrapper(legacy, name="test_adapter")

        assert wrapper.name == "test_adapter"
        assert wrapper.capabilities is not None

    def test_capabilities_detection(self):
        """Should detect capabilities from legacy adapter."""
        legacy = MockLegacyAdapter()
        wrapper = LegacyAdapterWrapper(legacy)

        caps = wrapper.capabilities

        assert caps.supports_async is False  # Legacy adapters are sync
        assert caps.supports_bulk_ingest is True  # Has add_sessions
        assert caps.supports_reset is True  # Has reset

    def test_reset_delegation(self):
        """Should delegate reset to legacy adapter."""
        legacy = MockLegacyAdapter()
        wrapper = LegacyAdapterWrapper(legacy)

        wrapper.reset("user_123")

        assert legacy._reset_called is True

    def test_add_sessions_conversion(self):
        """Should convert Session objects to legacy format."""
        legacy = MockLegacyAdapter()
        wrapper = LegacyAdapterWrapper(legacy)

        sessions = [
            Session(content="Hello world", date="2025-01-15"),
            Session(content="Goodbye world", date="2025-01-16"),
        ]

        wrapper.add_sessions("user_123", sessions)

        assert len(legacy._sessions) == 2
        assert legacy._sessions[0]["content"] == "Hello world"
        assert legacy._sessions[0]["date"] == "2025-01-15"

    def test_query_returns_query_result(self):
        """Should convert legacy query response to QueryResult."""
        legacy = MockLegacyAdapter()
        wrapper = LegacyAdapterWrapper(legacy)

        result = wrapper.query("user_123", "What did I say?")

        assert isinstance(result, QueryResult)
        assert result.answer == "The answer based on memory"
        assert result.error is None

    def test_query_extracts_usage(self):
        """Should extract token usage from last_query_stats."""
        legacy = MockLegacyAdapter()
        wrapper = LegacyAdapterWrapper(legacy)

        result = wrapper.query("user_123", "Test query")

        assert result.usage.model == "gpt-4o-mini"
        assert result.usage.prompt_tokens == 150
        assert result.usage.completion_tokens == 50

    def test_query_extracts_latency(self):
        """Should extract latency from last_query_stats."""
        legacy = MockLegacyAdapter()
        wrapper = LegacyAdapterWrapper(legacy)

        result = wrapper.query("user_123", "Test query")

        assert result.latency.retrieval_ms == 120.5
        assert result.latency.total_ms is not None

    def test_query_extracts_context(self):
        """Should extract context preview from last_query_stats."""
        legacy = MockLegacyAdapter()
        wrapper = LegacyAdapterWrapper(legacy)

        result = wrapper.query("user_123", "Test query")

        assert result.context_text == "Some context from memory..."

    def test_query_extracts_retrieved_items(self):
        """Should extract retrieved items from last_query_stats."""
        legacy = MockLegacyAdapter()
        legacy.log_node_content = True  # Enable retrieval
        wrapper = LegacyAdapterWrapper(legacy)

        result = wrapper.query("user_123", "Test query")

        assert len(result.retrieved) >= 2
        assert result.retrieved[0].id == "node_1"
        assert result.retrieved[0].score == 0.95

    def test_query_handles_error(self):
        """Should handle query errors gracefully."""
        legacy = MockLegacyAdapter()
        legacy.query = MagicMock(side_effect=Exception("Connection failed"))
        wrapper = LegacyAdapterWrapper(legacy)

        result = wrapper.query("user_123", "Test query")

        assert result.answer == ""
        assert result.error == "Connection failed"

    @pytest.mark.asyncio
    async def test_async_methods(self):
        """Should provide async wrappers around sync methods."""
        legacy = MockLegacyAdapter()
        wrapper = LegacyAdapterWrapper(legacy)

        await wrapper.areset("user_123")
        assert legacy._reset_called is True

        await wrapper.aadd_sessions("user_456", [Session(content="async test")])
        assert len(legacy._sessions) == 1

        result = await wrapper.aquery("user_789", "async query")
        assert isinstance(result, QueryResult)


class TestWrapLegacyAdapterFactory:
    """Tests for wrap_legacy_adapter factory function."""

    def test_factory_creates_wrapper(self):
        """wrap_legacy_adapter should create LegacyAdapterWrapper."""
        legacy = MockLegacyAdapter()
        wrapper = wrap_legacy_adapter(legacy, name="my_adapter")

        assert isinstance(wrapper, LegacyAdapterWrapper)
        assert wrapper.name == "my_adapter"

    def test_factory_default_name(self):
        """Should use class name as default adapter name."""
        legacy = MockLegacyAdapter()
        wrapper = wrap_legacy_adapter(legacy)

        assert wrapper.name == "MockLegacyAdapter"
