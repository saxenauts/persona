"""
Unit tests for query expansion module.

Tests the fallback rule-based expansion (no LLM calls).
"""

import pytest
from datetime import date, timedelta

from persona.core.query_expansion import (
    _fallback_expansion,
    date_range_to_cypher_filter,
    DateRange,
    QueryExpansion,
)


class TestFallbackExpansion:
    """Test rule-based fallback expansion."""

    def test_yesterday_detection(self):
        """Test 'yesterday' is parsed correctly."""
        current = date(2025, 12, 26)
        result = _fallback_expansion("What did I eat yesterday?", current)

        assert result.date_range is not None
        assert result.date_range.start == date(2025, 12, 25)
        assert result.date_range.end == date(2025, 12, 25)

    def test_last_week_detection(self):
        """Test 'last week' is parsed correctly."""
        current = date(2025, 12, 26)
        result = _fallback_expansion("What happened last week?", current)

        assert result.date_range is not None
        assert result.date_range.start == date(2025, 12, 19)
        assert result.date_range.end == date(2025, 12, 26)

    def test_past_week_detection(self):
        """Test 'past week' is parsed correctly."""
        current = date(2025, 12, 26)
        result = _fallback_expansion("Tell me about the past week", current)

        assert result.date_range is not None
        assert result.date_range.start == current - timedelta(days=7)
        assert result.date_range.end == current

    def test_last_month_detection(self):
        """Test 'last month' is parsed correctly."""
        current = date(2025, 12, 26)
        result = _fallback_expansion("What happened last month?", current)

        assert result.date_range is not None
        assert result.date_range.start == current - timedelta(days=30)
        assert result.date_range.end == current

    def test_today_detection(self):
        """Test 'today' is parsed correctly."""
        current = date(2025, 12, 26)
        result = _fallback_expansion("What did I do today?", current)

        assert result.date_range is not None
        assert result.date_range.start == current
        assert result.date_range.end == current

    def test_no_temporal_reference(self):
        """Test queries without temporal references return None date_range."""
        current = date(2025, 12, 26)
        result = _fallback_expansion("Who is my best friend?", current)

        assert result.date_range is None
        assert result.original_query == "Who is my best friend?"
        assert result.semantic_query == "Who is my best friend?"

    def test_case_insensitive(self):
        """Test temporal detection is case-insensitive."""
        current = date(2025, 12, 26)
        result = _fallback_expansion("What happened YESTERDAY?", current)

        assert result.date_range is not None
        assert result.date_range.start == date(2025, 12, 25)


class TestDateRangeToCypher:
    """Test Cypher filter generation."""

    def test_single_day_range(self):
        """Test single-day range generates correct Cypher."""
        dr = DateRange(start=date(2025, 12, 25), end=date(2025, 12, 25))
        cypher = date_range_to_cypher_filter(dr)

        assert "m.timestamp >= datetime('2025-12-25T00:00:00')" in cypher
        assert "m.timestamp <= datetime('2025-12-25T23:59:59" in cypher

    def test_multi_day_range(self):
        """Test multi-day range generates correct Cypher."""
        dr = DateRange(start=date(2025, 12, 19), end=date(2025, 12, 26))
        cypher = date_range_to_cypher_filter(dr)

        assert "2025-12-19" in cypher
        assert "2025-12-26" in cypher

    def test_custom_property_name(self):
        """Test custom property name is used."""
        dr = DateRange(start=date(2025, 12, 25), end=date(2025, 12, 25))
        cypher = date_range_to_cypher_filter(dr, property_name="created_at")

        assert "m.created_at >=" in cypher
        assert "m.created_at <=" in cypher


class TestQueryExpansionModel:
    """Test QueryExpansion Pydantic model."""

    def test_model_defaults(self):
        """Test model has correct defaults."""
        expansion = QueryExpansion(original_query="test")

        assert expansion.date_range is None
        assert expansion.entities == []
        assert expansion.relationship_threads == []
        assert expansion.semantic_query == ""

    def test_model_with_date_range(self):
        """Test model accepts DateRange."""
        dr = DateRange(start=date(2025, 12, 19), end=date(2025, 12, 26))
        expansion = QueryExpansion(
            original_query="What happened last week?",
            date_range=dr,
            semantic_query="What happened",
        )

        assert expansion.date_range == dr
        assert expansion.original_query == "What happened last week?"
        assert expansion.semantic_query == "What happened"
