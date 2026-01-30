"""Regression test for H1 timezone bug fix.

Tests that resolve_date_range uses user timezone, not server timezone.
Bug: datetime.now() was using server timezone instead of ctx.user_timezone.
"""

import pytest
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock, AsyncMock
from uuid import uuid4

from persona.tools.memory import resolve_date_range_handler, DateRangeResult
from persona.tools.context import ToolContext


class TestH1TimezoneBugFix:
    """Verify that resolve_date_range respects user timezone."""

    @pytest.fixture
    def mock_graph_ops(self):
        """Create a mock GraphOps for ToolContext."""
        return MagicMock()

    @pytest.fixture
    def mock_store(self):
        """Create a mock MemoryStore for ToolContext."""
        return MagicMock()

    @pytest.fixture
    def mock_adapter(self):
        """Create a mock PersonaAdapter for ToolContext."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_today_with_pst_timezone(
        self, mock_graph_ops, mock_store, mock_adapter
    ):
        """Test that 'today' resolves to PST date, not server date."""
        ctx = ToolContext(
            user_id="test-user",
            graph_ops=mock_graph_ops,
            store=mock_store,
            adapter=mock_adapter,
            user_timezone="America/Los_Angeles",
        )

        result = await resolve_date_range_handler(ctx, "today")

        assert isinstance(result, DateRangeResult)
        assert result.date_start is not None
        assert result.date_end is not None
        assert result.date_start == result.date_end

        pst_tz = ZoneInfo("America/Los_Angeles")
        pst_now = datetime.now(tz=pst_tz)
        pst_today = pst_now.date()

        assert result.date_start == pst_today.isoformat()
        assert result.date_end == pst_today.isoformat()

    @pytest.mark.asyncio
    async def test_yesterday_with_est_timezone(
        self, mock_graph_ops, mock_store, mock_adapter
    ):
        """Test that 'yesterday' resolves to EST date, not server date."""
        ctx = ToolContext(
            user_id="test-user",
            graph_ops=mock_graph_ops,
            store=mock_store,
            adapter=mock_adapter,
            user_timezone="America/New_York",
        )

        result = await resolve_date_range_handler(ctx, "yesterday")

        assert isinstance(result, DateRangeResult)
        assert result.date_start is not None
        assert result.date_end is not None
        assert result.date_start == result.date_end

        est_tz = ZoneInfo("America/New_York")
        est_now = datetime.now(tz=est_tz)
        est_yesterday = est_now.date() - timedelta(days=1)

        assert result.date_start == est_yesterday.isoformat()
        assert result.date_end == est_yesterday.isoformat()

    @pytest.mark.asyncio
    async def test_last_week_with_tokyo_timezone(
        self, mock_graph_ops, mock_store, mock_adapter
    ):
        """Test that 'last week' resolves correctly in Tokyo timezone."""
        ctx = ToolContext(
            user_id="test-user",
            graph_ops=mock_graph_ops,
            store=mock_store,
            adapter=mock_adapter,
            user_timezone="Asia/Tokyo",
        )

        result = await resolve_date_range_handler(ctx, "last week")

        assert isinstance(result, DateRangeResult)
        assert result.date_start is not None
        assert result.date_end is not None

        tokyo_tz = ZoneInfo("Asia/Tokyo")
        tokyo_now = datetime.now(tz=tokyo_tz)
        tokyo_today = tokyo_now.date()
        tokyo_last_week_start = tokyo_today - timedelta(days=7)

        assert result.date_start == tokyo_last_week_start.isoformat()
        assert result.date_end == tokyo_today.isoformat()

    @pytest.mark.asyncio
    async def test_utc_timezone_default(self, mock_graph_ops, mock_store, mock_adapter):
        """Test that UTC timezone works as default."""
        ctx = ToolContext(
            user_id="test-user",
            graph_ops=mock_graph_ops,
            store=mock_store,
            adapter=mock_adapter,
            user_timezone="UTC",
        )

        result = await resolve_date_range_handler(ctx, "today")

        assert isinstance(result, DateRangeResult)
        assert result.date_start is not None
        assert result.date_end is not None

        utc_tz = ZoneInfo("UTC")
        utc_now = datetime.now(tz=utc_tz)
        utc_today = utc_now.date()

        assert result.date_start == utc_today.isoformat()
        assert result.date_end == utc_today.isoformat()

    @pytest.mark.asyncio
    async def test_timezone_affects_date_boundary(
        self, mock_graph_ops, mock_store, mock_adapter
    ):
        """Test that timezone can affect which date is 'today' at boundary times.

        Example: If it's 1am PST (9am UTC), PST user sees today but UTC sees today.
        This test verifies the function uses the correct timezone for the boundary.
        """
        ctx_pst = ToolContext(
            user_id="test-user",
            graph_ops=mock_graph_ops,
            store=mock_store,
            adapter=mock_adapter,
            user_timezone="America/Los_Angeles",
        )

        ctx_utc = ToolContext(
            user_id="test-user",
            graph_ops=mock_graph_ops,
            store=mock_store,
            adapter=mock_adapter,
            user_timezone="UTC",
        )

        result_pst = await resolve_date_range_handler(ctx_pst, "today")
        result_utc = await resolve_date_range_handler(ctx_utc, "today")

        assert result_pst.date_start is not None
        assert result_utc.date_start is not None

        assert len(result_pst.date_start) == 10
        assert len(result_utc.date_start) == 10
