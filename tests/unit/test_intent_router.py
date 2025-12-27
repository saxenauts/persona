"""
Unit tests for IntentRouter.

Tests the UserCard-based routing logic without LLM calls.
"""

import pytest
from datetime import date, timedelta

from persona.core.intent_router import IntentRouter, RetrievalHints, RetrievalMode
from persona.models.memory import UserCard


class TestIntentRouterBasics:
    @pytest.fixture
    def router(self):
        return IntentRouter()

    @pytest.fixture
    def empty_user_card(self):
        return UserCard(user_id="test_user")

    @pytest.fixture
    def rich_user_card(self):
        return UserCard(
            user_id="test_user",
            keyword_hints={
                "fitness": ["mem_1", "mem_2"],
                "workout": ["mem_1", "mem_3"],
                "sarah": ["mem_4"],
            },
            pinned_memories={
                "fitness_baseline": ["mem_5"],
                "career_goals": ["mem_6", "mem_7"],
            },
            temporal_anchors={
                "wedding": "2020-06-15",
                "job_start": "2023-01-10",
                "trip_to_japan": "2024-03-01:2024-03-15",
            },
            entity_aliases={
                "my coach": "Jordan",
                "the move": "Austin relocation",
                "wifey": "Sarah",
            },
            dominant_memory_types={"episode": 0.6, "psyche": 0.3, "note": 0.1},
            dominant_link_types={"PREVIOUS": 0.5, "DERIVED_FROM": 0.3},
            key_relationships=["partner Sarah", "coach Jordan"],
            current_focus=["marathon training", "career transition"],
        )

    @pytest.mark.asyncio
    async def test_route_with_empty_card(self, router, empty_user_card):
        hints = await router.route("What did I eat yesterday?", empty_user_card)

        assert isinstance(hints, RetrievalHints)
        assert hints.mode == RetrievalMode.FAST
        assert hints.resolved_query == "What did I eat yesterday?"

    @pytest.mark.asyncio
    async def test_route_returns_retrieval_hints(self, router, rich_user_card):
        hints = await router.route("How is my fitness?", rich_user_card)

        assert isinstance(hints, RetrievalHints)
        assert hints.confidence >= 0.0
        assert hints.confidence <= 1.0


class TestAliasResolution:
    @pytest.fixture
    def router(self):
        return IntentRouter()

    @pytest.fixture
    def user_card(self):
        return UserCard(
            user_id="test",
            entity_aliases={
                "my coach": "Jordan",
                "the move": "Austin relocation",
                "wifey": "Sarah",
            },
        )

    @pytest.mark.asyncio
    async def test_alias_resolution(self, router, user_card):
        hints = await router.route("What did my coach say?", user_card)

        assert "Jordan" in hints.resolved_query
        assert "my coach" not in hints.resolved_query

    @pytest.mark.asyncio
    async def test_multiple_alias_resolution(self, router, user_card):
        hints = await router.route(
            "When was the move and what did my coach say?", user_card
        )

        assert "Jordan" in hints.resolved_query
        assert "Austin relocation" in hints.resolved_query

    @pytest.mark.asyncio
    async def test_case_insensitive_alias(self, router, user_card):
        hints = await router.route("What did MY COACH recommend?", user_card)

        assert "Jordan" in hints.resolved_query


class TestKeywordExtraction:
    @pytest.fixture
    def router(self):
        return IntentRouter()

    @pytest.fixture
    def user_card(self):
        return UserCard(
            user_id="test",
            keyword_hints={
                "fitness": ["mem_1", "mem_2"],
                "workout": ["mem_1", "mem_3"],
            },
            current_focus=["marathon training"],
            key_relationships=["partner Sarah"],
        )

    @pytest.mark.asyncio
    async def test_keyword_extraction_from_hints(self, router, user_card):
        hints = await router.route("How is my fitness going?", user_card)

        assert "fitness" in hints.search_keywords

    @pytest.mark.asyncio
    async def test_keyword_extraction_from_focus(self, router, user_card):
        hints = await router.route("Tell me about marathon training", user_card)

        assert "marathon training" in hints.search_keywords

    @pytest.mark.asyncio
    async def test_keyword_extraction_from_relationships(self, router, user_card):
        hints = await router.route("What did Sarah say?", user_card)

        assert any("Sarah" in kw or "partner" in kw for kw in hints.search_keywords)


class TestSeedMemoryDiscovery:
    @pytest.fixture
    def router(self):
        return IntentRouter()

    @pytest.fixture
    def user_card(self):
        return UserCard(
            user_id="test",
            keyword_hints={
                "fitness": ["mem_1", "mem_2"],
                "workout": ["mem_1", "mem_3"],
            },
            pinned_memories={
                "fitness_baseline": ["mem_5"],
            },
        )

    @pytest.mark.asyncio
    async def test_seed_memory_from_keyword(self, router, user_card):
        hints = await router.route("How is my fitness?", user_card)

        assert "mem_1" in hints.seed_memory_ids or "mem_2" in hints.seed_memory_ids

    @pytest.mark.asyncio
    async def test_seed_memory_from_pinned(self, router, user_card):
        hints = await router.route("fitness_baseline stats", user_card)

        assert "mem_5" in hints.seed_memory_ids


class TestTemporalResolution:
    @pytest.fixture
    def router(self):
        return IntentRouter()

    @pytest.fixture
    def user_card(self):
        return UserCard(
            user_id="test",
            temporal_anchors={
                "wedding": "2020-06-15",
                "job_start": "2023-01-10",
                "trip_to_japan": "2024-03-01:2024-03-15",
            },
        )

    @pytest.mark.asyncio
    async def test_yesterday_resolution(self, router, user_card):
        current = date(2025, 12, 26)
        hints = await router.route("What happened yesterday?", user_card, current)

        assert hints.date_range is not None
        assert hints.date_range[0] == date(2025, 12, 25)
        assert hints.date_range[1] == date(2025, 12, 25)

    @pytest.mark.asyncio
    async def test_last_week_resolution(self, router, user_card):
        current = date(2025, 12, 26)
        hints = await router.route("What happened last week?", user_card, current)

        assert hints.date_range is not None
        assert hints.date_range[0] == date(2025, 12, 19)

    @pytest.mark.asyncio
    async def test_named_anchor_resolution(self, router, user_card):
        hints = await router.route("What happened around the wedding?", user_card)

        assert hints.date_range is not None

    @pytest.mark.asyncio
    async def test_date_range_anchor(self, router, user_card):
        hints = await router.route("Tell me about the trip_to_japan", user_card)

        assert hints.date_range is not None
        assert hints.date_range[0] == date(2024, 3, 1)
        assert hints.date_range[1] == date(2024, 3, 15)


class TestMemoryTypeBoosts:
    @pytest.fixture
    def router(self):
        return IntentRouter()

    @pytest.fixture
    def user_card(self):
        return UserCard(
            user_id="test",
            dominant_memory_types={"episode": 0.6, "psyche": 0.3, "note": 0.1},
        )

    @pytest.mark.asyncio
    async def test_task_query_boosts_notes(self, router, user_card):
        hints = await router.route("What tasks should I do today?", user_card)

        assert "note" in hints.memory_type_boost

    @pytest.mark.asyncio
    async def test_identity_query_boosts_psyche(self, router, user_card):
        hints = await router.route("Who am I?", user_card)

        assert "psyche" in hints.memory_type_boost

    @pytest.mark.asyncio
    async def test_event_query_boosts_episode(self, router, user_card):
        hints = await router.route("What happened yesterday?", user_card)

        assert "episode" in hints.memory_type_boost

    @pytest.mark.asyncio
    async def test_no_signal_uses_dominant(self, router, user_card):
        hints = await router.route("Tell me something interesting", user_card)

        assert len(hints.memory_type_boost) > 0


class TestModeSelection:
    @pytest.fixture
    def router(self):
        return IntentRouter()

    @pytest.fixture
    def rich_user_card(self):
        return UserCard(
            user_id="test",
            keyword_hints={"fitness": ["mem_1", "mem_2"]},
            pinned_memories={"fitness_baseline": ["mem_5"]},
        )

    @pytest.fixture
    def empty_user_card(self):
        return UserCard(user_id="test")

    @pytest.mark.asyncio
    async def test_simple_query_fast_mode(self, router, rich_user_card):
        hints = await router.route("How is my fitness?", rich_user_card)

        assert hints.mode == RetrievalMode.FAST

    @pytest.mark.asyncio
    async def test_complex_query_lower_confidence(self, router, empty_user_card):
        hints = await router.route(
            "Compare my fitness progress over the last month with my diet changes and summarize the patterns",
            empty_user_card,
        )

        assert hints.confidence < 0.5

    @pytest.mark.asyncio
    async def test_seeds_increase_confidence(self, router, rich_user_card):
        hints_rich = await router.route("fitness progress", rich_user_card)
        hints_empty = await router.route("fitness progress", UserCard(user_id="test"))

        assert hints_rich.confidence > hints_empty.confidence

    @pytest.mark.asyncio
    async def test_long_query_decreases_confidence(self, router, rich_user_card):
        short_query = "How is fitness?"
        long_query = " ".join(["How is fitness?"] * 10)

        hints_short = await router.route(short_query, rich_user_card)
        hints_long = await router.route(long_query, rich_user_card)

        assert hints_short.confidence > hints_long.confidence


class TestRetrievalHintsModel:
    def test_default_values(self):
        hints = RetrievalHints()

        assert hints.mode == RetrievalMode.FAST
        assert hints.search_keywords == []
        assert hints.seed_memory_ids == []
        assert hints.top_k == 5
        assert hints.hop_depth == 1
        assert hints.date_range is None

    def test_confidence_bounds(self):
        hints = RetrievalHints(confidence=0.8)
        assert hints.confidence == 0.8

        with pytest.raises(ValueError):
            RetrievalHints(confidence=1.5)

        with pytest.raises(ValueError):
            RetrievalHints(confidence=-0.1)
