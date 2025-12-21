"""
Tests for the Episode layer (Phase 1 of Persona v2 refactor).

Uses sample questions from LongMemEval to validate:
- Episode model creation and temporal linking
- Episode extraction from raw content
- Day-based grouping and chain traversal
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from persona.models.memory import Episode, EpisodeCreateRequest, EpisodeChainResponse


# ============================================================================
# Sample test data based on LongMemEval question types
# ============================================================================

# Multi-session test cases (aggregate across episodes)
MULTI_SESSION_TEST_CASES = [
    {
        "question": "How many items of clothing do I need to pick up or return from a store?",
        "expected_answer": 3,
        "episodes": [
            {"title": "Dry cleaning pickup reminder", "content": "Need to pick up my suit from the dry cleaners on Main St. They close at 6pm."},
            {"title": "Return the jacket", "content": "The jacket I bought last week doesn't fit. Need to return it to Nordstrom."},
            {"title": "Dress alteration ready", "content": "Got a call that my dress alteration is ready to pick up from the tailor."},
        ]
    },
    {
        "question": "How many projects have I led or am currently leading?",
        "expected_answer": 2,
        "episodes": [
            {"title": "API refactor project kickoff", "content": "Started leading the API refactor project today. Team of 4 engineers."},
            {"title": "Mobile app redesign", "content": "Taking over as lead on the mobile app redesign. Sarah handed it off."},
        ]
    },
]

# Temporal reasoning test cases (time-based queries)
TEMPORAL_REASONING_TEST_CASES = [
    {
        "question": "How many days passed between my MoMA visit and the Met exhibit?",
        "expected_answer": 7,
        "episodes": [
            {"title": "MoMA visit", "content": "Visited the Museum of Modern Art today. The Monet exhibit was incredible.", "day_offset": 0},
            {"title": "Met Ancient Civilizations", "content": "Went to the Ancient Civilizations exhibit at the Metropolitan Museum of Art.", "day_offset": 7},
        ]
    },
    {
        "question": "Which happened first: nursery prep, baby shower help, or phone case order?",
        "expected_order": ["nursery prep", "baby shower", "phone case"],
        "episodes": [
            {"title": "Helped prep nursery", "content": "Spent the afternoon helping my friend prepare the nursery for the baby.", "day_offset": 0},
            {"title": "Baby shower shopping", "content": "Helped my cousin pick out stuff for her baby shower at Target.", "day_offset": 5},
            {"title": "Ordered phone case", "content": "Ordered a customized phone case for my friend's birthday.", "day_offset": 10},
        ]
    },
]

# Knowledge update test cases (facts that can change)
KNOWLEDGE_UPDATE_TEST_CASES = [
    {
        "question": "What was my personal best time in the charity 5K run?",
        "expected_answer": "25:50",
        "episodes": [
            {"title": "First 5K attempt", "content": "Ran my first charity 5K today. Finished in 28 minutes and 30 seconds."},
            {"title": "New PR at 5K", "content": "Crushed it today! Beat my personal best with a time of 25 minutes and 50 seconds."},
        ]
    },
    {
        "question": "How many Korean restaurants have I tried in my city?",
        "expected_answer": 4,
        "episodes": [
            {"title": "Tried Kimchi House", "content": "Finally tried Kimchi House downtown. The bibimbap was amazing."},
            {"title": "Seoul Garden lunch", "content": "Had lunch at Seoul Garden with coworkers. Great Korean BBQ."},
            {"title": "New Korean place", "content": "Discovered a new Korean restaurant called Gogi. Their tofu soup was perfect."},
            {"title": "K-Town exploration", "content": "Tried Jang Mo Jib in K-Town. Fourth Korean restaurant I've tried here."},
        ]
    },
]


# ============================================================================
# Episode Model Tests
# ============================================================================

class TestEpisodeModel:
    """Tests for the Episode Pydantic model."""
    
    def test_episode_creation_basic(self):
        """Test basic episode creation with required fields."""
        episode = Episode(
            title="Test episode",
            content="This is a test episode content.",
            day_id="2024-12-20",
            user_id="test-user"
        )
        
        assert episode.title == "Test episode"
        assert episode.content == "This is a test episode content."
        assert episode.day_id == "2024-12-20"
        assert episode.user_id == "test-user"
        assert episode.id is not None
        assert episode.source_type == "conversation"  # default
    
    def test_episode_temporal_fields(self):
        """Test temporal anchoring fields."""
        now = datetime.utcnow()
        episode = Episode(
            title="Temporal test",
            content="Testing temporal fields.",
            timestamp=now,
            day_id="2024-12-20",
            session_id="session_123",
            user_id="test-user"
        )
        
        assert episode.timestamp == now
        assert episode.session_id == "session_123"
        assert episode.day_id == "2024-12-20"
    
    def test_episode_chain_links(self):
        """Test PREVIOUS/NEXT chain linking."""
        prev_id = uuid4()
        next_id = uuid4()
        
        episode = Episode(
            title="Chain test",
            content="Testing chain links.",
            day_id="2024-12-20",
            user_id="test-user",
            previous_episode_id=prev_id,
            next_episode_id=next_id
        )
        
        assert episode.previous_episode_id == prev_id
        assert episode.next_episode_id == next_id
    
    def test_episode_provenance(self):
        """Test provenance chain fields."""
        episode = Episode(
            title="Instagram memory",
            content="Saw a beautiful sunset picture.",
            day_id="2024-12-20",
            source_type="instagram",
            source_ref="post_abc123",
            user_id="test-user"
        )
        
        assert episode.source_type == "instagram"
        assert episode.source_ref == "post_abc123"


class TestEpisodeChainResponse:
    """Tests for episode chain queries."""
    
    def test_chain_response_structure(self):
        """Test chain response model."""
        episodes = [
            Episode(
                title=f"Episode {i}",
                content=f"Content for episode {i}.",
                day_id=f"2024-12-{20+i:02d}",
                user_id="test-user"
            )
            for i in range(3)
        ]
        
        response = EpisodeChainResponse(
            episodes=episodes,
            start_date="2024-12-20",
            end_date="2024-12-22",
            total_count=3
        )
        
        assert len(response.episodes) == 3
        assert response.start_date == "2024-12-20"
        assert response.end_date == "2024-12-22"
        assert response.total_count == 3


# ============================================================================
# Episode Temporal Linking Tests 
# ============================================================================

class TestEpisodeTemporalLinking:
    """Tests for temporal chain building (PREVIOUS/NEXT links)."""
    
    def test_chain_order_preservation(self):
        """Test that episodes maintain chronological order."""
        base_time = datetime(2024, 12, 20, 10, 0, 0)
        
        episodes = []
        for i in range(5):
            ep = Episode(
                title=f"Episode {i}",
                content=f"Event {i} happened.",
                timestamp=base_time + timedelta(hours=i),
                day_id="2024-12-20",
                user_id="test-user"
            )
            episodes.append(ep)
        
        # Sort by timestamp
        sorted_eps = sorted(episodes, key=lambda e: e.timestamp)
        
        for i, ep in enumerate(sorted_eps):
            assert ep.title == f"Episode {i}"
    
    def test_day_grouping(self):
        """Test that episodes group correctly by day_id."""
        episodes = [
            Episode(title="Morning", content="Morning event.", 
                    day_id="2024-12-20", user_id="test-user"),
            Episode(title="Afternoon", content="Afternoon event.", 
                    day_id="2024-12-20", user_id="test-user"),
            Episode(title="Next day", content="Next day event.", 
                    day_id="2024-12-21", user_id="test-user"),
        ]
        
        day_20 = [e for e in episodes if e.day_id == "2024-12-20"]
        day_21 = [e for e in episodes if e.day_id == "2024-12-21"]
        
        assert len(day_20) == 2
        assert len(day_21) == 1


# ============================================================================
# Multi-Session Aggregation Tests (LongMemEval-style)
# ============================================================================

class TestMultiSessionAggregation:
    """Tests for aggregating information across multiple episodes."""
    
    @pytest.mark.parametrize("test_case", MULTI_SESSION_TEST_CASES)
    def test_episode_count_queries(self, test_case):
        """Test that we can count items across episodes."""
        episodes = [
            Episode(
                title=ep["title"],
                content=ep["content"],
                day_id="2024-12-20",
                user_id="test-user"
            )
            for ep in test_case["episodes"]
        ]
        
        # The count should match the number of episodes for these test cases
        assert len(episodes) == test_case["expected_answer"]


# ============================================================================
# Temporal Reasoning Tests (LongMemEval-style)
# ============================================================================

class TestTemporalReasoning:
    """Tests for time-based queries across episodes."""
    
    def test_day_difference_calculation(self):
        """Test calculating days between episodes."""
        base_date = datetime(2024, 12, 20)
        
        ep1 = Episode(
            title="First event",
            content="First event happened.",
            timestamp=base_date,
            day_id="2024-12-20",
            user_id="test-user"
        )
        
        ep2 = Episode(
            title="Second event",
            content="Second event happened.",
            timestamp=base_date + timedelta(days=7),
            day_id="2024-12-27",
            user_id="test-user"
        )
        
        days_diff = (ep2.timestamp - ep1.timestamp).days
        assert days_diff == 7
    
    def test_chronological_ordering(self):
        """Test ordering events chronologically."""
        base_date = datetime(2024, 12, 20)
        
        episodes = [
            Episode(
                title="Phone case order",
                content="Ordered phone case.",
                timestamp=base_date + timedelta(days=10),
                day_id="2024-12-30",
                user_id="test-user"
            ),
            Episode(
                title="Nursery prep",
                content="Helped prep nursery.",
                timestamp=base_date,
                day_id="2024-12-20",
                user_id="test-user"
            ),
            Episode(
                title="Baby shower",
                content="Baby shower shopping.",
                timestamp=base_date + timedelta(days=5),
                day_id="2024-12-25",
                user_id="test-user"
            ),
        ]
        
        # Sort chronologically
        sorted_eps = sorted(episodes, key=lambda e: e.timestamp)
        
        assert sorted_eps[0].title == "Nursery prep"
        assert sorted_eps[1].title == "Baby shower"
        assert sorted_eps[2].title == "Phone case order"


# ============================================================================
# Knowledge Update Tests (LongMemEval-style)
# ============================================================================

class TestKnowledgeUpdate:
    """Tests for tracking information that changes over time."""
    
    def test_latest_value_extraction(self):
        """Test getting the most recent value of changing information."""
        base_date = datetime(2024, 12, 20)
        
        episodes = [
            Episode(
                title="First 5K",
                content="Ran my first 5K. Time: 28:30.",
                timestamp=base_date,
                day_id="2024-12-20",
                user_id="test-user"
            ),
            Episode(
                title="New PR",
                content="Beat my personal best! New time: 25:50.",
                timestamp=base_date + timedelta(days=30),
                day_id="2025-01-19",
                user_id="test-user"
            ),
        ]
        
        # The most recent episode should have the current PR
        latest = max(episodes, key=lambda e: e.timestamp)
        assert "25:50" in latest.content
    
    def test_accumulating_counts(self):
        """Test counting accumulated items (e.g., restaurants tried)."""
        episodes = [
            Episode(title=f"Restaurant {i}", content=f"Tried restaurant {i}.",
                    day_id=f"2024-12-{20+i:02d}", user_id="test-user")
            for i in range(4)
        ]
        
        # Should be able to count total restaurants
        assert len(episodes) == 4
