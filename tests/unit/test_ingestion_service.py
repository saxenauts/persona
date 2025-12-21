"""
Tests for the unified memory ingestion service.

Tests the extraction of Episode, Psyche, and Goals from raw input.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock
import json

from persona.models.episode import Episode
from persona.models.psyche import Psyche
from persona.models.goal import Goal
from persona.services.ingestion_service import (
    MemoryIngestionService,
    IngestionResult,
    IngestionOutput,
    EpisodeOutput,
    PsycheOutput,
    GoalOutput,
    ingest_memory
)


# ============================================================================
# Mock LLM Response Fixtures
# ============================================================================

@pytest.fixture
def mock_chat_client():
    """Mock chat client that returns structured JSON."""
    mock = AsyncMock()
    return mock


@pytest.fixture
def mock_embedding_client():
    """Mock embedding client."""
    mock = AsyncMock()
    mock.embed = AsyncMock(return_value=[[0.1] * 1536])  # Default single embedding
    return mock


# ============================================================================
# Test Cases Based on Real Use Cases
# ============================================================================

class TestIngestionOutput:
    """Test the output models."""
    
    def test_ingestion_output_with_all_types(self):
        """Test creating output with episode, psyche, and goals."""
        output = IngestionOutput(
            episode=EpisodeOutput(
                title="Coffee with Sam",
                content="Met Sam for coffee..."
            ),
            psyche=[
                PsycheOutput(type="value", content="I care about helping friends")
            ],
            goals=[
                GoalOutput(type="task", title="Intro Sam to Maya", content="", status="active")
            ]
        )
        
        assert output.episode.title == "Coffee with Sam"
        assert len(output.psyche) == 1
        assert len(output.goals) == 1
    
    def test_ingestion_output_episode_only(self):
        """Test output with only episode (most common case)."""
        output = IngestionOutput(
            episode=EpisodeOutput(
                title="Morning standup",
                content="Quick sync with the team..."
            )
        )
        
        assert output.episode.title == "Morning standup"
        assert output.psyche == []
        assert output.goals == []


class TestMemoryIngestionService:
    """Test the ingestion service."""
    
    @pytest.mark.asyncio
    async def test_ingest_simple_conversation(self, mock_chat_client, mock_embedding_client):
        """Test ingesting a simple conversation."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "episode": {
                "title": "Weather chat",
                "content": "Had a brief conversation about the weather being nice today."
            },
            "psyche": [],
            "goals": []
        })
        mock_chat_client.chat = AsyncMock(return_value=mock_response)
        
        with patch('persona.services.ingestion_service.get_chat_client', return_value=mock_chat_client), \
             patch('persona.services.ingestion_service.get_embedding_client', return_value=mock_embedding_client):
            
            service = MemoryIngestionService()
            result = await service.ingest(
                raw_content="The weather is really nice today!",
                user_id="test-user"
            )
            
            assert result.success
            assert result.episode.title == "Weather chat"
            assert len(result.psyche) == 0
            assert len(result.goals) == 0
    
    @pytest.mark.asyncio
    async def test_ingest_with_task(self, mock_chat_client, mock_embedding_client):
        """Test ingesting content with an action item."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "episode": {
                "title": "Errand planning",
                "content": "Need to pick up groceries on the way home."
            },
            "psyche": [],
            "goals": [
                {"type": "task", "title": "Buy groceries", "content": "On the way home", "status": "active"}
            ]
        })
        mock_chat_client.chat = AsyncMock(return_value=mock_response)
        mock_embedding_client.embed = AsyncMock(return_value=[[0.1] * 1536, [0.2] * 1536])
        
        with patch('persona.services.ingestion_service.get_chat_client', return_value=mock_chat_client), \
             patch('persona.services.ingestion_service.get_embedding_client', return_value=mock_embedding_client):
            
            service = MemoryIngestionService()
            result = await service.ingest(
                raw_content="Remind me to pick up groceries on the way home",
                user_id="test-user"
            )
            
            assert result.success
            assert len(result.goals) == 1
            assert result.goals[0].title == "Buy groceries"
            assert result.goals[0].source_episode_id == result.episode.id
    
    @pytest.mark.asyncio
    async def test_ingest_with_psyche(self, mock_chat_client, mock_embedding_client):
        """Test ingesting content with identity information."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "episode": {
                "title": "Diet change decision",
                "content": "Decided to go vegetarian after watching a documentary about animal welfare."
            },
            "psyche": [
                {"type": "preference", "content": "I'm vegetarian now"}
            ],
            "goals": []
        })
        mock_chat_client.chat = AsyncMock(return_value=mock_response)
        mock_embedding_client.embed = AsyncMock(return_value=[[0.1] * 1536, [0.2] * 1536])
        
        with patch('persona.services.ingestion_service.get_chat_client', return_value=mock_chat_client), \
             patch('persona.services.ingestion_service.get_embedding_client', return_value=mock_embedding_client):
            
            service = MemoryIngestionService()
            result = await service.ingest(
                raw_content="I've decided to go vegetarian. Watched a documentary about animal welfare and it really moved me.",
                user_id="test-user"
            )
            
            assert result.success
            assert len(result.psyche) == 1
            assert result.psyche[0].content == "I'm vegetarian now"
            assert result.psyche[0].type == "preference"
    
    @pytest.mark.asyncio
    async def test_ingest_complex_input(self, mock_chat_client, mock_embedding_client):
        """Test ingesting complex input with all three types."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "episode": {
                "title": "Coffee catch-up with Sam about funding",
                "content": "Met Sam for coffee. He's stressed about his Series A - investors pushing back. Offered to intro him to Maya."
            },
            "psyche": [
                {"type": "value", "content": "I care about helping friends through tough times"}
            ],
            "goals": [
                {"type": "task", "title": "Intro Sam to Maya", "content": "For fundraising advice", "status": "active"}
            ]
        })
        mock_chat_client.chat = AsyncMock(return_value=mock_response)
        mock_embedding_client.embed = AsyncMock(return_value=[[0.1] * 1536, [0.2] * 1536, [0.3] * 1536])
        
        with patch('persona.services.ingestion_service.get_chat_client', return_value=mock_chat_client), \
             patch('persona.services.ingestion_service.get_embedding_client', return_value=mock_embedding_client):
            
            service = MemoryIngestionService()
            result = await service.ingest(
                raw_content="Met Sam for coffee at Blue Bottle. He's really stressed about his Series A - investors are pushing back on valuation. I offered to intro him to Maya who helped me navigate similar talks.",
                user_id="test-user"
            )
            
            assert result.success
            assert result.episode.title == "Coffee catch-up with Sam about funding"
            assert len(result.psyche) == 1
            assert len(result.goals) == 1
            
            # Check linking
            assert result.psyche[0].source_episode_id == result.episode.id
            assert result.goals[0].source_episode_id == result.episode.id


class TestEmbeddingGeneration:
    """Test embedding generation during ingestion."""
    
    @pytest.mark.asyncio
    async def test_embeddings_generated_for_all_items(self, mock_chat_client, mock_embedding_client):
        """Test that embeddings are generated for episode, psyche, and goals."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "episode": {"title": "Test", "content": "Test content"},
            "psyche": [{"type": "trait", "content": "Likes coffee"}],
            "goals": [{"type": "task", "title": "Do thing", "content": "", "status": "active"}]
        })
        mock_chat_client.chat = AsyncMock(return_value=mock_response)
        
        # Return 3 embeddings (1 episode + 1 psyche + 1 goal)
        mock_embedding_client.embed = AsyncMock(return_value=[
            [0.1] * 1536,
            [0.2] * 1536,
            [0.3] * 1536
        ])
        
        with patch('persona.services.ingestion_service.get_chat_client', return_value=mock_chat_client), \
             patch('persona.services.ingestion_service.get_embedding_client', return_value=mock_embedding_client):
            
            service = MemoryIngestionService()
            result = await service.ingest("Test input", "test-user")
            
            # Verify embed was called with all texts
            mock_embedding_client.embed.assert_called_once()
            texts = mock_embedding_client.embed.call_args[0][0]
            assert len(texts) == 3  # episode + psyche + goal
            
            # Verify embeddings were assigned
            assert result.episode.embedding is not None
            assert result.psyche[0].embedding is not None
            assert result.goals[0].embedding is not None


class TestTemporalAnchoring:
    """Test timestamp and temporal fields."""
    
    @pytest.mark.asyncio
    async def test_timestamp_preserved(self, mock_chat_client, mock_embedding_client):
        """Test that provided timestamp is used."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "episode": {"title": "Old memory", "content": "Something from last week"},
            "psyche": [],
            "goals": []
        })
        mock_chat_client.chat = AsyncMock(return_value=mock_response)
        
        with patch('persona.services.ingestion_service.get_chat_client', return_value=mock_chat_client), \
             patch('persona.services.ingestion_service.get_embedding_client', return_value=mock_embedding_client):
            
            custom_timestamp = datetime(2024, 12, 15, 10, 30, 0)
            
            service = MemoryIngestionService()
            result = await service.ingest(
                raw_content="Something from last week",
                user_id="test-user",
                timestamp=custom_timestamp
            )
            
            assert result.episode.timestamp == custom_timestamp
            assert result.episode.day_id == "2024-12-15"
    
    @pytest.mark.asyncio
    async def test_created_at_is_now(self, mock_chat_client, mock_embedding_client):
        """Test that created_at is always current time."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "episode": {"title": "Test", "content": "Test"},
            "psyche": [],
            "goals": []
        })
        mock_chat_client.chat = AsyncMock(return_value=mock_response)
        
        with patch('persona.services.ingestion_service.get_chat_client', return_value=mock_chat_client), \
             patch('persona.services.ingestion_service.get_embedding_client', return_value=mock_embedding_client):
            
            old_timestamp = datetime(2020, 1, 1)
            
            service = MemoryIngestionService()
            result = await service.ingest(
                raw_content="Old content",
                user_id="test-user",
                timestamp=old_timestamp  # Event happened in 2020
            )
            
            # timestamp is when it happened (2020)
            assert result.episode.timestamp.year == 2020
            # created_at is now (2024 or later)
            assert result.episode.created_at.year >= 2024
