"""
Tests for the unified memory model and ingestion.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, patch, MagicMock
import json

from persona.models.memory import Memory, MemoryLink, IngestionOutput, EpisodeOutput, PsycheOutput, GoalOutput
from persona.services.ingestion_service import MemoryIngestionService, IngestionResult


# ============================================================================
# Memory Model Tests
# ============================================================================

class TestMemoryModel:
    """Tests for the unified Memory model."""
    
    def test_create_episode_memory(self):
        """Test creating an episode memory."""
        memory = Memory(
            type="episode",
            title="Coffee with Sam",
            content="Met Sam for coffee to discuss his startup.",
            user_id="test-user"
        )
        
        assert memory.type == "episode"
        assert memory.title == "Coffee with Sam"
        assert memory.id is not None
        assert memory.user_id == "test-user"
    
    def test_create_psyche_memory(self):
        """Test creating a psyche memory."""
        memory = Memory(
            type="psyche",
            title="preference",
            content="Prefers remote work",
            user_id="test-user"
        )
        
        assert memory.type == "psyche"
        assert memory.content == "Prefers remote work"
    
    def test_create_goal_memory(self):
        """Test creating a goal memory."""
        memory = Memory(
            type="goal",
            title="Buy groceries",
            content="Pick up milk and eggs",
            status="active",
            due_date=datetime(2024, 12, 25),
            user_id="test-user"
        )
        
        assert memory.type == "goal"
        assert memory.status == "active"
        assert memory.due_date is not None
    
    def test_memory_timestamps(self):
        """Test timestamp fields."""
        old_time = datetime(2024, 1, 1)
        memory = Memory(
            type="episode",
            title="Old memory",
            content="From last year",
            timestamp=old_time,
            user_id="test-user"
        )
        
        assert memory.timestamp == old_time
        assert memory.created_at > old_time  # created_at is now


class TestMemoryLink:
    """Tests for memory links."""
    
    def test_create_link(self):
        """Test creating a link between memories."""
        link = MemoryLink(
            source_id=uuid4(),
            target_id=uuid4(),
            relation="derived_from"
        )
        
        assert link.relation == "derived_from"
    
    def test_temporal_link(self):
        """Test PREVIOUS/NEXT links."""
        link = MemoryLink(
            source_id=uuid4(),
            target_id=uuid4(),
            relation="PREVIOUS"
        )
        
        assert link.relation == "PREVIOUS"


# ============================================================================
# Ingestion Tests
# ============================================================================

class TestMemoryIngestion:
    """Tests for memory ingestion service."""
    
    @pytest.fixture
    def mock_chat_client(self):
        mock = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_embedding_client(self):
        mock = AsyncMock()
        mock.embed = AsyncMock(return_value=[[0.1] * 1536])
        return mock
    
    @pytest.mark.asyncio
    async def test_ingest_simple(self, mock_chat_client, mock_embedding_client):
        """Test simple ingestion."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "episode": {"title": "Weather chat", "content": "Talked about weather."},
            "psyche": [],
            "goals": []
        })
        mock_chat_client.chat = AsyncMock(return_value=mock_response)
        
        with patch('persona.services.ingestion_service.get_chat_client', return_value=mock_chat_client), \
             patch('persona.services.ingestion_service.get_embedding_client', return_value=mock_embedding_client):
            
            service = MemoryIngestionService()
            result = await service.ingest("Nice weather today!", "test-user")
            
            assert result.success
            assert len(result.memories) == 1
            assert result.memories[0].type == "episode"
    
    @pytest.mark.asyncio
    async def test_ingest_with_all_types(self, mock_chat_client, mock_embedding_client):
        """Test ingestion with episode, psyche, and goal."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "episode": {"title": "Career chat", "content": "Discussed remote work."},
            "psyche": [{"type": "preference", "content": "Prefers remote work"}],
            "goals": [{"type": "task", "title": "Update resume", "content": "", "status": "active"}]
        })
        mock_chat_client.chat = AsyncMock(return_value=mock_response)
        mock_embedding_client.embed = AsyncMock(return_value=[[0.1] * 1536, [0.2] * 1536, [0.3] * 1536])
        
        with patch('persona.services.ingestion_service.get_chat_client', return_value=mock_chat_client), \
             patch('persona.services.ingestion_service.get_embedding_client', return_value=mock_embedding_client):
            
            service = MemoryIngestionService()
            result = await service.ingest("I prefer working from home. Need to update my resume.", "test-user")
            
            assert result.success
            assert len(result.memories) == 3
            
            types = [m.type for m in result.memories]
            assert "episode" in types
            assert "psyche" in types
            assert "goal" in types
            
            # Check links
            assert len(result.links) == 2  # psyche and goal link to episode
    
    @pytest.mark.asyncio
    async def test_ingest_embeddings(self, mock_chat_client, mock_embedding_client):
        """Test that embeddings are generated."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "episode": {"title": "Test", "content": "Test content."},
            "psyche": [],
            "goals": []
        })
        mock_chat_client.chat = AsyncMock(return_value=mock_response)
        mock_embedding_client.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        
        with patch('persona.services.ingestion_service.get_chat_client', return_value=mock_chat_client), \
             patch('persona.services.ingestion_service.get_embedding_client', return_value=mock_embedding_client):
            
            service = MemoryIngestionService()
            result = await service.ingest("Test", "test-user")
            
            assert result.memories[0].embedding is not None
            mock_embedding_client.embed.assert_called_once()
