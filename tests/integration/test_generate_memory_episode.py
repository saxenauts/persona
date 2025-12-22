"""
Integration test for generate_memory_episode using real longmemeval data.

Run with: docker compose run --rm app poetry run pytest tests/integration/test_generate_memory_episode.py -v -s
"""

import asyncio
import pytest
from datetime import datetime
from uuid import UUID, uuid4

from persona.services.ingestion_service import MemoryIngestionService, IngestionResult
from persona.core.memory_store import MemoryStore
from persona.core.backends.neo4j_graph import Neo4jGraphDatabase
from persona.models.memory import Memory, MemoryLink


# ============================================================================
# Test Data (from longmemeval benchmark)
# ============================================================================

LONGMEMEVAL_CONVERSATION_1 = """User: The farmer needs to transport a fox, a chicken, and some grain across a river using a boat. Can you help?

Assistant: To solve this puzzle, the farmer can follow these steps:
1. First, take the chicken across the river.
2. Go back and take the fox across.
3. Bring the chicken back.
4. Take the grain across.
5. Finally, go back for the chicken."""

# A second conversation about goals/tasks
LONGMEMEVAL_CONVERSATION_2 = """User: I need to finish reviewing the quarterly reports by Friday. Also, I started a new diet - trying to cut carbs and eat more protein. Can you remind me about these?
Assistant: Of course! I've noted down two important items for you:
1. Quarterly reports review - due Friday
2. New diet plan - low carb, high protein

I'll remind you about both!"""


# ============================================================================
# The generate_memory_episode function (wires ingestion to persistence)
# ============================================================================

async def generate_memory_episode(
    raw_content: str,
    user_id: str,
    session_id: str = None,
    timestamp: datetime = None,
    source_type: str = "conversation"
) -> IngestionResult:
    """
    Full pipeline: Extract memories via LLM -> Generate embeddings -> Persist to Neo4j.
    
    This is the main entry point for the Memory Engine.
    """
    # Initialize services
    ingestion_service = MemoryIngestionService()
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    
    # Ensure user exists
    if not await graph_db.user_exists(user_id):
        await graph_db.create_user(user_id)
    
    memory_store = MemoryStore(graph_db)
    
    # Step 1: Extract and create Memory objects
    result = await ingestion_service.ingest(
        raw_content=raw_content,
        user_id=user_id,
        session_id=session_id,
        timestamp=timestamp or datetime.utcnow(),
        source_type=source_type
    )
    
    if not result.success:
        return result
    
    # Step 2: Persist all memories
    for memory in result.memories:
        # Find links for this memory
        memory_links = [l for l in result.links if l.source_id == memory.id]
        await memory_store.create(memory, links=memory_links)
    
    # Step 3: Link episodes in temporal chain
    episode = next((m for m in result.memories if m.type == "episode"), None)
    if episode:
        previous = await memory_store.get_most_recent_episode(user_id)
        if previous and previous.id != episode.id:
            await memory_store.link_temporal_chain(episode, previous)
    
    await graph_db.close()
    return result


# ============================================================================
# Tests
# ============================================================================

TEST_USER_ID = "test_user_longmemeval"


@pytest.mark.asyncio
async def test_generate_memory_episode_basic():
    """Test basic ingestion and persistence of a conversation."""
    result = await generate_memory_episode(
        raw_content=LONGMEMEVAL_CONVERSATION_1,
        user_id=TEST_USER_ID,
        session_id="session_1"
    )
    
    assert result.success, f"Ingestion failed: {result.error}"
    assert len(result.memories) >= 1, "Should have at least 1 memory (episode)"
    
    # Check episode was created
    episode = next((m for m in result.memories if m.type == "episode"), None)
    assert episode is not None, "Episode should be created"
    assert episode.embedding is not None, "Episode should have embedding"
    
    print(f"\n✅ Created {len(result.memories)} memories:")
    for m in result.memories:
        print(f"  - [{m.type}] {m.title}: {m.content[:50]}...")


@pytest.mark.asyncio
async def test_generate_memory_episode_with_goals():
    """Test that goals/tasks are extracted from conversation."""
    result = await generate_memory_episode(
        raw_content=LONGMEMEVAL_CONVERSATION_2,
        user_id=TEST_USER_ID,
        session_id="session_2"
    )
    
    assert result.success, f"Ingestion failed: {result.error}"
    
    # Should have episode + potentially goals
    episode = next((m for m in result.memories if m.type == "episode"), None)
    goals = [m for m in result.memories if m.type == "goal"]
    
    print(f"\n✅ Created {len(result.memories)} memories:")
    print(f"  Episode: {episode.title if episode else 'None'}")
    print(f"  Goals: {len(goals)}")
    for g in goals:
        print(f"    - {g.title}")


@pytest.mark.asyncio  
async def test_temporal_chain_linking():
    """Test that multiple episodes are linked in temporal chain."""
    # Create first episode
    result1 = await generate_memory_episode(
        raw_content="Morning standup: discussed API refactoring progress.",
        user_id=TEST_USER_ID,
        session_id="chain_test_1"
    )
    assert result1.success
    
    # Create second episode
    result2 = await generate_memory_episode(
        raw_content="Afternoon: finished the refactoring and deployed to staging.",
        user_id=TEST_USER_ID,
        session_id="chain_test_2"
    )
    assert result2.success
    
    print("\n✅ Created 2 episodes with temporal chain linking")


if __name__ == "__main__":
    asyncio.run(test_generate_memory_episode_basic())
