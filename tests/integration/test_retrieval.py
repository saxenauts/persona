"""
Integration tests for v2 Memory Retrieval functions.

Tests using the fitness_test_v2 user data (47 episodes, 100 psyche, 72 goals).

Run: docker compose run --rm app poetry run pytest tests/integration/test_retrieval.py -v -s
"""

import pytest
import asyncio

from persona.core.memory_store import MemoryStore
from persona.core.backends.neo4j_graph import Neo4jGraphDatabase
from persona.core.rag_interface import RAGInterface


USER_ID = "fitness_test_v2"


# ========== Search Tests ==========

@pytest.mark.asyncio
async def test_search_text_finds_fitness():
    """
    Test: search_text("fitness") finds relevant episodes.
    """
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    memory_store = MemoryStore(graph_db)
    
    results = await memory_store.search_text(USER_ID, "fitness")
    
    print(f"\n📝 Text search for 'fitness': {len(results)} results")
    for r in results[:5]:
        print(f"   [{r.type}] {r.title}")
    
    await graph_db.close()
    assert len(results) > 0, "Should find memories mentioning 'fitness'"


@pytest.mark.asyncio
async def test_search_text_with_type_filter():
    """
    Test: search_text filtered by type.
    """
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    memory_store = MemoryStore(graph_db)
    
    episodes = await memory_store.search_text(USER_ID, "workout", types=["episode"])
    psyche = await memory_store.search_text(USER_ID, "workout", types=["psyche"])
    
    print(f"\n📝 Episodes with 'workout': {len(episodes)}")
    print(f"📝 Psyche with 'workout': {len(psyche)}")
    
    await graph_db.close()
    
    for e in episodes[:3]:
        assert e.type == "episode"
    for p in psyche[:3]:
        assert p.type == "psyche"


# ========== Query Tests ==========

@pytest.mark.asyncio
async def test_get_by_type():
    """
    Test: get_by_type returns correct types.
    """
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    memory_store = MemoryStore(graph_db)
    
    episodes = await memory_store.get_by_type("episode", USER_ID, limit=10)
    goals = await memory_store.get_by_type("goal", USER_ID, limit=10)
    psyche = await memory_store.get_by_type("psyche", USER_ID, limit=10)
    
    print(f"\n📊 By type:")
    print(f"   Episodes: {len(episodes)}")
    print(f"   Goals: {len(goals)}")
    print(f"   Psyche: {len(psyche)}")
    
    await graph_db.close()
    
    assert all(e.type == "episode" for e in episodes)
    assert all(g.type == "goal" for g in goals)
    assert all(p.type == "psyche" for p in psyche)


@pytest.mark.asyncio
async def test_get_recent():
    """
    Test: get_recent returns memories in reverse chronological order.
    """
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    memory_store = MemoryStore(graph_db)
    
    recent = await memory_store.get_recent(USER_ID, limit=5)
    
    print(f"\n📅 Recent memories:")
    for m in recent:
        print(f"   {m.timestamp}: [{m.type}] {m.title}")
    
    await graph_db.close()
    
    # Check descending order
    for i in range(len(recent) - 1):
        assert recent[i].timestamp >= recent[i+1].timestamp


# ========== Context Tests ==========

@pytest.mark.asyncio
async def test_get_user_context():
    """
    Test: get_user_context composes structured context.
    """
    rag = RAGInterface(USER_ID)
    
    context = await rag.get_user_context()
    
    print(f"\n📋 User Context ({len(context)} chars):")
    print("=" * 50)
    print(context[:1000])
    if len(context) > 1000:
        print("...")
    print("=" * 50)
    
    # Check sections exist
    assert "## Recent Context" in context or "## Your Goals" in context or "## About You" in context


@pytest.mark.asyncio
async def test_get_user_context_with_conversation():
    """
    Test: get_user_context includes current conversation.
    """
    rag = RAGInterface(USER_ID)
    current = "USER: How many fitness classes do I attend?\nASSISTANT: Let me check..."
    
    context = await rag.get_user_context(current_conversation=current)
    
    print(f"\n📋 Context with conversation:")
    print(context[-500:])
    
    assert "## Current Conversation" in context
    assert "fitness classes" in context


# ========== Goal Hierarchy Tests ==========

@pytest.mark.asyncio
async def test_get_goal_hierarchy():
    """
    Test: get_goal_hierarchy returns all goals.
    """
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    memory_store = MemoryStore(graph_db)
    
    goals = await memory_store.get_goal_hierarchy(USER_ID)
    
    print(f"\n🎯 Goal hierarchy: {len(goals)} goals")
    for g in goals[:5]:
        print(f"   {g.title}")
    
    await graph_db.close()
    assert len(goals) > 0


# ========== Summary ==========

@pytest.mark.asyncio
async def test_retrieval_summary():
    """
    Summary test showing all retrieval capabilities.
    """
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    memory_store = MemoryStore(graph_db)
    rag = RAGInterface(USER_ID)
    
    print("\n" + "=" * 60)
    print("RETRIEVAL SUMMARY for user: fitness_test_v2")
    print("=" * 60)
    
    # Counts
    episodes = await memory_store.get_by_type("episode", USER_ID, limit=100)
    goals = await memory_store.get_by_type("goal", USER_ID, limit=100)
    psyche = await memory_store.get_by_type("psyche", USER_ID, limit=100)
    
    print(f"\n📊 Memory Counts:")
    print(f"   Episodes: {len(episodes)}")
    print(f"   Goals: {len(goals)}")
    print(f"   Psyche: {len(psyche)}")
    
    # Search
    fitness_hits = await memory_store.search_text(USER_ID, "fitness")
    print(f"\n🔍 Text search 'fitness': {len(fitness_hits)} hits")
    
    # Context
    context = await rag.get_user_context()
    print(f"\n📋 User context: {len(context)} chars")
    
    await graph_db.close()
    
    print("\n" + "=" * 60)
    print("✅ All retrieval functions working!")
    print("=" * 60)
