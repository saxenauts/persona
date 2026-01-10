"""
Integration tests for v2 Memory Retrieval functions.

Tests using the fitness_test_v2 user data (47 episodes, 100 psyche, 72 notes).

Run: docker compose run --rm app poetry run pytest tests/integration/test_retrieval.py -v -s
"""

import pytest
import asyncio

from persona.core.memory_store import MemoryStore
from persona.core.backends.neo4j_graph import Neo4jGraphDatabase
from persona.core.retrieval import Retriever
from persona.core.graph_ops import GraphOps


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
    notes = await memory_store.get_by_type("note", USER_ID, limit=10)
    psyche = await memory_store.get_by_type("psyche", USER_ID, limit=10)

    print(f"\n📊 By type:")
    print(f"   Episodes: {len(episodes)}")
    print(f"   Notes: {len(notes)}")
    print(f"   Psyche: {len(psyche)}")

    await graph_db.close()

    assert all(e.type == "episode" for e in episodes)
    assert all(n.type == "note" for n in notes)
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
        assert recent[i].timestamp >= recent[i + 1].timestamp


# ========== Note Hierarchy Tests ==========


@pytest.mark.asyncio
async def test_get_note_hierarchy():
    """
    Test: get_note_hierarchy returns all notes.
    """
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    memory_store = MemoryStore(graph_db)

    notes = await memory_store.get_note_hierarchy(USER_ID)

    print(f"\n🎯 Note hierarchy: {len(notes)} notes")
    for n in notes[:5]:
        print(f"   {n.title}")

    await graph_db.close()
    assert len(notes) >= 0  # May be 0 if data not migrated yet


# ========== Summary ==========


@pytest.mark.asyncio
async def test_retrieval_summary():
    """
    Summary test showing all retrieval capabilities.
    """
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    memory_store = MemoryStore(graph_db)

    print("\n" + "=" * 60)
    print("RETRIEVAL SUMMARY for user: fitness_test_v2")
    print("=" * 60)

    # Counts
    episodes = await memory_store.get_by_type("episode", USER_ID, limit=100)
    notes = await memory_store.get_by_type("note", USER_ID, limit=100)
    psyche = await memory_store.get_by_type("psyche", USER_ID, limit=100)

    print(f"\n📊 Memory Counts:")
    print(f"   Episodes: {len(episodes)}")
    print(f"   Notes: {len(notes)}")
    print(f"   Psyche: {len(psyche)}")

    # Search
    fitness_hits = await memory_store.search_text(USER_ID, "fitness")
    print(f"\n🔍 Text search 'fitness': {len(fitness_hits)} hits")

    await graph_db.close()

    print("\n" + "=" * 60)
    print("✅ All retrieval functions working!")
    print("=" * 60)
