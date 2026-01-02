"""
Integration tests for Memeplex storage and retrieval.

Tests real Neo4j operations - requires running Neo4j instance.

Run: docker compose run --rm test poetry run pytest tests/integration/test_memeplex.py -v -s
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from persona.core.memory_store import MemoryStore
from persona.core.backends.neo4j_graph import Neo4jGraphDatabase
from persona.models.memory import (
    Memeplex,
    ActiveMemory,
    MemoryStats,
    EpisodeMemory,
    NoteMemory,
)


@pytest.mark.asyncio
async def test_memeplex_create_and_retrieve():
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    store = MemoryStore(graph_db)

    user_id = f"test-memeplex-{uuid4()}"

    try:
        await graph_db.create_user(user_id)

        memeplex = Memeplex(
            user_id=user_id,
            active_memories=[
                ActiveMemory(
                    memory_id=uuid4(),
                    memory_type="note",
                    title="Job search at TechCorp",
                    keywords=["techcorp", "interview", "salary"],
                    context_snippet="3 rounds done, waiting on offer",
                )
            ],
            last_session_summary="Discussed interview prep and salary expectations.",
            recent_keywords=["interview", "techcorp", "remote", "offer"],
            memory_stats=MemoryStats(
                total_memories=50,
                total_episodes=30,
                total_notes=10,
                total_entities=10,
                active_notes=3,
            ),
            timeline_summary="Active since Dec 2024, 12 sessions",
        )

        saved = await store.save_memeplex(memeplex)
        assert saved.user_id == user_id

        retrieved = await store.get_memeplex(user_id)

        assert retrieved is not None
        assert retrieved.user_id == user_id
        assert len(retrieved.active_memories) == 1
        assert retrieved.active_memories[0].title == "Job search at TechCorp"
        assert "techcorp" in retrieved.active_memories[0].keywords
        assert (
            retrieved.last_session_summary
            == "Discussed interview prep and salary expectations."
        )
        assert "interview" in retrieved.recent_keywords
        assert retrieved.memory_stats.total_memories == 50
        assert retrieved.timeline_summary == "Active since Dec 2024, 12 sessions"

        print(f"\n✅ Memeplex CRUD works for user {user_id}")

    finally:
        await graph_db.delete_user(user_id)
        await graph_db.close()


@pytest.mark.asyncio
async def test_memeplex_get_or_create():
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    store = MemoryStore(graph_db)

    user_id = f"test-memeplex-new-{uuid4()}"

    try:
        await graph_db.create_user(user_id)

        result = await store.get_memeplex(user_id)
        assert result is None

        memeplex = await store.get_or_create_memeplex(user_id)
        assert memeplex is not None
        assert memeplex.user_id == user_id
        assert memeplex.active_memories == []

        memeplex_again = await store.get_or_create_memeplex(user_id)
        assert memeplex_again is not None
        assert memeplex_again.user_id == user_id

        print(f"\n✅ get_or_create_memeplex works for user {user_id}")

    finally:
        await graph_db.delete_user(user_id)
        await graph_db.close()


@pytest.mark.asyncio
async def test_memeplex_update():
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    store = MemoryStore(graph_db)

    user_id = f"test-memeplex-update-{uuid4()}"

    try:
        await graph_db.create_user(user_id)

        memeplex = Memeplex(
            user_id=user_id,
            last_session_summary="Initial summary",
            recent_keywords=["keyword1"],
        )
        await store.save_memeplex(memeplex)

        memeplex.last_session_summary = "Updated summary after new session"
        memeplex.recent_keywords = ["keyword1", "keyword2", "keyword3"]
        memeplex.active_memories.append(
            ActiveMemory(
                memory_id=uuid4(),
                memory_type="entity",
                title="Sarah",
                keywords=["sister", "NYC"],
                context_snippet="Planning wedding",
            )
        )
        await store.save_memeplex(memeplex)

        retrieved = await store.get_memeplex(user_id)

        assert retrieved is not None
        assert retrieved.last_session_summary == "Updated summary after new session"
        assert len(retrieved.recent_keywords) == 3
        assert len(retrieved.active_memories) == 1
        assert retrieved.active_memories[0].title == "Sarah"

        print(f"\n✅ Memeplex update works for user {user_id}")

    finally:
        await graph_db.delete_user(user_id)
        await graph_db.close()


@pytest.mark.asyncio
async def test_compute_memory_stats():
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    store = MemoryStore(graph_db)

    user_id = f"test-stats-{uuid4()}"

    try:
        await graph_db.create_user(user_id)

        now = datetime.utcnow()

        episode1 = EpisodeMemory(
            id=uuid4(),
            title="Morning workout",
            content="Did a 30 minute run",
            timestamp=now - timedelta(hours=2),
            user_id=user_id,
            session_id="session-1",
        )
        episode2 = EpisodeMemory(
            id=uuid4(),
            title="Evening meeting",
            content="Had team sync",
            timestamp=now - timedelta(hours=1),
            user_id=user_id,
            session_id="session-2",
        )
        note1 = NoteMemory(
            id=uuid4(),
            title="Buy groceries",
            content="Milk, eggs, bread",
            timestamp=now,
            user_id=user_id,
            status="active",
            session_id="session-2",
        )
        note2 = NoteMemory(
            id=uuid4(),
            title="Call mom",
            content="Birthday next week",
            timestamp=now,
            user_id=user_id,
            status="done",
            session_id="session-2",
        )

        await store.create(episode1)
        await store.create(episode2)
        await store.create(note1)
        await store.create(note2)

        stats = await store.compute_memory_stats(user_id)

        assert stats.total_memories == 4
        assert stats.total_episodes == 2
        assert stats.total_notes == 2
        assert stats.active_notes == 1
        assert stats.session_count == 2
        assert stats.earliest_memory is not None
        assert stats.latest_memory is not None

        print(f"\n✅ compute_memory_stats works")
        print(f"   Total: {stats.total_memories}, Episodes: {stats.total_episodes}")
        print(f"   Notes: {stats.total_notes}, Active: {stats.active_notes}")
        print(f"   Sessions: {stats.session_count}")

    finally:
        await graph_db.delete_user(user_id)
        await graph_db.close()


@pytest.mark.asyncio
async def test_memeplex_to_system_prompt():
    memeplex = Memeplex(
        user_id="test-user",
        active_memories=[
            ActiveMemory(
                memory_id=uuid4(),
                memory_type="note",
                title="Job search",
                keywords=["interview", "techcorp", "remote"],
                context_snippet="3 rounds done",
            ),
            ActiveMemory(
                memory_id=uuid4(),
                memory_type="entity",
                title="Sarah",
                keywords=["sister", "wedding"],
                context_snippet="Planning June wedding",
            ),
        ],
        last_session_summary="Discussed interview and wedding planning.",
        recent_keywords=["interview", "wedding", "techcorp"],
        memory_stats=MemoryStats(
            total_memories=100,
            total_entities=15,
            active_notes=5,
        ),
        timeline_summary="Active since Nov 2024, 47 sessions",
    )

    prompt = memeplex.to_system_prompt()

    assert "<memeplex>" in prompt
    assert "</memeplex>" in prompt
    assert "ACTIVE NOW:" in prompt
    assert "Job search" in prompt
    assert "Sarah" in prompt
    assert "interview" in prompt
    assert "RECENT:" in prompt
    assert "MEMORY OVERVIEW:" in prompt
    assert "100 memories" in prompt
    assert "recall()" in prompt

    print(f"\n✅ to_system_prompt() renders correctly:")
    print(prompt)


@pytest.mark.asyncio
async def test_persona_service_loads_memeplex():
    from persona.core.graph_ops import GraphOps
    from persona.services.persona_service import PersonaService, AGENT_SYSTEM_PROMPT

    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    store = MemoryStore(graph_db)

    user_id = f"test-persona-memeplex-{uuid4()}"

    try:
        await graph_db.create_user(user_id)

        memeplex = Memeplex(
            user_id=user_id,
            active_memories=[
                ActiveMemory(
                    memory_id=uuid4(),
                    memory_type="note",
                    title="Test active note",
                    keywords=["testing", "integration"],
                    context_snippet="Testing memeplex integration",
                )
            ],
            last_session_summary="This is a test session.",
            recent_keywords=["test", "memeplex"],
            memory_stats=MemoryStats(total_memories=10),
        )
        await store.save_memeplex(memeplex)

        retrieved = await store.get_memeplex(user_id)
        assert retrieved is not None

        prompt_section = retrieved.to_system_prompt()
        assert "Test active note" in prompt_section
        assert "testing, integration" in prompt_section

        formatted = AGENT_SYSTEM_PROMPT.format(memeplex_context=prompt_section)
        assert "<memeplex>" in formatted
        assert "Test active note" in formatted
        assert "## Available Tools" in formatted

        print(f"\n✅ PersonaService memeplex integration works")
        print(f"   Active memories in prompt: {len(retrieved.active_memories)}")

    finally:
        await graph_db.delete_user(user_id)
        await graph_db.close()
