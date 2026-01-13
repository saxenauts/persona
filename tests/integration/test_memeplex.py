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
            topics=["fitness", "career", "cooking"],
            people=["Sarah (wife)", "Max (colleague)"],
            projects=["Job Search", "Home Renovation"],
            places=["SF (home)", "Denver (hometown)"],
            concepts=["stoicism"],
            last_week_topics=["career", "interview prep"],
            last_month_topics=["fitness", "career", "cooking"],
            recent_focus="Interview prep for TechCorp - 3 rounds done",
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
        assert "fitness" in retrieved.topics
        assert "Sarah (wife)" in retrieved.people
        assert "Job Search" in retrieved.projects
        assert "career" in retrieved.last_week_topics
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
        assert memeplex.index == ""

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
            index="Initial index with topic1",
        )
        await store.save_memeplex(memeplex)

        memeplex.index = "Updated index with topic1, topic2, topic3 and Sarah (sister)"
        await store.save_memeplex(memeplex)

        retrieved = await store.get_memeplex(user_id)

        assert retrieved is not None
        assert "Updated index" in retrieved.index
        assert "Sarah (sister)" in retrieved.index

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
            event_time=now - timedelta(hours=2),
            user_id=user_id,
            session_id="session-1",
        )
        episode2 = EpisodeMemory(
            id=uuid4(),
            title="Evening meeting",
            content="Had team sync",
            event_time=now - timedelta(hours=1),
            user_id=user_id,
            session_id="session-2",
        )
        note1 = NoteMemory(
            id=uuid4(),
            title="Buy groceries",
            content="Milk, eggs, bread",
            event_time=now,
            user_id=user_id,
            status="active",
            session_id="session-2",
        )
        note2 = NoteMemory(
            id=uuid4(),
            title="Call mom",
            content="Birthday next week",
            event_time=now,
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
        topics=["fitness", "career", "cooking"],
        people=["Sarah (wife)", "Max (colleague)"],
        projects=["Job Search", "Home Renovation"],
        places=["SF (home)"],
        last_week_topics=["career", "interview"],
        last_month_topics=["fitness", "career"],
        recent_focus="Interview prep for TechCorp",
        memory_stats=MemoryStats(
            total_memories=100,
            total_entities=15,
            active_notes=5,
        ),
        timeline_summary="Active since Nov 2024, 47 sessions",
    )

    prompt = memeplex.to_system_prompt()

    assert "Your Knowledge" in prompt
    assert "Topics" in prompt
    assert "fitness" in prompt
    assert "Sarah (wife)" in prompt
    assert "Job Search" in prompt
    assert "Last week" in prompt
    assert "100 memories" in prompt

    print(f"\n✅ to_system_prompt() renders correctly:")
    print(prompt)


@pytest.mark.asyncio
async def test_persona_service_loads_memeplex():
    from persona.core.graph_ops import GraphOps
    from persona.services.persona_service import PersonaService
    from persona.llm.prompts import PERSONAL_AI_SYSTEM_PROMPT

    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    store = MemoryStore(graph_db)

    user_id = f"test-persona-memeplex-{uuid4()}"

    try:
        await graph_db.create_user(user_id)

        memeplex = Memeplex(
            user_id=user_id,
            topics=["testing", "integration"],
            people=["Test Person"],
            recent_focus="Testing memeplex integration",
            memory_stats=MemoryStats(total_memories=10),
        )
        await store.save_memeplex(memeplex)

        retrieved = await store.get_memeplex(user_id)
        assert retrieved is not None

        prompt_section = retrieved.to_system_prompt()
        assert "testing" in prompt_section
        assert "Test Person" in prompt_section

        formatted = PERSONAL_AI_SYSTEM_PROMPT.format(
            world_model=prompt_section, user_context=""
        )
        assert "Your Knowledge" in formatted
        assert "testing" in formatted

        print(f"\n✅ PersonaService memeplex integration works")
        print(f"   Topics in prompt: {len(retrieved.topics)}")

    finally:
        await graph_db.delete_user(user_id)
        await graph_db.close()
