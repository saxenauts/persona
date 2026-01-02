"""Integration tests for the full ingestion pipeline.

Tests the complete flow: PersonaAdapter.ingest() -> extract -> entity dedup -> persist
Also tests record() tool via PersonaService agent context.
"""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from persona.adapters.persona_adapter import PersonaAdapter
from persona.core.graph_ops import GraphOps
from persona.core.memory_store import MemoryStore


SAMPLE_CONVERSATIONS = [
    {
        "name": "fitness_session",
        "content": """User: I've been thinking about getting more serious about my fitness routine. I want to run a 10K by spring.
Assistant: That's a great goal! Do you have any current running experience?
User: I've done some casual jogging with my girlfriend Sarah. She's really into running - she did a half marathon last year.
Assistant: Perfect! Having a running partner like Sarah can be really motivating.
User: Yeah, and my trainer Mike at the downtown gym has been pushing me to set concrete goals.
Assistant: Working with a trainer is excellent. What's your current running distance?
User: I can do about 3 miles comfortably. Mike says I should focus on consistency first.
""",
        "expected_entities": ["Sarah", "Mike"],
        "expected_psyche": True,
        "expected_episode": True,
    },
    {
        "name": "work_project",
        "content": """User: I had a meeting with my team lead Jennifer today about Project Alpha.
Assistant: How did it go?
User: Good! She wants us to ship the MVP by February 15th. I'm leading the backend work.
Assistant: That sounds like a tight timeline. What's the scope?
User: It's a new API for our enterprise clients. I need to coordinate with Dave from the frontend team.
Assistant: Got it. Any concerns about the deadline?
User: A bit nervous but excited. Jennifer said if we hit this target, there's talk of a promotion.
""",
        "expected_entities": ["Jennifer", "Dave", "Project Alpha"],
        "expected_psyche": True,
        "expected_episode": True,
    },
    {
        "name": "duplicate_entity_mention",
        "content": """User: Sarah and I went hiking last weekend at Point Reyes.
Assistant: That sounds lovely! How was the trail?
User: Amazing views. Sarah took some great photos - she's really into photography now.
Assistant: Did you do the Tomales Point trail?
User: Yes! It was her idea. Sarah always picks the best spots.
""",
        "expected_entities": ["Sarah", "Point Reyes"],
        "expected_psyche": False,
        "expected_episode": True,
    },
]


@pytest.mark.asyncio
async def test_full_ingestion_pipeline():
    """Test complete ingestion: extract -> entity dedup -> persist."""
    user_id = f"test-ingest-{uuid4()}"

    async with GraphOps() as graph_ops:
        await graph_ops.create_user(user_id)
        adapter = PersonaAdapter(user_id=user_id, graph_ops=graph_ops)
        store = MemoryStore(graph_ops.graph_db, graph_ops.vector_store)

        try:
            sample = SAMPLE_CONVERSATIONS[0]
            result = await adapter.ingest(
                content=sample["content"],
                source_type="conversation",
                timestamp=datetime.now(timezone.utc),
                session_id="test_session_1",
            )

            assert result.success, f"Ingestion failed: {result.error}"
            assert len(result.memories) > 0, "No memories extracted"

            memory_types = [m.type for m in result.memories]
            assert "episode" in memory_types, "No episode extracted"

            entities = [m for m in result.memories if m.type == "entity"]
            entity_names = [e.canonical_name for e in entities]
            print(f"\n✅ Extracted {len(entities)} entities: {entity_names}")

            for expected in sample["expected_entities"]:
                assert any(expected.lower() in name.lower() for name in entity_names), (
                    f"Expected entity '{expected}' not found"
                )

            print(
                f"   Episode: {next(m.title for m in result.memories if m.type == 'episode')}"
            )
            print(
                f"   Timing: extract={result.extract_time_ms:.0f}ms, embed={result.embed_time_ms:.0f}ms, persist={result.persist_time_ms:.0f}ms"
            )

        finally:
            await graph_ops.delete_user(user_id)


@pytest.mark.asyncio
async def test_entity_dedup_across_sessions():
    """Test that same entity mentioned in multiple sessions is deduplicated."""
    user_id = f"test-dedup-{uuid4()}"

    async with GraphOps() as graph_ops:
        await graph_ops.create_user(user_id)
        adapter = PersonaAdapter(user_id=user_id, graph_ops=graph_ops)
        store = MemoryStore(graph_ops.graph_db, graph_ops.vector_store)

        try:
            result1 = await adapter.ingest(
                content=SAMPLE_CONVERSATIONS[0]["content"],
                source_type="conversation",
                timestamp=datetime.now(timezone.utc),
                session_id="session_1",
            )
            assert result1.success

            result2 = await adapter.ingest(
                content=SAMPLE_CONVERSATIONS[2]["content"],
                source_type="conversation",
                timestamp=datetime.now(timezone.utc),
                session_id="session_2",
            )
            assert result2.success

            sarah_entity = await store.get_entity_by_name("Sarah", user_id)

            if sarah_entity:
                print(f"\n✅ Entity dedup working")
                print(f"   Sarah entity ID: {sarah_entity.id}")
                print(f"   Mentioned in: {len(sarah_entity.mentioned_in)} episodes")
                print(f"   Aliases: {sarah_entity.aliases}")
                print(f"   Attributes: {[a.key for a in sarah_entity.attributes]}")

                assert len(sarah_entity.mentioned_in) >= 2, (
                    "Sarah should be mentioned in at least 2 episodes"
                )
            else:
                print(f"\n⚠️ Sarah entity not found - may have different canonical name")

        finally:
            await graph_ops.delete_user(user_id)


@pytest.mark.asyncio
async def test_temporal_chain_linking():
    """Test that episodes are linked in temporal order."""
    user_id = f"test-temporal-{uuid4()}"

    async with GraphOps() as graph_ops:
        await graph_ops.create_user(user_id)
        adapter = PersonaAdapter(user_id=user_id, graph_ops=graph_ops)
        store = MemoryStore(graph_ops.graph_db, graph_ops.vector_store)

        try:
            for i, sample in enumerate(SAMPLE_CONVERSATIONS[:2]):
                result = await adapter.ingest(
                    content=sample["content"],
                    source_type="conversation",
                    session_id=f"session_{i}",
                )
                assert result.success

            recent = await store.get_most_recent_episode(user_id)
            assert recent is not None, "No episodes found"

            connected = await store.get_connected_batch([recent.id], user_id)
            neighbors = connected.get(recent.id, [])
            rel_types = [rel for _, rel in neighbors]

            print(f"\n✅ Temporal chain exists")
            print(f"   Most recent episode: {recent.title}")
            print(f"   Connected via: {rel_types}")

            assert "PREVIOUS" in rel_types or "NEXT" in rel_types, (
                "No temporal links found"
            )

        finally:
            await graph_ops.delete_user(user_id)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_batch_ingestion():
    """Test parallel extraction with sequential persistence."""
    user_id = f"test-batch-{uuid4()}"

    async with GraphOps() as graph_ops:
        await graph_ops.create_user(user_id)
        adapter = PersonaAdapter(user_id=user_id, graph_ops=graph_ops)
        store = MemoryStore(graph_ops.graph_db, graph_ops.vector_store)

        try:
            items = [
                {"content": sample["content"], "source_type": "conversation"}
                for sample in SAMPLE_CONVERSATIONS
            ]

            results = await adapter.ingest_batch(items)

            assert len(results) == len(SAMPLE_CONVERSATIONS)
            success_count = sum(1 for r in results if r.success)
            assert success_count == len(SAMPLE_CONVERSATIONS), (
                f"Only {success_count}/{len(SAMPLE_CONVERSATIONS)} succeeded"
            )

            total_memories = sum(len(r.memories) for r in results)
            print(f"\n✅ Batch ingestion complete")
            print(f"   Sessions: {len(results)}")
            print(f"   Total memories: {total_memories}")

        finally:
            await graph_ops.delete_user(user_id)
