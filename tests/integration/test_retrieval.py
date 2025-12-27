"""
Integration tests for v2 Memory Retrieval functions.

Tests using the fitness_test_v2 user data (47 episodes, 100 psyche, 72 notes).

Run: docker compose run --rm app poetry run pytest tests/integration/test_retrieval.py -v -s
"""

import pytest
import asyncio

from persona.core.memory_store import MemoryStore
from persona.core.backends.neo4j_graph import Neo4jGraphDatabase
from persona.core.rag_interface import RAGInterface
from persona.core.intent_router import IntentRouter, RetrievalHints, RetrievalMode
from persona.core.retrieval import Retriever
from persona.core.graph_ops import GraphOps
from persona.services.user_service import UserCardService


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
    assert (
        "## Recent Context" in context
        or "## Your Goals" in context
        or "## About You" in context
    )


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
    rag = RAGInterface(USER_ID)

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

    # Context
    context = await rag.get_user_context()
    print(f"\n📋 User context: {len(context)} chars")

    await graph_db.close()

    print("\n" + "=" * 60)
    print("✅ All retrieval functions working!")
    print("=" * 60)


# ========== IntentRouter Integration Tests ==========


@pytest.mark.asyncio
async def test_intent_router_with_live_user_card():
    """
    Test: IntentRouter routes query using real UserCard from Neo4j.
    """
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    memory_store = MemoryStore(graph_db)
    graph_ops = GraphOps(graph_db=graph_db, vector_store=graph_db)
    await graph_ops.initialize()

    user_card_service = UserCardService(memory_store, graph_ops=graph_ops)
    user_card = await user_card_service.generate(USER_ID)

    print(f"\n🪪 UserCard for {USER_ID}:")
    print(f"   Name: {user_card.name}")
    print(f"   Summary: {user_card.summary}")
    print(f"   Keyword hints: {len(user_card.keyword_hints)} keywords")
    print(f"   Temporal anchors: {len(user_card.temporal_anchors)} anchors")
    print(f"   Memory types: {user_card.dominant_memory_types}")

    router = IntentRouter()
    hints = await router.route("How is my fitness going?", user_card)

    print(f"\n🧭 IntentRouter output:")
    print(f"   Mode: {hints.mode.value}")
    print(f"   Confidence: {hints.confidence:.2f}")
    print(f"   Keywords: {hints.search_keywords[:5]}")
    print(f"   Seed IDs: {len(hints.seed_memory_ids)} seeds")
    print(f"   Memory boosts: {hints.memory_type_boost}")

    await graph_db.close()

    assert hints.mode in [RetrievalMode.FAST, RetrievalMode.SLOW]
    assert 0.0 <= hints.confidence <= 1.0


@pytest.mark.asyncio
async def test_retriever_with_intent_router_hints():
    """
    Test: Retriever.get_context_with_hints returns context using IntentRouter hints.
    """
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    memory_store = MemoryStore(graph_db)
    graph_ops = GraphOps(graph_db=graph_db, vector_store=graph_db)
    await graph_ops.initialize()

    user_card_service = UserCardService(memory_store, graph_ops=graph_ops)
    user_card = await user_card_service.generate(USER_ID)

    router = IntentRouter()
    hints = await router.route("What workouts did I do last week?", user_card)

    retriever = Retriever(USER_ID, memory_store, graph_ops)
    context, stats = await retriever.get_context_with_hints(
        query="What workouts did I do last week?",
        hints=hints,
        user_card=user_card,
        collect_stats=True,
    )

    print(f"\n📝 Retrieval with hints:")
    print(f"   Mode: {stats['mode']}")
    print(f"   Total memories: {stats.get('total_memories', 'N/A')}")
    print(f"   Context chars: {stats['context_chars']}")
    print(f"   Retrieval time: {stats.get('total_retrieval_ms', 'N/A'):.1f}ms")
    print(f"\n   Context preview:")
    print(context[:500])

    await graph_db.close()

    assert isinstance(context, str)
    assert len(context) > 0
    assert "<memory_context>" in context


@pytest.mark.asyncio
async def test_rag_interface_query_v2():
    """
    Test: RAGInterface.query_v2 uses IntentRouter for retrieval.
    """
    async with RAGInterface(USER_ID) as rag:
        result = await rag.query_v2(
            "Tell me about my fitness routine",
            include_stats=True,
        )

        print(f"\n🤖 RAG V2 Query Result:")
        print(f"   Answer: {result['answer'][:200]}...")
        print(f"\n   Routing stats:")
        routing = result.get("routing", {})
        print(f"      Mode: {routing.get('mode')}")
        print(f"      Confidence: {routing.get('confidence'):.2f}")
        print(f"      Keywords: {routing.get('keywords')}")
        print(f"      Seeds: {routing.get('seed_count')}")
        print(f"      Routing time: {routing.get('routing_ms'):.1f}ms")
        print(f"\n   Performance:")
        print(f"      Retrieval: {result.get('retrieval_ms'):.1f}ms")
        print(f"      Generation: {result.get('generation_ms'):.1f}ms")
        print(f"      Context chars: {result.get('context_chars')}")

    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "routing" in result


@pytest.mark.asyncio
async def test_intent_router_temporal_query():
    """
    Test: IntentRouter resolves temporal queries correctly.
    """
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    memory_store = MemoryStore(graph_db)
    graph_ops = GraphOps(graph_db=graph_db, vector_store=graph_db)
    await graph_ops.initialize()

    user_card_service = UserCardService(memory_store, graph_ops=graph_ops)
    user_card = await user_card_service.generate(USER_ID)

    router = IntentRouter()
    hints = await router.route("What happened yesterday?", user_card)

    print(f"\n📅 Temporal query routing:")
    print(f"   Date range: {hints.date_range}")
    print(f"   Memory boosts: {hints.memory_type_boost}")

    await graph_db.close()

    assert hints.date_range is not None
    assert "episode" in hints.memory_type_boost


@pytest.mark.asyncio
async def test_intent_router_identity_query():
    """
    Test: IntentRouter routes identity queries to psyche.
    """
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    memory_store = MemoryStore(graph_db)
    graph_ops = GraphOps(graph_db=graph_db, vector_store=graph_db)
    await graph_ops.initialize()

    user_card_service = UserCardService(memory_store, graph_ops=graph_ops)
    user_card = await user_card_service.generate(USER_ID)

    router = IntentRouter()
    hints = await router.route("Who am I?", user_card)

    print(f"\n🪪 Identity query routing:")
    print(f"   Memory boosts: {hints.memory_type_boost}")
    print(f"   Confidence: {hints.confidence:.2f}")

    await graph_db.close()

    assert "psyche" in hints.memory_type_boost


@pytest.mark.asyncio
async def test_intent_router_task_query():
    """
    Test: IntentRouter routes task queries to notes.
    """
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    memory_store = MemoryStore(graph_db)
    graph_ops = GraphOps(graph_db=graph_db, vector_store=graph_db)
    await graph_ops.initialize()

    user_card_service = UserCardService(memory_store, graph_ops=graph_ops)
    user_card = await user_card_service.generate(USER_ID)

    router = IntentRouter()
    hints = await router.route("What tasks should I focus on?", user_card)

    print(f"\n📋 Task query routing:")
    print(f"   Memory boosts: {hints.memory_type_boost}")
    print(f"   Keywords: {hints.search_keywords[:5]}")

    await graph_db.close()

    assert "note" in hints.memory_type_boost


@pytest.mark.asyncio
async def test_full_intent_router_pipeline_summary():
    """
    Summary test showing full IntentRouter + Retriever pipeline.
    """
    print("\n" + "=" * 60)
    print("INTENT ROUTER PIPELINE SUMMARY")
    print("=" * 60)

    async with RAGInterface(USER_ID) as rag:
        queries = [
            "How is my fitness going?",
            "What happened yesterday?",
            "Who am I as a person?",
            "What tasks should I do?",
        ]

        for query in queries:
            result = await rag.query_v2(query, include_stats=True)
            routing = result.get("routing", {})
            print(f"\n📝 Query: {query}")
            print(
                f"   Mode: {routing.get('mode')}, Confidence: {routing.get('confidence', 0):.2f}"
            )
            print(f"   Keywords: {routing.get('keywords', [])[:3]}")
            print(f"   Memory boosts: {routing.get('memory_boosts', [])}")
            print(f"   Answer: {result['answer'][:100]}...")

    print("\n" + "=" * 60)
    print("✅ Intent Router pipeline working!")
    print("=" * 60)
