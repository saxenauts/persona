"""
Ingest real longmemeval benchmark data into Neo4j.

This script ingests 2 benchmark questions with ALL their haystack sessions,
then displays the resulting graph.

Run: docker compose run --rm app poetry run python scripts/ingest_benchmark.py
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from persona.services.ingestion_service import MemoryIngestionService
from persona.core.memory_store import MemoryStore
from persona.core.backends.neo4j_graph import Neo4jGraphDatabase


BENCHMARK_FILE = Path("/app/tests/integration/fitness_benchmark_subset.json")

# Questions to ingest (indices in the benchmark file)
QUESTIONS_TO_INGEST = [0]  # Question 0 is "How many times a week do I have fitness class?" (47 sessions)

USER_ID = "fitness_test_v2"


async def generate_memory_episode(
    raw_content: str,
    user_id: str,
    session_id: str,
    graph_db,
    memory_store,
    ingestion_service
):
    """Ingest a single session."""
    result = await ingestion_service.ingest(
        raw_content=raw_content,
        user_id=user_id,
        session_id=session_id,
        timestamp=datetime.utcnow(),
        source_type="conversation"
    )
    
    if not result.success:
        print(f"  ❌ Failed: {result.error}")
        return None
    
    # Get previous episode BEFORE creating new ones
    previous = await memory_store.get_most_recent_episode(user_id)
    
    # Persist all memories
    for memory in result.memories:
        memory_links = [l for l in result.links if l.source_id == memory.id]
        await memory_store.create(memory, links=memory_links)
    
    # Link episodes in temporal chain
    episode = next((m for m in result.memories if m.type == "episode"), None)
    if episode and previous and previous.id != episode.id:
        await memory_store.link_temporal_chain(episode, previous)
    
    return result


def format_session(session_messages):
    """Convert session messages to raw text."""
    lines = []
    for msg in session_messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


async def main():
    print("=" * 60)
    print("LongMemEval Benchmark Ingestion: Fitness Question")
    print("=" * 60)
    
    # Load benchmark data
    with open(BENCHMARK_FILE) as f:
        benchmark = json.load(f)
    
    # Initialize services
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()

    # CLEANUP: Delete existing user data to ensure clean slate
    print(f"🧹 Cleaning graph for user: {USER_ID}...")
    await graph_db.delete_user(USER_ID)
    
    # Create user (this will be a fresh start)
    await graph_db.create_user(USER_ID)
    print(f"✅ Created fresh user: {USER_ID}")
    
    memory_store = MemoryStore(graph_db)
    ingestion_service = MemoryIngestionService()
    
    total_sessions = 0
    total_memories = 0
    
    for q_idx in QUESTIONS_TO_INGEST:
        question = benchmark[q_idx]
        q_id = question["question_id"]
        q_text = question["question"]
        q_answer = question["answer"]
        sessions = question["haystack_sessions"]
        session_ids = question.get("haystack_session_ids", [f"session_{i}" for i in range(len(sessions))])
        
        print(f"\n📌 Question: {q_text}")
        print(f"   Answer: {q_answer}")
        print(f"   Total Sessions: {len(sessions)}")
        print("-" * 40)
        
        # Ingest first 20 sessions
        max_sessions = 20
        for i, (session, sid) in enumerate(zip(sessions[:max_sessions], session_ids[:max_sessions])):
            raw_content = format_session(session)
            
            result = await generate_memory_episode(
                raw_content=raw_content,
                user_id=USER_ID,
                session_id=f"{q_id}_{sid}",
                graph_db=graph_db,
                memory_store=memory_store,
                ingestion_service=ingestion_service
            )
            
            if result:
                mem_count = len(result.memories)
                total_memories += mem_count
                print(f"  [{i+1}/{len(sessions)}] {sid[:20]}... → {mem_count} memories")
            
            total_sessions += 1
    
    await graph_db.close()
    
    print("\n" + "=" * 60)
    print(f"✅ INGESTION COMPLETE")
    print(f"   Sessions ingested: {total_sessions}")
    print(f"   Memories created: {total_memories}")
    print(f"   User ID: {USER_ID}")
    print("=" * 60)
    
    print("\n📊 NEO4J QUERIES TO VERIFY (Use these in Neo4j Browser):")
    print("-" * 40)
    
    clean_uid = USER_ID.replace("-", "_").replace(" ", "_")
    user_label = f"User_{clean_uid}"
    
    print(f"""
// 1. Verify User Isolation: Count nodes with correct label
MATCH (n:{user_label}) 
RETURN count(n) as {user_label}_count;

// 2. View most recent Episodes (Temporal Chain)
MATCH (n:Episode:{user_label})
RETURN n.id, n.created_at, n.summary
ORDER BY n.created_at DESC
LIMIT 10;

// 3. Inspect Memory Hierarchy (Episode -> Psyche -> Goal)
MATCH (e:Episode:{user_label})-[r]-(m:Psyche:{user_label})
RETURN e.summary as Episode, type(r) as Relation, m.content as PsycheMemory
LIMIT 20;

// 4. Check Vector Index Usage (Implicitly via a search query simulation)
CALL db.index.vector.queryNodes('vector_idx_{clean_uid}', 5, [0.1, 0.1, ...]) 
YIELD node, score
RETURN node.name, score;
""")


if __name__ == "__main__":
    asyncio.run(main())
