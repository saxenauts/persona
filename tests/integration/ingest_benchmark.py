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


BENCHMARK_FILE = Path("/app/../evals/data/longmemeval/longmemeval_s_cleaned.json")

# Questions to ingest (indices in the benchmark file)
QUESTIONS_TO_INGEST = [0, 4]  # e47becba (degree), 1e043500 (spotify playlist)

USER_ID = "benchmark_user_v2"


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
    
    # Persist all memories
    for memory in result.memories:
        memory_links = [l for l in result.links if l.source_id == memory.id]
        await memory_store.create(memory, links=memory_links)
    
    # Link episodes in temporal chain
    episode = next((m for m in result.memories if m.type == "episode"), None)
    if episode:
        previous = await memory_store.get_most_recent_episode(user_id)
        if previous and previous.id != episode.id:
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
    print("LongMemEval Benchmark Ingestion")
    print("=" * 60)
    
    # Load benchmark data
    with open(BENCHMARK_FILE) as f:
        benchmark = json.load(f)
    
    # Initialize services
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    
    # Ensure user exists
    if not await graph_db.user_exists(USER_ID):
        await graph_db.create_user(USER_ID)
        print(f"✅ Created user: {USER_ID}")
    
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
        print(f"   Sessions: {len(sessions)}")
        print("-" * 40)
        
        # Ingest each session (limit to first 5 for speed in demo)
        max_sessions = 5  # Change to len(sessions) to ingest ALL
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
                print(f"  [{i+1}/{max_sessions}] {sid[:20]}... → {mem_count} memories")
            
            total_sessions += 1
    
    await graph_db.close()
    
    print("\n" + "=" * 60)
    print(f"✅ INGESTION COMPLETE")
    print(f"   Sessions ingested: {total_sessions}")
    print(f"   Memories created: {total_memories}")
    print(f"   User ID: {USER_ID}")
    print("=" * 60)
    
    print("\n📊 NEO4J QUERIES TO VERIFY:")
    print("-" * 40)
    print(f"""
// Get all memories for this user
MATCH (n) WHERE n.user_id = '{USER_ID}'
RETURN n.type, n.title, n.content
ORDER BY n.timestamp

// Count by type
MATCH (n) WHERE n.user_id = '{USER_ID}'
RETURN n.type, count(*) as count

// View all relationships
MATCH (a)-[r]->(b)
WHERE a.user_id = '{USER_ID}'
RETURN a.title, type(r), b.title
LIMIT 50

// Visualize full graph
MATCH (n) WHERE n.user_id = '{USER_ID}'
OPTIONAL MATCH (n)-[r]-(m)
RETURN n, r, m
""")


if __name__ == "__main__":
    asyncio.run(main())
