"""
Full ingestion test for Example 8: Fitness Classes
47 sessions, 475 messages

Run: docker compose run --rm app poetry run python tests/integration/ingest_fitness_full.py
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from persona.services.ingestion_service import MemoryIngestionService
from persona.core.memory_store import MemoryStore
from persona.core.backends.neo4j_graph import Neo4jGraphDatabase

USER_ID = "fitness_test_v2"


async def generate_memory_episode(raw_content, user_id, session_id, graph_db, memory_store, ingestion_service):
    result = await ingestion_service.ingest(
        raw_content=raw_content,
        user_id=user_id,
        session_id=session_id,
        timestamp=datetime.utcnow(),
        source_type="conversation"
    )
    
    if not result.success:
        return None
    
    for memory in result.memories:
        memory_links = [l for l in result.links if l.source_id == memory.id]
        await memory_store.create(memory, links=memory_links)
    
    episode = next((m for m in result.memories if m.type == "episode"), None)
    if episode:
        previous = await memory_store.get_most_recent_episode(user_id)
        if previous and previous.id != episode.id:
            await memory_store.link_temporal_chain(episode, previous)
    
    return result


def format_session(session):
    lines = []
    for msg in session:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


async def main():
    # Load data from JSON file
    data_file = Path(__file__).parent / "fitness_data.json"
    with open(data_file) as f:
        data = json.load(f)
    
    question = data["question"]
    answer = data["answer"]
    sessions = data["sessions"]
    session_ids = data["session_ids"]
    
    print("=" * 60)
    print(f"INGESTING: {question}")
    print(f"Answer: {answer}")
    print(f"Sessions: {len(sessions)}")
    print("=" * 60)
    
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    
    if not await graph_db.user_exists(USER_ID):
        await graph_db.create_user(USER_ID)
    
    memory_store = MemoryStore(graph_db)
    ingestion_service = MemoryIngestionService()
    
    total_memories = 0
    
    for i, (session, sid) in enumerate(zip(sessions, session_ids)):
        raw = format_session(session)
        result = await generate_memory_episode(
            raw_content=raw,
            user_id=USER_ID,
            session_id=sid,
            graph_db=graph_db,
            memory_store=memory_store,
            ingestion_service=ingestion_service
        )
        
        if result:
            total_memories += len(result.memories)
            print(f"[{i+1:02d}/{len(sessions)}] {sid[:25]:25s} -> {len(result.memories)} memories")
        else:
            print(f"[{i+1:02d}/{len(sessions)}] {sid[:25]:25s} -> FAILED")
    
    await graph_db.close()
    
    print("\n" + "=" * 60)
    print(f"COMPLETE: {total_memories} memories from {len(sessions)} sessions")
    print(f"User ID: {USER_ID}")
    print("=" * 60)
    print(f"\nNeo4j queries:")
    print(f"MATCH (n) WHERE n.UserId = '{USER_ID}' RETURN n.type, count(*)")
    print(f"MATCH (n) WHERE n.UserId = '{USER_ID}' AND n.properties CONTAINS 'fitness' RETURN n")


if __name__ == "__main__":
    asyncio.run(main())
