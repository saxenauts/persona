"""
Test ingestion with REAL longmemeval benchmark data.

These are the actual answer sessions from the benchmark that contain the needle.

Question 1: "What degree did I graduate with?" → Answer: "Business Administration"
Question 2: "What is the name of the playlist I created on Spotify?" → Answer: "Summer Vibes"

Run: docker compose run --rm app poetry run pytest tests/integration/test_real_benchmark.py -v -s
"""

import asyncio
import pytest
from datetime import datetime

from persona.services.ingestion_service import MemoryIngestionService, IngestionResult
from persona.core.memory_store import MemoryStore
from persona.core.backends.neo4j_graph import Neo4jGraphDatabase


# ============================================================================
# REAL BENCHMARK DATA (Answer Sessions from LongMemEval)
# ============================================================================

# Question: "What degree did I graduate with?" → Answer: "Business Administration"
# Session ID: answer_280352e9
DEGREE_SESSION = """USER: I graduated with a degree in Business Administration, which has definitely helped me in my new role. Do you have any advice on how to stay organized when it comes to paperwork and documentation, especially when it comes to tracking expenses and reimbursements?

ASSISTANT: Congratulations on your degree in Business Administration! That's a great foundation for your new role.

Staying organized with paperwork and documentation is crucial, especially when it comes to tracking expenses and reimbursements. Here are some tips to help you stay on top of it:

1. **Create a filing system**: Develop a consistent filing system, both physical and digital, to store and categorize your documents.
2. **Digitize your documents**: Scan or digitize paper documents to reduce clutter and make them easily searchable.
3. **Expense tracking tools**: Utilize expense tracking tools like Expensify, Concur, or Zoho Expense to streamline your expense reporting.
4. **Receipt organization**: Designate a specific place to store receipts, such as a receipt binder or a digital folder.
5. **Categorize and tag documents**: Use clear categories and tags to label your documents.
6. **Set reminders**: Schedule regular reminders to review and submit expense reports.

By implementing these strategies, you'll be well on your way to staying organized and efficient!"""

# Question: "What is the name of the playlist I created on Spotify?" → Answer: "Summer Vibes"
# Session ID: answer session from question 1e043500
SPOTIFY_SESSION = """USER: What's a good music streaming service that can help me discover new music and artists, especially in the ambient and lo-fi genres? Also, by the way, I've been listening to this one playlist on Spotify that I created, called Summer Vibes, and it's got all these chill tracks that are perfect for relaxing or working out.

ASSISTANT: For discovering new music in ambient and lo-fi genres, here are some great options:

1. **Spotify**: Great for personalized recommendations and curated playlists like Lo-Fi Beats and Ambient Relaxation.
2. **Apple Music**: Offers curated playlists and radio stations in these genres.
3. **YouTube Music**: Has a vast library including many independent ambient artists.
4. **Bandcamp**: Excellent for discovering independent artists in niche genres.
5. **SoundCloud**: Popular with lo-fi producers and ambient artists.

Your "Summer Vibes" playlist sounds perfect! Spotify's algorithm will learn from it and suggest similar tracks. You might also enjoy their "Release Radar" for new music in your preferred styles."""

# A random haystack session (no answer) - for testing noise
HAYSTACK_SESSION = """USER: I'm looking for some recommendations on cashback apps. Can you tell me which ones are the most popular and offer the best rewards?

ASSISTANT: Here are some of the most popular cashback apps:

1. **Rakuten**: Up to 40% cashback at over 2,500 stores including Amazon and Walmart.
2. **Ibotta**: Cashback on groceries, gas, and other purchases. Average 2-5% cashback.
3. **Fetch Rewards**: Cashback on gas, groceries, and dining.
4. **Drop**: Cashback at brands like Starbucks, Amazon, and Uber.
5. **Dosh**: Cashback on purchases at thousands of stores and restaurants.

These apps are great for earning rewards on purchases you're already making!"""


# ============================================================================
# Integration Test
# ============================================================================

USER_ID = "benchmark_real_v2"


async def generate_memory_episode(raw_content, user_id, session_id, graph_db, memory_store, ingestion_service):
    """Full pipeline: Ingest → Embed → Persist."""
    result = await ingestion_service.ingest(
        raw_content=raw_content,
        user_id=user_id,
        session_id=session_id,
        timestamp=datetime.utcnow(),
        source_type="conversation"
    )
    
    if not result.success:
        return result
    
    for memory in result.memories:
        memory_links = [l for l in result.links if l.source_id == memory.id]
        await memory_store.create(memory, links=memory_links)
    
    episode = next((m for m in result.memories if m.type == "episode"), None)
    if episode:
        previous = await memory_store.get_most_recent_episode(user_id)
        if previous and previous.id != episode.id:
            await memory_store.link_temporal_chain(episode, previous)
    
    return result


@pytest.mark.asyncio
async def test_ingest_real_benchmark_sessions():
    """Ingest real benchmark sessions and verify memories are created."""
    
    # Initialize
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    
    if not await graph_db.user_exists(USER_ID):
        await graph_db.create_user(USER_ID)
    
    memory_store = MemoryStore(graph_db)
    ingestion_service = MemoryIngestionService()
    
    sessions = [
        ("degree_answer", DEGREE_SESSION),
        ("spotify_answer", SPOTIFY_SESSION),
        ("haystack_cashback", HAYSTACK_SESSION),
    ]
    
    all_memories = []
    
    print("\n" + "=" * 60)
    print("INGESTING REAL BENCHMARK SESSIONS")
    print("=" * 60)
    
    for session_id, content in sessions:
        result = await generate_memory_episode(
            raw_content=content,
            user_id=USER_ID,
            session_id=session_id,
            graph_db=graph_db,
            memory_store=memory_store,
            ingestion_service=ingestion_service
        )
        
        assert result.success, f"Failed to ingest {session_id}: {result.error}"
        all_memories.extend(result.memories)
        
        print(f"\n📝 Session: {session_id}")
        for m in result.memories:
            print(f"   [{m.type}] {m.title}")
    
    await graph_db.close()
    
    print("\n" + "=" * 60)
    print(f"✅ TOTAL MEMORIES CREATED: {len(all_memories)}")
    print(f"   User ID: {USER_ID}")
    print("=" * 60)
    
    # Assertions
    assert len(all_memories) >= 3, "Should have at least 3 episodes"
    
    episodes = [m for m in all_memories if m.type == "episode"]
    assert len(episodes) == 3, "Should have 3 episodes"
    
    # Print queries for user
    print(f"""
📊 NEO4J QUERIES TO VERIFY (http://localhost:7474):
--------------------------------------------------

// 1. Get all memories 
MATCH (n) WHERE n.user_id = '{USER_ID}'
RETURN n.type, n.title, substring(n.content, 0, 100) as content_preview
ORDER BY n.timestamp

// 2. Count by type
MATCH (n) WHERE n.user_id = '{USER_ID}'
RETURN n.type, count(*) as count

// 3. View relationships
MATCH (a)-[r]->(b)
WHERE a.user_id = '{USER_ID}'
RETURN a.title, type(r), b.title

// 4. SEARCH for "Business Administration"
MATCH (n) WHERE n.user_id = '{USER_ID}' AND n.content CONTAINS 'Business Administration'
RETURN n.title, n.content

// 5. SEARCH for "Summer Vibes"
MATCH (n) WHERE n.user_id = '{USER_ID}' AND n.content CONTAINS 'Summer Vibes'
RETURN n.title, n.content

// 6. Full graph visualization
MATCH (n) WHERE n.user_id = '{USER_ID}'
OPTIONAL MATCH (n)-[r]-(m)
RETURN n, r, m
""")


if __name__ == "__main__":
    asyncio.run(test_ingest_real_benchmark_sessions())
