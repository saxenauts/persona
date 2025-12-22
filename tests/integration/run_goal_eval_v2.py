import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
from uuid import UUID

from persona.services.ingestion_service import MemoryIngestionService, IngestionResult
from persona.services.ask_service import AskService
from persona.models.schema import AskRequest
from persona.core.graph_ops import GraphOps
from persona.core.backends.neo4j_graph import Neo4jGraphDatabase
from server.logging_config import get_logger

# Configure Logging
logger = get_logger("goal_eval")

# Constants
DATASET_PATH = "tests/integration/goal_tracking_dataset_20_sessions.json"
USER_ID = "eval_goal_tracking_user_v2"
START_DATE = datetime(2024, 1, 1)  # Simulation start date

# Golden Questions from Plan
GOLDEN_QUESTIONS = [
    {
        "id": "Q1",
        "question": "How did my early diet choices affect my European trip planning?",
        "expected_themes": ["Whole Foods", "Budget", "Savings", "Europe Trip", "$200"],
        "reasoning": "Tests ability to track complex cause-and-effect relationships across themes (Diet -> Finance)."
    },
    {
        "id": "Q2",
        "question": "We decided on a tech stack a while ago, specifically for privacy. Did we ever implement it?",
        "expected_themes": ["Supabase", "RLS", "Privacy", "Implemented", "FocusFlow"],
        "reasoning": "Tests deep context recall of technical decisions and their eventual execution."
    },
    {
        "id": "Q3",
        "question": "I met someone at a meetup who liked my design choices. Who was it and what did they like?",
        "expected_themes": ["Bald Guy", "Investor", "Dark Mode", "Meetup"],
        "reasoning": "Tests entity tracking (People) and linking them to specific feedback/events."
    },
    {
        "id": "Q4",
        "question": "What happened to the 'Clean Body' ritual?",
        "expected_themes": ["Pizza", "Relapse", "Meal Prep", "Consistent", "Smoothies"],
        "reasoning": "Tests tracking of habit adherence, failures, and recovery over time."
    },
    {
        "id": "Q5",
        "question": "I had a random idea about using AI for the app. Did I follow up on it?",
        "expected_themes": ["AI", "OpenAI", "Summary Feature", "Followed Up"],
        "reasoning": "Tests tracking of 'random' ideas that evolve into concrete project features."
    }
]

async def setup_clean_state():
    """Clear data for the test user to ensure a clean slate."""
    logger.info(f"Cleaning state for user {USER_ID}...")
    db = Neo4jGraphDatabase()
    await db.initialize()
    # Delete all nodes for this user using the correct method
    await db.delete_user(USER_ID)
    # Re-create user node to ensure it exists for ingestion checks
    await db.create_user(USER_ID)
    
    # CRITICAL: Ensure vector index exists for this user
    # MemoryStore writes embeddings, but doesn't create the index. 
    # specific to user isolation model.
    from persona.core.backends.neo4j_vector import Neo4jVectorStore
    try:
        vs = Neo4jVectorStore(graph_driver=db.driver)
        await vs._ensure_user_index(USER_ID)
        logger.info(f"Vector index ensured for {USER_ID}")
    except Exception as e:
        logger.error(f"Failed to ensure vector index: {e}")

    logger.info("State cleaned.")

async def ingest_dataset():
    """Ingest the 20-session dataset sequentially with proper persistence."""
    print(f"Checking dataset at {DATASET_PATH}...")
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}")
        return

    with open(DATASET_PATH, "r") as f:
        sessions = json.load(f)

    # Initialize database and services  
    from persona.core.memory_store import MemoryStore
    
    graph_db = Neo4jGraphDatabase()
    await graph_db.initialize()
    
    memory_store = MemoryStore(graph_db)
    ingestion_service = MemoryIngestionService()
    
    print(f"Starting ingestion of {len(sessions)} sessions...")
    
    total_memories = 0
    
    for i, session in enumerate(sessions):
        # Simulate time passing (e.g., 2-3 days per session)
        current_date = START_DATE + timedelta(days=i * 2) 
        session_id = session.get("session_id", f"session_{i+1}")
        
        # Construct full transcript for context
        transcript = ""
        for turn in session["turns"]:
            role = "User" if turn["role"] == "user" else "Assistant"
            transcript += f"{role}: {turn['content']}\n\n"
            
        print(f"Ingesting Session {i+1}/{len(sessions)} (Date: {current_date.strftime('%Y-%m-%d')})...")
        
        try:
            # Step 1: Extract memories (in-memory)
            result = await ingestion_service.ingest(
                raw_content=transcript,
                user_id=USER_ID,
                timestamp=current_date,
                session_id=session_id,
                source_type="conversation",
                source_ref=session_id
            )
            
            if not result.success:
                print(f"Session {i+1} failed: {result.error}")
                continue
            
            # Step 2: Get previous episode BEFORE creating new ones (for temporal chain)
            previous = await memory_store.get_most_recent_episode(USER_ID)
            
            # Step 3: PERSIST all memories to Neo4j
            for memory in result.memories:
                memory_links = [l for l in result.links if l.source_id == memory.id]
                await memory_store.create(memory, links=memory_links)
            
            # Step 4: Link episodes in temporal chain
            episode = next((m for m in result.memories if m.type == "episode"), None)
            if episode and previous and previous.id != episode.id:
                await memory_store.link_temporal_chain(episode, previous)
            
            total_memories += len(result.memories)
            print(f"Session {i+1} persisted. Memories: {len(result.memories)} (Total: {total_memories})")
            
        except Exception as e:
            print(f"Session {i+1} Exception: {e}")
            import traceback
            traceback.print_exc()
    
    await graph_db.close()
    print(f"\n✅ Ingestion complete. Total memories: {total_memories}")

async def run_evaluation():
    """Run the Golden Questions against the ingested data with retrieval logging."""
    print("\n=== STARTING EVALUATION ===")
    
    results = {}
    
    # We use GraphOps context manager for the defined AskService lifecycle
    async with GraphOps() as graph_ops:
        # Initialize RAG for context retrieval logging
        from persona.core.rag_interface import RAGInterface
        from persona.core.graph_ops import GraphContextRetriever
        from persona.llm.llm_graph import generate_structured_insights
        
        rag = RAGInterface(USER_ID)
        rag.graph_ops = graph_ops
        rag.graph_context_retriever = GraphContextRetriever(graph_ops)
        
        for q in GOLDEN_QUESTIONS:
            print(f"\n--- Evaluating {q['id']}: {q['question'][:50]}... ---")
            
            # Step 1: Get retrieved context (for logging)
            try:
                retrieved_context = await rag.get_context(q["question"])
                print(f"  [Retrieved Context Length: {len(retrieved_context)} chars]")
                
                # Truncated preview for console
                context_preview = retrieved_context[:500] + "..." if len(retrieved_context) > 500 else retrieved_context
                print(f"  Context Preview: {context_preview[:200]}...")
            except Exception as e:
                print(f"  [Retrieval Error: {e}]")
                retrieved_context = f"ERROR: {e}"
            
            # Step 2: Generate answer
            output_schema = {
                "answer": "Detailed answer to the user's question based on memory.",
                "confidence": "High, Medium, or Low",
                "reasoning": "Explanation of how the answer was derived from memory."
            }
            
            request = AskRequest(
                query=q["question"],
                output_schema=output_schema
            )
            
            try:
                response = await AskService.ask_insights(USER_ID, request, graph_ops)
                answer_data = response.result
                
                results[q["id"]] = {
                    "question": q["question"],
                    "expected_themes": q["expected_themes"],
                    "generated_answer": answer_data.get("answer"),
                    "confidence": answer_data.get("confidence"),
                    "reasoning": answer_data.get("reasoning"),
                    "retrieved_context": retrieved_context,  # FULL context for analysis
                    "context_length": len(retrieved_context)
                }
                
                print(f"  Answer: {answer_data.get('answer', '')[:100]}...")
                print(f"  Confidence: {answer_data.get('confidence')}")
            
            except Exception as e:
                print(f"  [LLM Error: {e}]")
                results[q["id"]] = {
                    "error": str(e),
                    "retrieved_context": retrieved_context
                }

    # Save Results (with full context for manual inspection)
    output_file = "tests/integration/goal_eval_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Evaluation complete. Full results saved to {output_file}")
    
    # Print Summary Table
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Q#':<5} {'Confidence':<12} {'Context':<10} {'Answer Preview':<40}")
    print("-" * 70)
    for q_id, res in results.items():
        if "error" in res:
            print(f"{q_id:<5} {'ERROR':<12} {'-':<10} {res['error'][:40]}")
        else:
            ctx_len = res.get('context_length', 0)
            answer_preview = res.get('generated_answer', '')[:37] + "..."
            print(f"{q_id:<5} {res.get('confidence', 'N/A'):<12} {ctx_len:<10} {answer_preview}")

async def main():
    await setup_clean_state()
    await ingest_dataset()
    await run_evaluation()

if __name__ == "__main__":
    asyncio.run(main())
