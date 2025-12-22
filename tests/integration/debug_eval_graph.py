import asyncio
from persona.core.backends.neo4j_graph import Neo4jGraphDatabase
from persona.core.backends.neo4j_vector import Neo4jVectorStore
from server.logging_config import get_logger

logger = get_logger("debug_eval")

USER_ID = "eval_goal_tracking_user_v2"

async def check_graph():
    print(f"Connecting to Neo4j...")
    db = Neo4jGraphDatabase()
    await db.initialize()
    
    # Check node count
    print(f"Checking nodes for {USER_ID}...")
    nodes = await db.get_all_nodes(USER_ID)
    print(f"Total nodes for {USER_ID}: {len(nodes)}")
    
    # Check node types
    types = {}
    for n in nodes:
        t = n.get("type", "unknown")
        types[t] = types.get(t, 0) + 1
    print(f"Node types: {types}")
    
    # Check content of a few nodes
    print("Sample Nodes:")
    for n in nodes[:3]:
        content = n.get('content', '') or n.get('properties', {}).get('content', '')
        print(f"- {n.get('name')}: {content[:50]}...")

    # Check Vector Search
    print(f"Init Vector Store...")
    vs = Neo4jVectorStore(graph_driver=db.driver)
    # Generate embedding for query "Whole Foods"
    from persona.llm.client_factory import get_embedding_client
    print("Generating embedding...")
    emb_client = get_embedding_client()
    query = "Whole Foods expensive"
    emb = await emb_client.embeddings([query])
    
    if emb[0]:
        print(f"Vector search for '{query}'...")
        results = await vs.search_similar(emb[0], USER_ID, limit=5)
        for r in results:
            print(f" - {r['score']:.4f}: {r['node_name']}")
    else:
        print("Failed to generate embedding")
        
    await db.close()

if __name__ == "__main__":
    asyncio.run(check_graph())
