import asyncio
import os
import sys
from persona.core.neo4j_database import Neo4jConnectionManager
from persona.services.ingest_service import IngestService
from persona.core.graph_ops import GraphOps
from persona.models.schema import UnstructuredData

async def analyze():
    print("🚀 Analying Ingestion Bloat...")
    
    # 1. Setup & Clean
    mgr = Neo4jConnectionManager()
    await mgr.initialize()
    graph_ops = GraphOps(mgr)
    
    print("🧹 Cleaning Graph...")
    await mgr.clean_graph()
    print("🔄 Ensuring Vector Index...")
    await mgr.ensure_vector_index()
    
    async def count_stats():
        async with mgr.driver.session() as session:
            n = await session.run("MATCH (n) RETURN count(n) as c")
            e = await session.run("MATCH ()-[r]->() RETURN count(r) as c")
            return (await n.single())['c'], (await e.single())['c']

    # 2. Baseline
    n0, e0 = await count_stats()
    print(f"Baseline: {n0} Nodes, {e0} Edges")
    
    # 3. Ingest Fact 1
    user_id = "ingest_analyst_1"
    await mgr.create_user(user_id)
    
    print("\n📥 Ingesting: 'I bought a Honda Civic yesterday.'")
    await IngestService.ingest_data(user_id, UnstructuredData(
        content="I bought a Honda Civic yesterday.",
        title="Fact 1"
    ), graph_ops)
    
    n1, e1 = await count_stats()
    print(f"State 1: {n1} Nodes, {e1} Edges (+{n1-n0} Nodes)")
    
    # 4. Ingest Fact 2
    print("\n📥 Ingesting: 'The GPS broke this morning.'")
    await IngestService.ingest_data(user_id, UnstructuredData(
        content="The GPS broke this morning.",
        title="Fact 2"
    ), graph_ops)
    
    n2, e2 = await count_stats()
    print(f"State 2: {n2} Nodes, {e2} Edges (+{n2-n1} Nodes)")
    
    # 5. List Nodes
    print("\n--- NODES DETECTED ---")
    async with mgr.driver.session() as session:
        nodes = await session.run("MATCH (n) RETURN labels(n), n.name, n.type LIMIT 20")
        async for r in nodes:
            print(f"[{r['labels(n)']}] {r['n.name']} ({r['n.type']})")

    await mgr.close()

if __name__ == "__main__":
    # Ensure env vars
    if not os.getenv("URI_NEO4J"):
        os.environ["URI_NEO4J"] = "bolt://localhost:7687"
        
    asyncio.run(analyze())
