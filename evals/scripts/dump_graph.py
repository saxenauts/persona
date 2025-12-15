import os
import asyncio
from neo4j import GraphDatabase

URI = os.getenv("URI_NEO4J", "bolt://localhost:7687")
AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))

async def dump_graph(user_id):
    driver = GraphDatabase.driver(URI, auth=AUTH)
    print(f"🔌 Connecting to {URI} for user {user_id}...")
    
    query_nodes = f"""
    MATCH (n)
    WHERE n.user_id = '{user_id}' OR n.userId = '{user_id}' OR any(l in labels(n) WHERE l = '{user_id}')  // Try various schemas
    RETURN n
    """
    
    # Actually, Persona usually labels nodes with the label? Or property?
    # Let's just match ALL and filter by property manually if needed, or check schema.
    # In `graph_ops.py`, usually `MATCH (n:User {id: $user_id})`... 
    # But for ingestion nodes, they are usually connected to user?
    # Let's try to find the User node first.
    
    with driver.session() as session:
        # 1. Find User Node
        result = session.run("MATCH (u:User {id: $uid}) RETURN u", uid=user_id)
        user_node = result.single()
        if not user_node:
            print(f"❌ User node '{user_id}' NOT FOUND.")
            # Try searching by label just in case
            # session.run(f"MATCH (n:`{user_id}`) RETURN n")
        else:
            print(f"✅ Found User Node: {user_node}")
            
        # 2. Find ALL nodes related to this user (assuming subgraph isolation)
        # Often Persona links everything to chunks or similar?
        # Let's dumping EVERYTHING if count is small, or filter by created time if possible.
        # Check node count
        count = session.run("MATCH (n) RETURN count(n) as c").single()['c']
        print(f"📊 Total Nodes in DB: {count}")
        
        print("\n--- CONNECTED NODES (1 hop) ---")
        # Try to find nodes connected to user
        # Note: Code uses UserId (PascalCase) property.
        nodes = session.run("MATCH (u:User {id: $uid})-[r]-(n) RETURN type(r) as t, labels(n) as l, properties(n) as p LIMIT 50", uid=user_id)
        found_any = False
        for r in nodes:
            print(f"(User)-[{r['t']}]-({r['l']} {r['p']})")
            found_any = True
        
        if not found_any:
            print("⚠️ No nodes connected to User! Checking for ORPHAN nodes via UserId property...")
            # Use correct case UserId
            orphans = session.run("MATCH (n {UserId: $uid}) RETURN labels(n), properties(n) LIMIT 50", uid=user_id)
            for r in orphans:
                print(f"[ORPHAN] {r}")
            
    driver.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python dump_graph.py <user_id>")
        # Default to one from logs if available
        # user_id = "Persona_bench_user_3_20251214_183735" 
    else:
        asyncio.run(dump_graph(sys.argv[1]))
