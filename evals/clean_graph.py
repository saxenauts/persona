import asyncio
import os
import sys
import argparse

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


async def main():
    """
    Connects directly to the Neo4j database and completely wipes all data,
    including all nodes, relationships, and vector indexes.
    """
    parser = argparse.ArgumentParser(description="Wipe the Neo4j graph database.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass the interactive confirmation prompt."
    )
    parser.add_argument("--uri", help="Neo4j URI (e.g., 'neo4j://localhost:7687')")
    parser.add_argument("--user", help="Neo4j username")
    parser.add_argument("--password", help="Neo4j password")
    args = parser.parse_args()
    
    from server.config import config
    if args.uri:
        config.NEO4J.URI = args.uri
    if args.user:
        config.NEO4J.USER = args.user
    if args.password:
        config.NEO4J.PASSWORD = args.password
        
    # Use the new backend abstraction
    from persona.core.backends.neo4j_graph import Neo4jGraphDatabase

    print("Connecting to Neo4j database...")
    db = Neo4jGraphDatabase()
    
    try:
        await db.initialize()
        print("✅ Connection successful.")
        
        print("\n🔥 Wiping the entire graph. This will delete ALL nodes, relationships, and indexes.")
        
        # Confirmation step to prevent accidental deletion
        if not args.force:
            confirm = input("Are you sure you want to proceed? (yes/no): ")
            if confirm.lower() != 'yes':
                print("🚫 Operation cancelled.")
                return

        print("\nCleaning graph...")
        await db.clean_graph()
        print("✅ Graph has been completely wiped.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
    finally:
        await db.close()
        print("\nConnection closed.")

if __name__ == "__main__":
    asyncio.run(main())