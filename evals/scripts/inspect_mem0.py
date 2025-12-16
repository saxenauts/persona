import os
from mem0 import Memory
import json
import sys

def inspect_mem0(user_id):
    print(f"🔍 Inspecting Mem0 User: {user_id}")
    
    config = {
        "llm": {
            "provider": "azure_openai",
            "config": {
                "model": os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4.1-mini"),
                "temperature": 0.0,
                "azure_kwargs": {
                    "azure_deployment": os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4.1-mini"),
                    "azure_endpoint": os.getenv("AZURE_API_BASE"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_key": os.getenv("AZURE_API_KEY"),
                }
            }
        },
        "embedder": {
            "provider": "azure_openai",
            "config": {
                "model": os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
                "azure_kwargs": {
                    "azure_deployment": os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
                    "azure_endpoint": os.getenv("AZURE_API_BASE"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_key": os.getenv("AZURE_API_KEY"),
                }
            }
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "mem0_benchmark",
                "path": "/tmp/qdrant_mem0_local",
            }
        },
        "history_db_path": "/tmp/mem0_history.db"
    }
    
    try:
        client = Memory.from_config(config)
        
        # 1. Get All Memories
        print("\n--- Stored Memories ---")
        memories = client.get_all(user_id=user_id)
        
        print(f"Raw Type: {type(memories)}")
        
        # Unpack if it's a dict containing 'results' or 'memories'
        if isinstance(memories, dict):
            if "results" in memories:
                memories = memories["results"]
            elif "memories" in memories:
                memories = memories["memories"]
            else:
                print(f"Unknown dict keys: {memories.keys()}")
                # Try to use the dict itself if it looks like a single memory (unlikely)
        
        if not memories:
            print("No memories found!")
        else:
            print(f"Found {len(memories)} memories.")
            for i, m in enumerate(memories):
                # mem0 0.0.x might return dict or object
                if isinstance(m, dict):
                     content = m.get("memory", str(m))
                else:
                     content = str(m)
                
                print(f"[{i+1}] {content[:200]}..." if len(content) > 200 else f"[{i+1}] {content}")

        # 2. Search Test (simulate retrieval)
        print("\n--- Search Test (Query: 'playlist') ---")
        search_res = client.search("playlist", user_id=user_id)
        for i, m in enumerate(search_res):
            content = m.get("memory") if isinstance(m, dict) else str(m)
            score = m.get("score", 0) if isinstance(m, dict) else "?"
            print(f"[{i+1}] (Score: {score}) {content[:150]}...")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_mem0.py <user_id>")
    else:
        inspect_mem0(sys.argv[1])
