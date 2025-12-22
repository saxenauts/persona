#!/usr/bin/env python3
"""
Quick verification script to test the new Retriever.
Run with: poetry run python tests/integration/test_retrieval_live.py
"""

import asyncio
from persona.core.rag_interface import RAGInterface

USER_ID = "goal_tracking_eval_user"

QUERIES = [
    "What goals do I have?",
    "What happened with running?",
    "Tell me about Whole Foods",
]


async def main():
    print("=== Testing New Retriever ===\n")
    
    async with RAGInterface(USER_ID) as rag:
        for query in QUERIES:
            print(f"Query: {query}")
            print("-" * 50)
            
            try:
                context = await rag.get_context(query, top_k=3, hop_depth=1)
                print(f"Context Length: {len(context)} chars")
                print(f"Preview:\n{context[:500]}...")
            except Exception as e:
                print(f"Error: {e}")
            
            print("\n")


if __name__ == "__main__":
    asyncio.run(main())
