import os
from mem0 import Memory
from .base import MemorySystem

class Mem0Adapter(MemorySystem):
    def __init__(self):
        # Configure Mem0 to use Azure OpenAI
        config = {
            "llm": {
                "provider": "azure_openai",
                "config": {
                    "model": os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o-mini"),
                    "temperature": 0.0,
                    "azure_kwargs": {
                        "azure_deployment": os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o-mini"),
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
                    "path": "/tmp/qdrant_mem0_local",  # Use local file path for simplicity
                }
            },
            "history_db_path": "/tmp/mem0_history.db"
        }
        self.client = Memory.from_config(config)

    def add_session(self, user_id: str, session_data: str, date: str):
        # mem0 treats everything as a "memory". 
        messages = [{"role": "user", "content": f"[Date: {date}] {session_data}"}]
        self.client.add(messages, user_id=user_id)

    def add_sessions(self, user_id: str, sessions: list):
        # Optimize: Add all messages in one call
        messages = []
        for s in sessions:
            messages.append({"role": "user", "content": f"[Date: {s['date']}] {s['content']}"})
        self.client.add(messages, user_id=user_id)

    def query(self, user_id: str, query: str) -> str:
        # Mem0 'search' returns relevant memories, but we want an ANSWER.
        # Mem0 doesn't inherently "answer" questions in its core `search` API, it just retrieves.
        # Wait, the user wants to benchmark the *system*.
        # Only 'MemoryClient' (platform) might have advanced answer generation?
        # Typically Mem0 is used for Retrieval. 
        # But `client.add` stores it. `client.search` retrieves.
        # To get an answer, we need to do RAG: Retrieve + Generate.
        # I will implement a basic RAG loop here using the same Azure LLM.
        
        memories = self.client.search(query, user_id=user_id, limit=5)
        # mem0 search returns list of dicts: {'id': ..., 'memory': '...', 'score': ...}
        # If it returns strings, handle that too.
        context_lines = []
        for m in memories:
            if isinstance(m, dict):
                context_lines.append(m.get("memory", str(m)))
            else:
                context_lines.append(str(m))
        
        context = "\n".join(context_lines)
        
        # Simple RAG generation using the initialized LLM config is not directly exposed by Mem0 class for generation?
        # I'll use langchain or simple openai call here for fairness, 
        # matching what Persona does (GraphContextRetriever + LLM).
        
        # Actually, let's use the `mem0` library's intended usage.
        # If it doesn't provide generation, I will wrap it. 
        # FOR FAIRNESS: I should use the SAME generation prompt for all if possible, 
        # OR rely on the library's "ask" feature if it exists.
        # Looking at docs, Mem0 is primarily a MEMORY layer, not a full agent.
        # So I will implement the Generation step myself to keep it consistent.
        return self._generate_answer(query, context)

    def _generate_answer(self, query: str, context: str) -> str:
        from langchain_openai import AzureChatOpenAI
        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
            openai_api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_key=os.getenv("AZURE_API_KEY"),
            temperature=0,
        )
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        response = llm.invoke(prompt)
        return response.content

    def reset(self, user_id: str):
        self.client.delete_all(user_id=user_id)
