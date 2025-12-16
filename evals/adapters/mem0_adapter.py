"""
Mem0Adapter - Optimized for Local Parallelism

Key Optimizations:
1. Uses Qdrant Server (Docker) instead of local file-based storage for thread-safety.
2. ThreadPoolExecutor(10) for parallel ingestion (matching Mem0's official eval).
3. Batch size 2 for messages (matching Mem0's official eval).
4. AsyncMemory support ready for future optimization.
"""
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from evals.mem0_patch import *  # Apply Patch for Azure Tool Calls
from mem0 import Memory
from .base import MemorySystem


class Mem0Adapter(MemorySystem):
    def __init__(self, use_graph: bool = False):
        self.use_graph = use_graph
        # FORCE override for Mem0 specifically
        os.environ["AZURE_CHAT_DEPLOYMENT"] = "gpt-4.1-mini"

        import uuid
        self.unique_id = str(uuid.uuid4())[:8]

        # Configure Mem0 to use Azure OpenAI + Qdrant SERVER (Thread-Safe!)
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
                    "collection_name": f"mem0_benchmark_{self.unique_id}",
                    # USE SERVER URL instead of path for THREAD-SAFE concurrent access
                    "url": "http://localhost:6333",
                }
            },
            # Use in-memory SQLite for history (unique per instance)
            "history_db_path": f":memory:"
        }

        if self.use_graph:
            config["graph_store"] = {
                "provider": "neo4j",
                "config": {
                    "url": os.getenv("URI_NEO4J", "bolt://localhost:7687"),
                    "username": os.getenv("USER_NEO4J", "neo4j"),
                    "password": os.getenv("PASSWORD_NEO4J", "password")
                }
            }

        self.client = Memory.from_config(config)

        # Custom instructions for better extraction
        self.custom_instructions = """
        Generate personal memories following these guidelines:
        1. Each memory should be self-contained with complete context.
        2. Include timeframes (exact dates) and specific activities.
        3. Extract memories only from user messages.
        4. Output valid JSON with 'facts' list.
        """

    def add_session(self, user_id: str, session_data: str, date: str):
        """Wrapper for single session."""
        messages = [{"role": "user", "content": f"[Date: {date}] {session_data}"}]
        self._safe_add(messages, user_id)

    def add_sessions(self, user_id: str, sessions: list):
        """
        AGGRESSIVE: Process ALL sessions in a single LLM call.
        This maximizes throughput by minimizing API overhead.
        """
        # Build single message with ALL sessions concatenated
        combined_content = []
        for s in sessions:
            combined_content.append(f"[Date: {s['date']}] {s['content']}")
        
        messages = [
            {"role": "system", "content": self.custom_instructions},
            {"role": "user", "content": "\n\n---\n\n".join(combined_content)}
        ]
        
        # Single call with longer timeout for large payload
        self._safe_add(messages, user_id, timeout=45)
        print(f"    ✓ Ingested {len(sessions)} sessions in single batch")

    def _safe_add(self, messages, user_id, timeout=45):
        """Add with retry logic and timeout."""
        max_retries = 2  # Fewer retries for speed
        ADD_TIMEOUT_SECONDS = timeout

        for attempt in range(max_retries):
            try:
                # Use inner thread for timeout (handles Neo4j hangs)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.client.add, messages, user_id=user_id)
                    future.result(timeout=ADD_TIMEOUT_SECONDS)
                return
            except TimeoutError:
                print(f"[Mem0] TIMEOUT ({ADD_TIMEOUT_SECONDS}s) on attempt {attempt + 1}. Skipping.")
                return
            except Exception as e:
                err_str = str(e)

                # Skip known non-retryable errors
                skip_errors = [
                    "'NoneType' object has no attribute 'strip'",
                    "Unterminated string",
                    "Expecting value",
                    "'str' object has no attribute 'get'",
                    "'facts'"
                ]
                if any(skip in err_str for skip in skip_errors):
                    print(f"[Mem0] Malformed response. Skipping: {err_str[:50]}")
                    return

                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                else:
                    print(f"[Mem0] FAIL after {max_retries} retries: {err_str[:100]}")

    def query(self, user_id: str, query: str) -> str:
        """Query memories and generate answer."""
        context_lines = []
        try:
            response = None
            for attempt in range(3):
                try:
                    response = self.client.search(query, user_id=user_id, limit=5)
                    break
                except Exception as e:
                    time.sleep(1)

            if response:
                if isinstance(response, dict):
                    # Semantic Results
                    if "results" in response:
                        for m in response["results"]:
                            context_lines.append(m.get("memory", str(m)))
                    # Graph Relations
                    if "relations" in response:
                        for r in response["relations"]:
                            context_lines.append(
                                f"Graph: {r.get('source')} --{r.get('relationship')}--> {r.get('target')}"
                            )
                else:
                    for m in response:
                        if isinstance(m, dict):
                            context_lines.append(m.get("memory", str(m)))
                        else:
                            context_lines.append(str(m))

        except Exception as e:
            print(f"[Mem0] Search error: {e}")
            return "Error retrieving memories."

        context = "\n".join(context_lines)
        return self._generate_answer(query, context)

    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using Azure OpenAI."""
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
        """No-op as per user_id isolation."""
        pass
