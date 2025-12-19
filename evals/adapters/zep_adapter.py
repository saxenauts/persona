import os
import uuid
import json
import asyncio
import re
import threading
from datetime import datetime
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType, EntityNode
from openai import AsyncAzureOpenAI
from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
from langchain_openai import AzureChatOpenAI
from neo4j import GraphDatabase, AsyncGraphDatabase
from .base import MemorySystem
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient
from tenacity import retry, AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_exception_message
import time
from collections import deque


class CallRateMonitor:
    """Monitor API call rates with rolling window and periodic logging."""
    
    def __init__(self, window_seconds: int = 60, log_interval: int = 30, estimated_tokens_per_call: int = 2000):
        self.window_seconds = window_seconds
        self.log_interval = log_interval
        self.estimated_tokens_per_call = estimated_tokens_per_call
        self.calls = deque()  # (timestamp, call_type) tuples
        self.lock = threading.Lock()
        self.last_log_time = time.time()
        self.total_calls = 0
        self.quota_tpm = 10_000_000  # gpt-5 quota
        
    def record_call(self, call_type: str = "llm"):
        """Record an API call and optionally log stats."""
        now = time.time()
        with self.lock:
            self.calls.append((now, call_type))
            self.total_calls += 1
            
            # Prune old calls outside window
            cutoff = now - self.window_seconds
            while self.calls and self.calls[0][0] < cutoff:
                self.calls.popleft()
            
            # Log every interval
            if now - self.last_log_time >= self.log_interval:
                self._log_stats(now)
                self.last_log_time = now
    
    def _log_stats(self, now: float):
        """Log current call rate statistics."""
        calls_in_window = len(self.calls)
        rpm = calls_in_window * (60.0 / self.window_seconds)
        estimated_tpm = rpm * self.estimated_tokens_per_call
        utilization = (estimated_tpm / self.quota_tpm) * 100
        
        print(f"📊 [RateMonitor] Last {self.window_seconds}s: {calls_in_window} calls | "
              f"RPM: {rpm:.1f} | Est. TPM: {estimated_tpm:,.0f}/{self.quota_tpm:,} ({utilization:.1f}%) | "
              f"Total: {self.total_calls}")


class ThreadSafeRateLimiter:
    """A thread-safe rate limiter using threading.Lock (works across event loops)."""
    
    # Shared monitor across all instances
    _monitor = CallRateMonitor(window_seconds=60, log_interval=30, estimated_tokens_per_call=2000)
    
    def __init__(self, requests_per_second: float):
        self.delay = 1.0 / requests_per_second
        self.lock = threading.Lock()
        self.last_request = 0.0

    def wait(self):
        """Synchronous wait - call this BEFORE entering async context."""
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_request
            wait_time = max(0, self.delay - elapsed)
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_request = time.monotonic()
            # Record this call for monitoring
            ThreadSafeRateLimiter._monitor.record_call("llm")

class CustomNeo4jDriver(Neo4jDriver):
    """Custom driver to increase connection pool size for high-throughput benchmarking."""
    def __init__(self, uri, user, password, database='neo4j'):
        # Re-implement init to set max_connection_pool_size
        self.client = AsyncGraphDatabase.driver(
            uri=uri,
            auth=(user or '', password or ''),
            max_connection_pool_size=500,  # Increased from default 100
            connection_acquisition_timeout=120.0, # Wait longer for connection
        )
        self._database = database
        
        # Schedule indices (same as original)
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.build_indices_and_constraints())
        except RuntimeError:
            pass
        self.aoss_client = None


class ZepAdapter(MemorySystem):
    # Thread-safe global rate limiter
    # GPT-5 TURBO: 160 RPS (target ~9600 RPM)
    _rate_limiter = ThreadSafeRateLimiter(requests_per_second=160.0)


    def _run_async(self, coro):
        """Run a coroutine in a fresh event loop. Always creates new loop to avoid contamination."""
        return asyncio.run(coro)

    def __init__(self):
        # Neo4j Config
        self.neo4j_uri = os.getenv("URI_NEO4J", "bolt://127.0.0.1:7687")
        if "neo4j:7687" in self.neo4j_uri:
            self.neo4j_uri = self.neo4j_uri.replace("neo4j", "127.0.0.1")
        print(f"[ZepAdapter] Configured Neo4j at {self.neo4j_uri}...")
        self.neo4j_user = os.getenv("USER_NEO4J", "neo4j")
        self.neo4j_password = os.getenv("PASSWORD_NEO4J", "password")
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        
        self.generator_llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
            openai_api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_key=os.getenv("AZURE_API_KEY"),
            temperature=0,
        )
        
        # Stage logging for diagnosis
        self.last_stage_logs = {}
        self._create_log_dir()
    
    def _create_log_dir(self):
        """Create directory for detailed stage logs."""
        self.log_dir = "evals/results/zep_stage_logs"
        os.makedirs(self.log_dir, exist_ok=True)
    
    def _log_stage(self, user_id: str, stage: str, data: dict):
        """Log stage data for later diagnosis."""
        self.last_stage_logs[f"{user_id}_{stage}"] = data
        # Also write to file for persistence
        log_file = f"{self.log_dir}/{user_id}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps({"stage": stage, "timestamp": datetime.now().isoformat(), **data}) + "\n")

        


    async def _get_graphiti(self):
        """Create a fresh Graphiti instance for this call (no singleton to avoid cross-loop issues)."""
        
        # Robust ResponseWrapper to handle non-object responses (strings, errors)
        class ResponseWrapper:
            def __init__(self, response):
                self.response = response
            
            @property
            def output_text(self):
                if isinstance(self.response, str):
                    return "{}"
                try:
                    return self.response.choices[0].message.content
                except (AttributeError, IndexError) as e:
                    return "{}"

            @property
            def refusal(self):
                if isinstance(self.response, str):
                    return "Rate limited or error"
                try:
                     return self.response.choices[0].message.refusal
                except:
                     return None

            def model_dump(self):
                if isinstance(self.response, str):
                    return {"error": self.response}
                try:
                    return self.response.model_dump()
                except:
                    return {"raw": str(self.response)}

        async def _patched_create_structured_completion(
            llm_self,
            model: str,
            messages: list,
            temperature: float | None,
            max_tokens: int,
            response_model: type,
            reasoning: str | None = None,
            verbosity: str | None = None,
        ):
            # Synchronous rate limiting BEFORE the async call
            ZepAdapter._rate_limiter.wait()
            
            # Simple retry logic for Azure 429s/timeouts
            async for attempt in AsyncRetrying(
                wait=wait_exponential(multiplier=1, min=2, max=120),
                stop=stop_after_attempt(8),
                retry=retry_if_exception_type(Exception)
            ):
                with attempt:
                    response = await llm_self.client.beta.chat.completions.parse(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_completion_tokens=max_tokens,  # GPT-5 requires max_completion_tokens
                        response_format=response_model,
                    )
                    return ResponseWrapper(response)

        # Apply monkey patch (idempotent - safe to call multiple times)
        AzureOpenAILLMClient._create_structured_completion = _patched_create_structured_completion
        print("[ZepAdapter] Monkey-patched AzureOpenAILLMClient with ThreadSafe Rate Limiting (160 RPS).")

        # Create fresh clients for this event loop
        azure_chat_client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version="2024-08-01-preview",
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
            timeout=300.0,  # Increased timeout for high concurrency
        )
        llm_client = AzureOpenAILLMClient(azure_client=azure_chat_client)
        
        azure_emb_client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version="2023-05-15",
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
            timeout=300.0,  # Increased timeout for high concurrency
        )
        embedder = AzureOpenAIEmbedderClient(
            azure_client=azure_emb_client,
            model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
        )
        
        # Use Custom Driver with large connection pool
        custom_driver = CustomNeo4jDriver(
            uri=self.neo4j_uri,
            user=self.neo4j_user,
            password=self.neo4j_password
        )

        graphiti_client = Graphiti(
            graph_driver=custom_driver,
            llm_client=llm_client,
            embedder=embedder
        )
        return graphiti_client


    async def _close_all(self):
        """No-op - clients are created fresh per call and cleaned up by asyncio.run()."""
        pass


    def add_session(self, user_id: str, session_data: str, date: str):
        self.add_sessions(user_id, [{"content": session_data, "date": date}])

    def add_sessions(self, user_id: str, sessions: list):
        """Ingest sessions SEQUENTIALLY to avoid event loop corruption from nest_asyncio.
        
        Key design decisions:
        - Sequential processing (no asyncio.gather) for stability with nest_asyncio.
        - Fail-fast: Raises immediately on first error to ensure data integrity.
        - Progress logging: Prints clear progress markers every 5 sessions.
        """
        safe_user_id = re.sub(r'[^a-zA-Z0-9_-]', '_', user_id)
        
        async def _ingest_all():
            client = await self._get_graphiti()
            
            # GPT-5 TURBO: 50 workers (x 5 parallel qs = 250 total) to eliminate DB lock contention
            MAX_CONCURRENT_EPISODES = 50
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_EPISODES)

            async def _ingest_one(idx, s):
                async with semaphore:
                    # Sync rate limit still applies globally across threads
                    self._rate_limiter.wait()
                    
                    try:
                        episode_date = datetime.strptime(s['date'], "%Y-%m-%d")
                    except:
                        episode_date = datetime.utcnow()
                    
                    await client.add_episode(
                        name=f"Session on {s['date']}",
                        episode_body=s['content'],
                        source=EpisodeType.text,
                        source_description="chat history + " + s['date'],
                        reference_time=episode_date,
                        group_id=safe_user_id
                    )
                    # Simple progress marker in logs
                    if (idx + 1) % 5 == 0 or (idx + 1) == len(sessions):
                         print(f"    [ZepAdapter] Progress: {idx+1}/{len(sessions)} sessions for {safe_user_id}...")

            tasks = [_ingest_one(i, s) for i, s in enumerate(sessions)]
            await asyncio.gather(*tasks)
            
            # Final success summary
            print(f"    [ZepAdapter] ✅ Ingested ALL {len(sessions)} sessions for {safe_user_id}")
            self._log_stage(safe_user_id, "stage1_ingestion_complete", {
                "total_sessions": len(sessions),
                "success_count": len(sessions),
                "failed_count": 0
            })

        self._run_async(_ingest_all())

    def _format_edge_date_range(self, edge) -> str:
        # Handle valid_at/invalid_at which might be None or datetime strings
        valid = edge.valid_at if edge.valid_at else 'date unknown'
        invalid = edge.invalid_at if edge.invalid_at else 'present'
        return f"{valid} - {invalid}"

    def _compose_search_context(self, edges: list, nodes: list) -> str:
        TEMPLATE = """
FACTS and ENTITIES represent relevant context to the current conversation.

# These are the most relevant facts and their valid date ranges. If the fact is about an event, the event takes place during this time.
# format: FACT (Date range: from - to)
<FACTS>
{facts}
</FACTS>

# These are the most relevant entities
# ENTITY_NAME: entity summary
<ENTITIES>
{entities}
</ENTITIES>
"""
        facts = [f'  - {edge.fact} ({self._format_edge_date_range(edge)})' for edge in edges]
        entities = [f'  - {node.name}: {node.summary}' for node in nodes]
        return TEMPLATE.format(facts='\n'.join(facts), entities='\n'.join(entities))

    def query(self, user_id: str, query: str) -> str:
        safe_user_id = re.sub(r'[^a-zA-Z0-9_-]', '_', user_id)
        
        async def _run_rag():
            client = await self._get_graphiti()
            try:
                # 1. Search
                results = await client.search(
                    query=query,
                    group_ids=[safe_user_id],
                    num_results=20
                )
                
                # 2. Fetch Nodes
                node_uuids = set()
                for edge in results:
                    if edge.source_node_uuid: node_uuids.add(edge.source_node_uuid)
                    if edge.target_node_uuid: node_uuids.add(edge.target_node_uuid)
                
                nodes = []
                if node_uuids:
                    nodes = await EntityNode.get_by_uuids(client.driver, list(node_uuids))
                
                # 3. Compose Context
                context = self._compose_search_context(results, nodes)
                
                # 4. Global Rate Limiting for Generation (sync method - no await)
                self._rate_limiter.wait()
                
                # RAG Generation
                user_prompt = f"""
                Your task is to briefly answer the question. You are given the following context from the previous conversation. If you don't know how to answer the question, abstain from answering.
                    <CONTEXT>
                    {context}
                    </CONTEXT>
                    <QUESTION>
                    {query}
                    </QUESTION>

                Answer:
                """
                
                # Use ainstoke if available or just run in executor
                response = await self.generator_llm.ainvoke(user_prompt)
                answer = response.content
                
                self._log_stage(safe_user_id, "stage4_generation", {
                    "query": query,
                    "answer": answer,
                    "edges_count": len(results),
                    "nodes_count": len(nodes)
                })
                
                return answer
            except Exception as e:
                print(f"[ZepAdapter] RAG error for {safe_user_id}: {e}")
                self._log_stage(safe_user_id, "stage3_retrieval", {
                    "query": query,
                    "status": "failed",
                    "error": str(e)
                })
                return f"Error: {e}"

        return self._run_async(_run_rag())

    def reset(self, user_id: str):
        # Clear graph for user.
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) WHERE n.group_id = $uid DETACH DELETE n", uid=user_id)
        except Exception as e:
            print(f"[ZepAdapter] Reset error: {e}")

