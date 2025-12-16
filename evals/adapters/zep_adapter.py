import os
import uuid
import json
import asyncio
from datetime import datetime
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType, EntityNode
from openai import AsyncAzureOpenAI
from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
from langchain_openai import AzureChatOpenAI
from neo4j import GraphDatabase
from .base import MemorySystem
from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient
from openai import AsyncAzureOpenAI

class ZepAdapter(MemorySystem):
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
        


    def _get_graphiti(self):
        # MONKEY PATCH: Graphiti's Azure client uses a non-existent 'responses.parse' endpoint.
        # We replace it with the standard 'beta.chat.completions.parse' implementation from OpenAIClient.
        # AND we wrap the response because Graphiti expects an object with .output_text property (JSON string).
        
        class ResponseWrapper:
            def __init__(self, response):
                self.response = response
            
            @property
            def output_text(self):
                # Graphiti expects JSON string in output_text
                return self.response.choices[0].message.content

            @property
            def refusal(self):
                 return self.response.choices[0].message.refusal

            def model_dump(self):
                return self.response.model_dump()

        async def _patched_create_structured_completion(
            self,
            model: str,
            messages: list,
            temperature: float | None,
            max_tokens: int,
            response_model: type,
            reasoning: str | None = None,
            verbosity: str | None = None,
        ):
            # Azure OpenAI supports standard beta.chat.completions.parse
            response = await self.client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_model,
            )
            return ResponseWrapper(response)

        AzureOpenAILLMClient._create_structured_completion = _patched_create_structured_completion
        print("[ZepAdapter] Monkey-patched AzureOpenAILLMClient to use beta.chat.completions.parse and ResponseWrapper.")

        # 1. Chat Client (for Graph construction & RAG)
        azure_chat_client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version="2024-08-01-preview", # Force specific version compatible with Graphiti Structured Outputs
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
        )
        llm_client = AzureOpenAILLMClient(azure_client=azure_chat_client)
        
        # 2. Embedding Client (for Vector Search)
        azure_emb_client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version="2023-05-15", # Embeddings usually work with older/stable API versions
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        )
        embedder = AzureOpenAIEmbedderClient(
            azure_client=azure_emb_client,
            model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
        )
        
        return Graphiti(
            uri=self.neo4j_uri,
            user=self.neo4j_user,
            password=self.neo4j_password,
            llm_client=llm_client,
            embedder=embedder
        )

    async def _close_client(self, client):
        """Helper to close Graphiti internal clients."""
        try:
            # print(f"[ZepAdapter] Closing clients...")
            if hasattr(client, "driver"):
                await client.driver.close()
            if hasattr(client.llm_client, "client"):
                await client.llm_client.client.close()
            if hasattr(client.embedder, "azure_client"):
                await client.embedder.azure_client.close()
        except Exception as e:
            print(f"[ZepAdapter] Close error: {e}")

    def add_session(self, user_id: str, session_data: str, date: str):
        async def _ingest():
            client = self._get_graphiti()
            try:
                try:
                    episode_date = datetime.strptime(date, "%Y-%m-%d")
                except:
                    episode_date = datetime.utcnow()
    
                await client.add_episode(
                    name=f"Session on {date}",
                    episode_body=session_data,
                    source=EpisodeType.text,
                    source_description="chat history",
                    reference_time=episode_date,
                    group_id=user_id 
                )
                self._log_stage(user_id, "stage1_ingestion", {
                    "date": date, 
                    "status": "success",
                    "content_preview": session_data[:100]
                })
            except Exception as e:
                print(f"[ZepAdapter] Ingest error: {e}")
                self._log_stage(user_id, "stage1_ingestion", {
                    "date": date, 
                    "status": "failed",
                    "error": str(e)
                })
            finally:
                await self._close_client(client)

        asyncio.run(_ingest())

    def add_sessions(self, user_id: str, sessions: list):
        async def _ingest_all():
            client = self._get_graphiti()
            try:
                tasks = []
                for s in sessions:
                    try:
                        episode_date = datetime.strptime(s['date'], "%Y-%m-%d")
                    except:
                        episode_date = datetime.utcnow()
                        
                    tasks.append(client.add_episode(
                        name=f"Session on {s['date']}",
                        episode_body=s['content'],
                        source=EpisodeType.text,
                        source_description="chat history + " + s['date'],
                        reference_time=episode_date,
                        group_id=user_id
                    ))
                
                # Run concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Log results
                success_count = sum(1 for r in results if not isinstance(r, Exception))
                self._log_stage(user_id, "stage1_ingestion_batch", {
                    "total_sessions": len(sessions),
                    "success_count": success_count,
                    "failed_count": len(sessions) - success_count,
                    "errors": [str(r) for r in results if isinstance(r, Exception)]
                })
            finally:
                await self._close_client(client)
            
        asyncio.run(_ingest_all())

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
        async def _run_search():
            client = self._get_graphiti()
            try:
                # We need to run search in asyncio loop
                # Graphiti search returns list[EntityEdge]
                results = await client.search(
                    query=query,
                    group_ids=[user_id],
                    num_results=20 # Increased to match notebook limit
                )
                
                # Extract Edges
                edges = results
                
                # Extract Unique Node UUIDs
                node_uuids = set()
                for edge in edges:
                    if edge.source_node_uuid:
                        node_uuids.add(edge.source_node_uuid)
                    if edge.target_node_uuid:
                        node_uuids.add(edge.target_node_uuid)
                
                # Fetch Nodes from DB
                nodes = []
                if node_uuids:
                    nodes = await EntityNode.get_by_uuids(client.driver, list(node_uuids))
                
                return edges, nodes
            finally:
                await self._close_client(client)

        try:
            edges, nodes = asyncio.run(_run_search())
            
            # Compose Context using official template
            context = self._compose_search_context(edges, nodes)
            
            self._log_stage(user_id, "stage3_retrieval", {
                "query": query,
                "edges_count": len(edges),
                "nodes_count": len(nodes),
                "context_preview": context[:500] if context else "EMPTY"
            })
            
            # RAG Generation
            # Use specific prompt to ensure usage of context
            system_prompt = "You are a helpful expert assistant answering questions based on the provided context."
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
            
            response = self.generator_llm.invoke(user_prompt)
            answer = response.content
            
            self._log_stage(user_id, "stage4_generation", {
                "query": query,
                "answer": answer
            })
            
            return answer
            
        except Exception as e:
            print(f"[ZepAdapter] Query error: {e}")
            self._log_stage(user_id, "stage3_retrieval", {
                "query": query,
                "status": "failed",
                "error": str(e)
            })
            return f"Error: {e}"

    def reset(self, user_id: str):
        # Clear graph for user.
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) WHERE n.group_id = $uid DETACH DELETE n", uid=user_id)
        except Exception as e:
            print(f"[ZepAdapter] Reset error: {e}")

