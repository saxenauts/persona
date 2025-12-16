import os
import uuid
import json
import asyncio
from datetime import datetime
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from openai import AsyncAzureOpenAI
from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
from langchain_openai import AzureChatOpenAI
from neo4j import GraphDatabase
from .base import MemorySystem

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
        # Initialize clients inside the current loop context
        azure_client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version="2024-08-01-preview",
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
        )
        
        llm_client = AzureOpenAILLMClient(azure_client=azure_client)
        
        return Graphiti(
            uri=self.neo4j_uri,
            user=self.neo4j_user,
            password=self.neo4j_password,
            llm_client=llm_client
        )

    def add_session(self, user_id: str, session_data: str, date: str):
        client = self._get_graphiti()
        
        try:
            episode_date = datetime.strptime(date, "%Y-%m-%d")
        except:
            episode_date = datetime.utcnow()
 
        try:
            asyncio.run(client.add_episode(
                name=f"Session on {date}",
                episode_body=session_data,
                source=EpisodeType.text,
                source_description="chat history",
                reference_time=episode_date,
                group_id=user_id 
            ))
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

    def add_sessions(self, user_id: str, sessions: list):
        async def _ingest_all():
            client = self._get_graphiti()
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
            
        asyncio.run(_ingest_all())

    def query(self, user_id: str, query: str) -> str:
        client = self._get_graphiti()
        
        try:
            # We need to run search in asyncio loop
            results = asyncio.run(client.search(
                query=query,
                group_ids=[user_id],
                num_results=10 # Increased from 5 to improve recall
            ))
            
            context_facts = [r.fact for r in results]
            context = "\n".join(context_facts)
            
            self._log_stage(user_id, "stage3_retrieval", {
                "query": query,
                "retrieved_facts_count": len(context_facts),
                "context_preview": context[:500] if context else "EMPTY",
                "facts": context_facts[:5]
            })
            
            # RAG Generation
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            response = self.generator_llm.invoke(prompt)
            answer = response.content
            
            self._log_stage(user_id, "stage4_generation", {
                "query": query,
                "context_len": len(context),
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

