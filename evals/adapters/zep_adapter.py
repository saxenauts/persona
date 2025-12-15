import os
import uuid
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
        print(f"[ZepAdapter] Connecting to Neo4j at {self.neo4j_uri}...")
        self.neo4j_user = os.getenv("USER_NEO4J", "neo4j")
        self.neo4j_password = os.getenv("PASSWORD_NEO4J", "password")
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        
        # Azure Config
        # Zep expects an initialized AsyncAzureOpenAI client
        self.azure_client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version="2024-08-01-preview",
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
        )
        self.llm_client = AzureOpenAILLMClient(
            azure_client=self.azure_client
        )
        
        # Initialize Graphiti
        self.graphiti = Graphiti(
            uri=self.neo4j_uri,
            user=self.neo4j_user,
            password=self.neo4j_password,
            llm_client=self.llm_client
        )
        
        # We need a separate LLM for the final answer generation to be fair (Generation Step)
        self.generator_llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
            openai_api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_key=os.getenv("AZURE_API_KEY"),
            temperature=0,
        )

    def add_session(self, user_id: str, session_data: str, date: str):
        import asyncio
        # Graphiti Ingestion
        # It needs 'episodes'.
        # We simulate an episode.
        
        # Parse date to datetime
        try:
            episode_date = datetime.strptime(date, "%Y-%m-%d")
        except:
            episode_date = datetime.utcnow()

        asyncio.run(self.graphiti.add_episode(
            name=f"Session on {date}",
            episode_body=session_data,
            source=EpisodeType.text,
            source_description="chat history",
            reference_time=episode_date,
            group_id=user_id 
        ))

    def add_sessions(self, user_id: str, sessions: list):
        import asyncio
        
        async def _ingest_all():
            tasks = []
            for s in sessions:
                try:
                    episode_date = datetime.strptime(s['date'], "%Y-%m-%d")
                except:
                    episode_date = datetime.utcnow()
                    
                tasks.append(self.graphiti.add_episode(
                    name=f"Session on {s['date']}",
                    episode_body=s['content'],
                    source=EpisodeType.text,
                    source_description="chat history + " + s['date'],
                    reference_time=episode_date,
                    group_id=user_id
                ))
            await asyncio.gather(*tasks)
            
        asyncio.run(_ingest_all())

    def query(self, user_id: str, query: str) -> str:
        import asyncio
        # Graphiti 'search' returns chunks/facts.
        # It is async
        stopwords = [] # Optional
        
        # We need to run search in asyncio loop
        results = asyncio.run(self.graphiti.search(
            query=query,
            group_ids=[user_id],
            num_results=5
        ))
        
        context = "\n".join([r.fact for r in results]) # Assuming EntityEdge has 'fact' or similar. 
        # Check Graphiti result structure. Returns 'EntityEdge' object.
        # graphiti.py search returns 'list[EntityEdge]'.
        # EntityEdge probably has 'fact' string field.

        
        # RAG Generation
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        response = self.generator_llm.invoke(prompt)
        return response.content

    def reset(self, user_id: str):
        # Clear graph for user.
        # Graphiti might not have a clean 'delete_user' yet, so we cypher delete.
        with self.driver.session() as session:
            session.run("MATCH (n) WHERE n.group_id = $uid DETACH DELETE n", uid=user_id)
