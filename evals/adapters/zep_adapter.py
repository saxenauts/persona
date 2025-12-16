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
        
    def _get_graphiti(self):
        # Initialize clients inside the current loop context
        azure_client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version="2024-08-01-preview",
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
        )

        async def _patched_create_structured_completion(self_instance, model, messages, temperature, max_tokens, response_model, reasoning=None, verbosity=None):
            response = await self_instance.client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            parsed = response.choices[0].message.parsed
            
            class ParsedInterceptor:
                def __init__(self, wrapped):
                    self._wrapped = wrapped
                    self.output_text = ""
                
                def __getattr__(self, name):
                    return getattr(self._wrapped, name)
                    
                def model_dump(self, *args, **kwargs):
                    return self._wrapped.model_dump(*args, **kwargs)
                
                def dict(self, *args, **kwargs):
                     return self._wrapped.dict(*args, **kwargs)

            return ParsedInterceptor(parsed)
        
        # Apply patch
        AzureOpenAILLMClient._create_structured_completion = _patched_create_structured_completion

        llm_client = AzureOpenAILLMClient(azure_client=azure_client)
        
        return Graphiti(
            uri=self.neo4j_uri,
            user=self.neo4j_user,
            password=self.neo4j_password,
            llm_client=llm_client
        )

    def add_session(self, user_id: str, session_data: str, date: str):
        import asyncio
        client = self._get_graphiti()
        
        try:
            episode_date = datetime.strptime(date, "%Y-%m-%d")
        except:
            episode_date = datetime.utcnow()
 
        asyncio.run(client.add_episode(
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
            await asyncio.gather(*tasks)
            
        asyncio.run(_ingest_all())
 
    def query(self, user_id: str, query: str) -> str:
        import asyncio
        client = self._get_graphiti()
        
        # Graphiti 'search' returns chunks/facts.
        # It is async
        stopwords = [] # Optional
        
        # We need to run search in asyncio loop
        results = asyncio.run(client.search(
            query=query,
            group_ids=[user_id],
            num_results=5
        ))
        
        context = "\n".join([r.fact for r in results]) # Assuming EntityEdge has 'fact' or similar. 
        # Check Graphiti result structure. Returns 'list[EntityEdge]'.
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
