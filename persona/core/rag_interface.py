from typing import List, Dict, Any, Optional
from persona.core.graph_ops import GraphOps, GraphContextRetriever
from persona.llm.llm_graph import generate_response_with_context
from persona.models.schema import Node
from server.logging_config import get_logger

logger = get_logger(__name__)

class RAGInterface:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.graph_ops = None
        self.graph_context_retriever = None
        self._memory_store = None

    async def __aenter__(self):
        self.graph_ops = await GraphOps().__aenter__()
        self.graph_context_retriever = GraphContextRetriever(self.graph_ops)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.graph_ops:
            await self.graph_ops.__aexit__(exc_type, exc_val, exc_tb)

    async def get_context(self, query: str, top_k: int = 5, max_hops: int = 2) -> str:
        if not self.graph_ops:
            await self.__aenter__()
        similar_nodes = await self.graph_ops.text_similarity_search(query=query, user_id=self.user_id, limit=top_k)
        results = similar_nodes.get('results', [])
        nodes = [Node(name=item['nodeName'], type="Unknown") for item in results]
        seed_scores = {item['nodeName']: float(item.get('score', 0.0)) for item in results}

        # Analyze query for retrieval routing
        filt_types, qdate, hops = self._analyze_query(query, default_hops=max_hops)
        logger.info(f"Nodes for RAG query: {nodes}; filter_types={filt_types}; qdate={qdate}; hops={hops}")
        context = await self.graph_context_retriever.get_relevant_graph_context(
            user_id=self.user_id,
            nodes=nodes,
            max_hops=hops,
            filter_types=filt_types,
            seed_scores=seed_scores,
            question_date=qdate,
        )
        return context

    async def query(self, query: str) -> str:
        if not self.graph_ops:
            await self.__aenter__()
        context = await self.get_context(query)
        logger.info(f"Context for RAG query: {context}")
        response = await generate_response_with_context(query, context)
        return response

    async def close(self):
        await self.__aexit__(None, None, None)

    async def query_vector_only(self, query: str) -> str:
        if not self.graph_ops:
            await self.__aenter__()
        similar_nodes = await self.graph_ops.text_similarity_search(query=query, user_id=self.user_id, limit=5)
        nodes = str([Node(name=node['nodeName'], type="Unknown") for node in similar_nodes.get('results', [])])
        logger.debug(f"Vector context for RAG query: {nodes}")
        response = await generate_response_with_context(query, nodes)
        return response
    
    async def format_vector_context(self, similar_nodes: List[Dict[str, Any]]) -> str:
        if not self.graph_ops:
            await self.__aenter__()
        formatted = "# Vector Search Context\n\n"
        for node in similar_nodes:
            formatted += f"## {node['nodeName']}\n"
            formatted += f"Similarity Score: {node['score']}\n"
            node_data = await self.graph_ops.get_node_data(node['nodeName'], self.user_id)
            formatted += f"Type: {node_data.type}\n"
            formatted += f"Properties: {', '.join([f'{k}: {v}' for k, v in node_data.properties.items()])}\n\n"
        logger.debug(f"Formatted vector context: {formatted}")
        return formatted

    def _analyze_query(self, query: str, default_hops: int = 2):
        """
        Lightweight heuristic routing for retrieval.
        Returns (filter_types, question_date, hops)
        """
        import re
        filter_types = set()
        hops = default_hops

        q = query.lower()
        # Extract (date: YYYY/.. or YYYY-..)
        m = re.search(r"\(date:\s*([0-9]{4}[/-][0-9]{2}[/-][0-9]{2}(?:[^)]*)?)\)", query)
        question_date = m.group(1).strip() if m else None

        # Preference-focused
        if any(tok in q for tok in ["favorite", "favourite", "prefer", "likes", "dislikes", "preference"]):
            filter_types.add("Preference")
            hops = min(hops, 1)

        # Temporal cues – if date present or words implying time ordering
        if question_date or any(tok in q for tok in ["before", "after", "first", "last", "earlier", "later", "when", "date", "days", "weeks", "months"]):
            # no change to hops yet; date will be used for scoring
            pass

        # Assistant-sourced hints – future: filter by properties.source == 'Assistant'
        # (Not applied yet due to property-level filter not implemented in traversal)

        return (filter_types if filter_types else None, question_date, hops)

    # ========== V2 Memory Engine: get_user_context ==========
    
    async def get_user_context(
        self,
        current_conversation: Optional[str] = None,
        include_goals: bool = True,
        include_psyche: bool = True,
        include_previous_episode: bool = True,
        max_episodes: int = 5,
        max_goals: int = 10,
        max_psyche: int = 10
    ) -> str:
        """
        Compose structured context from memory layers using the universal XML formatter.
        """
        from persona.core.memory_store import MemoryStore
        from persona.core.backends.neo4j_graph import Neo4jGraphDatabase
        from persona.core.context import format_memories_for_llm
        
        # Initialize memory store if needed
        if not self._memory_store:
            graph_db = Neo4jGraphDatabase()
            await graph_db.initialize()
            self._memory_store = MemoryStore(graph_db)
        
        all_memories = []
        
        # 1. Previous Episodes
        if include_previous_episode:
            episodes = await self._memory_store.get_recent(
                self.user_id, 
                memory_type="episode", 
                limit=max_episodes
            )
            all_memories.extend(episodes)
        
        # 2. Active Goals
        if include_goals:
            goals = await self._memory_store.get_by_type("goal", self.user_id, limit=max_goals)
            all_memories.extend(goals)
        
        # 3. Psyche (traits, preferences)
        if include_psyche:
            psyche = await self._memory_store.get_by_type("psyche", self.user_id, limit=max_psyche)
            all_memories.extend(psyche)
        
        # Generate XML context
        context = format_memories_for_llm(all_memories)
        
        # 4. Current Conversation (Keep as raw text outside XML context for now)
        if current_conversation:
            context = f"{context}\n\n<current_conversation>\n{current_conversation}\n</current_conversation>"
        
        logger.info(f"Generated universal user context: {len(all_memories)} memories, {len(context)} chars")
        return context

