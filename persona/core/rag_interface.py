from typing import List, Dict, Any
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
        nodes = [Node(name=node['nodeName']) for node in similar_nodes['results']]
        logger.debug(f"Nodes for RAG query: {nodes}")
        context = await self.graph_context_retriever.get_relevant_graph_context(user_id=self.user_id, nodes=nodes, max_hops=max_hops)
        return context

    async def query(self, query: str) -> str:
        if not self.graph_ops:
            await self.__aenter__()
        context = await self.get_context(query)
        logger.debug(f"Context for RAG query: {context}")
        response = await generate_response_with_context(query, context)
        return response

    async def close(self):
        await self.__aexit__(None, None, None)

    async def query_vector_only(self, query: str) -> str:
        if not self.graph_ops:
            await self.__aenter__()
        similar_nodes = await self.graph_ops.text_similarity_search(query=query, user_id=self.user_id, limit=5)
        nodes = str([Node(name=node['nodeName']) for node in similar_nodes['results']])
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
            formatted += f"Perspective: {node_data.perspective}\n"
            formatted += f"Properties: {', '.join([f'{k}: {v}' for k, v in node_data.properties.items()])}\n\n"
        logger.debug(f"Formatted vector context: {formatted}")
        return formatted