from typing import List, Dict, Any
from luna9.core.graph_ops import GraphOps, GraphContextRetriever
from luna9.llm.llm_graph import generate_response_with_context
from luna9.models.schema import Node

class RAGInterface:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.graph_ops = GraphOps()
        self.graph_context_retriever = GraphContextRetriever(self.graph_ops)

    async def get_context(self, query: str, top_k: int = 5, max_hops: int = 2) -> str:
        similar_nodes = await self.graph_ops.text_similarity_search(query=query, user_id=self.user_id, limit=top_k)
        nodes = [Node(name=node['nodeName']) for node in similar_nodes['results']]
        context = await self.graph_context_retriever.get_relevant_graph_context(user_id=self.user_id, nodes=nodes, max_hops=max_hops)
        return context


    async def query(self, query: str) -> str:
        context = await self.get_context(query)
        response = await generate_response_with_context(query, context)
        return response

    async def close(self):
        await self.graph_ops.close()

    async def query_vector_only(self, query: str) -> str:
        similar_nodes = await self.graph_ops.text_similarity_search(query=query, user_id=self.user_id, limit=5)
        nodes = str([Node(name=node['nodeName']) for node in similar_nodes['results']])
        response = await generate_response_with_context(query, nodes)
        return response
    
    async def format_vector_context(self, similar_nodes: List[Dict[str, Any]]) -> str:
        formatted = "# Vector Search Context\n\n"
        for node in similar_nodes:
            formatted += f"## {node['nodeName']}\n"
            formatted += f"Similarity Score: {node['score']}\n"
            node_data = await self.graph_ops.get_node_data(node['nodeName'], self.user_id)
            formatted += f"Perspective: {node_data.perspective}\n"
            formatted += f"Properties: {', '.join([f'{k}: {v}' for k, v in node_data.properties.items()])}\n\n"
        print(f"Vector context: {formatted}")
        return formatted