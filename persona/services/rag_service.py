from persona.core.rag_interface import RAGInterface
from persona.core.graph_ops import GraphOps, GraphContextRetriever

class RAGService:
    @staticmethod
    async def query(user_id: str, query: str, graph_ops: GraphOps):
        rag = RAGInterface(user_id)
        rag.graph_ops = graph_ops
        rag.graph_context_retriever = GraphContextRetriever(graph_ops)
        
        response = await rag.query(query)
        print("response", response)
        return response