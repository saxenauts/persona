from persona.core.rag_interface import RAGInterface
from persona.core.graph_ops import GraphOps, GraphContextRetriever
from server.logging_config import get_logger

logger = get_logger(__name__)

class RAGService:
    @staticmethod
    async def query(user_id: str, query: str, graph_ops: GraphOps):
        rag = RAGInterface(user_id)
        rag.graph_ops = graph_ops
        rag.graph_context_retriever = GraphContextRetriever(graph_ops)
        
        response = await rag.query(query)
        logger.debug(f"RAG service response: {response}")
        return response