from persona_graph.core.rag_interface import RAGInterface

class RAGService:
    @staticmethod
    async def query(user_id: str, query: str):
        rag = RAGInterface(user_id)
        response= await rag.query(query)
        print("response", response)
        return response