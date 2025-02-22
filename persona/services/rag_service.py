from persona.core.rag_interface import RAGInterface

class RAGService:
    @staticmethod
    async def query(user_id: str, query: str):
        async with RAGInterface(user_id) as rag:
            response = await rag.query(query)
            print("response", response)
            return response