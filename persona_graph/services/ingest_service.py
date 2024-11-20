from persona_graph.core.constructor import GraphConstructor
from persona_graph.models.schema import UnstructuredData
from persona_graph.core.graph_ops import GraphOps

class IngestService:
    @staticmethod
    async def ingest_data(user_id: str, content: str):
        async with GraphConstructor(user_id) as constructor:
            data = UnstructuredData(title="Ingested Data", content=content)
            await constructor.process_unstructured_data(data)
        
        return {"message": "Data ingested successfully"}