from luna9.core.constructor import GraphConstructor
from luna9.models.schema import UnstructuredData

class IngestService:
    @staticmethod
    async def ingest_data(user_id: str, content: UnstructuredData):
        """
        Ingest unstructured data into the graph.
        """
        async with GraphConstructor(user_id) as constructor:
            data = UnstructuredData(title="Ingested Data", content=content)
            await constructor.ingest_unstructured_data_to_graph(data)
        
        return {"message": "Data ingested successfully"}