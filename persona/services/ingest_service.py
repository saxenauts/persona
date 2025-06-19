from persona.core.constructor import GraphConstructor
from persona.core.graph_ops import GraphOps, GraphContextRetriever
from persona.models.schema import UnstructuredData

class IngestService:
    @staticmethod
    async def ingest_data(user_id: str, data: UnstructuredData, graph_ops: GraphOps):
        """
        Ingest unstructured data into the graph.
        """
        constructor = GraphConstructor(user_id)
        constructor.graph_ops = graph_ops
        constructor.graph_context_retriever = GraphContextRetriever(graph_ops)
        
        await constructor.ingest_unstructured_data_to_graph(data)
        
        return {"message": "Data ingested successfully"}