from persona.core.constructor import GraphConstructor
from persona.core.graph_ops import GraphOps, GraphContextRetriever
from persona.models.schema import UnstructuredData
from server.logging_config import get_logger

logger = get_logger(__name__)

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

    @staticmethod
    async def ingest_batch(user_id: str, batch: list[UnstructuredData], graph_ops: GraphOps):
        """
        Ingest a batch of unstructured data in parallel (controlled concurrency).
        """
        import asyncio
        
        constructor = GraphConstructor(user_id)
        constructor.graph_ops = graph_ops
        constructor.graph_context_retriever = GraphContextRetriever(graph_ops)
        
        # Concurrency Limit - Increased to 80 (Extreme Batch) for 1M TPM Quota
        sem = asyncio.Semaphore(80)
        
        async def _safe_ingest(item_data):
            async with sem:
                # We can't easily catch errors per item without making the whole batch partial.
                # For now, let's fail fast or log? 
                # Benchmarking prefers fail fast to know if something broke.
                try:
                    await constructor.ingest_unstructured_data_to_graph(item_data)
                    logger.info(f"Item processed for user {user_id}")
                except Exception as e:
                    logger.error(f"Failed to process item for user {user_id}: {e}")
                    raise
        
        tasks = [_safe_ingest(item) for item in batch]
        await asyncio.gather(*tasks)
        
        return {"message": f"Successfully ingested batch of {len(batch)} items"}