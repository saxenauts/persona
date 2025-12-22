from persona.core.interfaces import GraphDatabase, VectorStore
from persona.llm.embeddings import generate_embeddings_async
from typing import List, Dict, Any, Optional
from server.logging_config import get_logger

logger = get_logger(__name__)


class GraphOps:
    """
    Graph operations layer that abstracts database backends.
    Uses GraphDatabase for graph operations and VectorStore for similarity search.
    
    This is the low-level data access layer. For high-level memory operations,
    use MemoryStore instead.
    """
    
    def __init__(
        self, 
        graph_db: Optional[GraphDatabase] = None,
        vector_store: Optional[VectorStore] = None
    ):
        """
        Initialize GraphOps with database backends.
        
        Args:
            graph_db: GraphDatabase implementation (defaults to Neo4j)
            vector_store: VectorStore implementation (defaults to Neo4j)
        """
        if graph_db is None or vector_store is None:
            from persona.core.factory import create_backends
            default_graph, default_vector = create_backends("neo4j")
            self.graph_db = graph_db or default_graph
            self.vector_store = vector_store or default_vector
        else:
            self.graph_db = graph_db
            self.vector_store = vector_store

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def initialize(self):
        """Initialize database connections."""
        await self.graph_db.initialize()
        await self.vector_store.initialize()

    async def close(self):
        """Close database connections."""
        logger.info("Closing database connections...")
        await self.graph_db.close()
        await self.vector_store.close()

    async def clean_graph(self):
        """Delete all graph data."""
        await self.graph_db.clean_graph()

    # -------------------------------------------------------------------------
    # User Management
    # -------------------------------------------------------------------------

    async def create_user(self, user_id: str) -> None:
        await self.graph_db.create_user(user_id)

    async def delete_user(self, user_id: str) -> None:
        await self.graph_db.delete_user(user_id)

    async def user_exists(self, user_id: str) -> bool:
        return await self.graph_db.user_exists(user_id)

    # -------------------------------------------------------------------------
    # Similarity Search
    # -------------------------------------------------------------------------

    async def text_similarity_search(
        self, 
        query: str, 
        user_id: str, 
        limit: int = 5, 
        index_name: str = "embeddings_index"
    ) -> Dict[str, Any]:
        """Perform similarity search on the graph based on a text query."""
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot perform similarity search.")
            return {"query": query, "results": []}

        logger.debug(f"Generating embedding for query: '{query}' for user ID: '{user_id}'")
        query_embeddings = await generate_embeddings_async([query])
        
        if not query_embeddings[0]:
            return {"query": query, "results": []}

        logger.debug(f"Performing similarity search for the query: '{query}' for user ID: '{user_id}'")
        results = await self.vector_store.search_similar(query_embeddings[0], user_id, limit)

        return {
            "query": query,
            "results": [
                {
                    "nodeName": result["node_name"],
                    "score": result["score"]
                } 
                for result in results
            ]
        }

    async def perform_similarity_search(
        self, 
        query: str, 
        embedding: List[float],
        user_id: str, 
        limit: int = 5
    ) -> Dict[str, Any]:
        """Perform similarity search using pre-computed embedding."""
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot perform similarity search.")
            return {"query": query, "results": []}

        try:
            logger.debug(f"Performing similarity search for: {query}")
            results = await self.vector_store.search_similar(embedding, user_id, limit)
            logger.debug(f"Found {len(results)} similar nodes for '{query}'")
            
            return {
                "query": query,
                "results": [
                    {
                        "nodeName": result["node_name"],
                        "score": result["score"]
                    } 
                    for result in results
                ]
            }
        except Exception as e:
            logger.error(f"Error in similarity search for {query}: {str(e)}")
            return {"query": query, "results": []}
