"""Factory functions for creating database backend instances."""

from typing import Tuple
from persona.core.interfaces import GraphDatabase, VectorStore


def create_backends(backend_type: str = "neo4j") -> Tuple[GraphDatabase, VectorStore]:
    """Create graph and vector database instances based on config.
    
    Args:
        backend_type: Type of backend to use ("neo4j", "falkor", etc.)
        
    Returns:
        Tuple of (GraphDatabase, VectorStore) instances
    """
    if backend_type == "neo4j":
        from persona.core.backends.neo4j_graph import Neo4jGraphDatabase
        from persona.core.backends.neo4j_vector import Neo4jVectorStore
        
        graph_db = Neo4jGraphDatabase()
        # Share the driver between graph and vector for efficiency
        vector_store = Neo4jVectorStore(graph_driver=None)  # Will use graph_db.driver after init
        
        return graph_db, vector_store
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


async def create_and_initialize_backends(backend_type: str = "neo4j") -> Tuple[GraphDatabase, VectorStore]:
    """Create and initialize database instances.
    
    This is the main entry point for getting ready-to-use backends.
    
    Args:
        backend_type: Type of backend to use
        
    Returns:
        Tuple of initialized (GraphDatabase, VectorStore) instances
    """
    if backend_type == "neo4j":
        from persona.core.backends.neo4j_graph import Neo4jGraphDatabase
        from persona.core.backends.neo4j_vector import Neo4jVectorStore
        
        graph_db = Neo4jGraphDatabase()
        await graph_db.initialize()
        
        # Share the driver with vector store
        vector_store = Neo4jVectorStore(graph_driver=graph_db.driver)
        await vector_store.initialize()
        
        return graph_db, vector_store
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
