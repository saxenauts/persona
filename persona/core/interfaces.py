"""
Abstract interfaces for database backends.

These interfaces allow Persona to work with different graph and vector databases
without coupling to a specific implementation (e.g., Neo4j, FalkorDB, Pinecone).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class GraphDatabase(ABC):
    """Abstract interface for graph database operations."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize connection and ensure schema is ready."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the database connection."""
        pass
    
    # Node Operations
    @abstractmethod
    async def create_nodes(self, nodes: List[Dict[str, Any]], user_id: str) -> None:
        """Create or merge nodes in the graph.
        
        Args:
            nodes: List of node dicts with keys: name, type, properties
            user_id: User ID to associate nodes with
        """
        pass
    
    @abstractmethod
    async def get_node(self, node_name: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by name.
        
        Returns:
            Dict with name, type, properties or None if not found
        """
        pass
    
    @abstractmethod
    async def get_all_nodes(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all nodes for a user."""
        pass
    
    @abstractmethod
    async def check_node_exists(self, node_name: str, node_type: str, user_id: str) -> bool:
        """Check if a node exists."""
        pass
    
    # Relationship Operations
    @abstractmethod
    async def create_relationships(self, relationships: List[Dict[str, Any]], user_id: str) -> None:
        """Create relationships between nodes.
        
        Args:
            relationships: List of dicts with keys: source, target, relation
            user_id: User ID for filtering
        """
        pass
    
    @abstractmethod
    async def get_node_relationships(self, node_name: str, user_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for a node."""
        pass
    
    @abstractmethod
    async def get_all_relationships(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for a user."""
        pass
    
    # User Management
    @abstractmethod
    async def create_user(self, user_id: str) -> None:
        """Create a user node."""
        pass
    
    @abstractmethod
    async def user_exists(self, user_id: str) -> bool:
        """Check if a user exists."""
        pass
    
    @abstractmethod
    async def delete_user(self, user_id: str) -> None:
        """Delete a user and all their associated data."""
        pass
    
    @abstractmethod
    async def clean_graph(self) -> None:
        """Delete all data (for testing)."""
        pass


class VectorStore(ABC):
    """Abstract interface for vector similarity operations."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize and ensure index exists."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the connection."""
        pass
    
    @abstractmethod
    async def add_embedding(self, node_name: str, embedding: List[float], user_id: str) -> None:
        """Add or update an embedding for a node.
        
        Args:
            node_name: Name of the node
            embedding: Vector embedding
            user_id: User ID for filtering
        """
        pass
    
    @abstractmethod
    async def search_similar(
        self, 
        embedding: List[float], 
        user_id: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            embedding: Query embedding
            user_id: User ID for filtering
            limit: Maximum number of results
            
        Returns:
            List of dicts with keys: node_name, score
        """
        pass
