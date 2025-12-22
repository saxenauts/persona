"""
Memory Store for Persona v2.

Unified storage for all memory types (episode, psyche, goal).
Handles temporal linking, retrieval, and graph operations.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import UUID

from persona.core.interfaces import GraphDatabase
from persona.models.memory import Memory, MemoryLink, MemoryQueryResponse
from server.logging_config import get_logger

logger = get_logger(__name__)


class MemoryStore:
    """
    Unified store for all memory types.
    
    All memories are stored as nodes in the graph.
    Links between memories are edges.
    """
    
    def __init__(self, graph_db: GraphDatabase):
        self.graph_db = graph_db
    
    async def create(
        self, 
        memory: Memory,
        links: Optional[List[MemoryLink]] = None
    ) -> Memory:
        """
        Create a memory and optionally link it to other memories.
        
        Args:
            memory: The Memory to create
            links: Optional list of links to create
            
        Returns:
            The created Memory
        """
        # Set day_id if not provided
        if not memory.day_id:
            memory.day_id = memory.timestamp.strftime("%Y-%m-%d")
        
        # Create the memory node with FLAT properties (not nested JSON)
        # This is backend-agnostic: each field becomes a native property
        node_data = {
            "name": str(memory.id),
            "type": memory.type,
            # All fields as flat properties
            "id": str(memory.id),
            "title": memory.title,
            "content": memory.content,
            "timestamp": memory.timestamp.isoformat(),
            "created_at": memory.created_at.isoformat(),
            "day_id": memory.day_id,
            "session_id": memory.session_id,
            "source_type": memory.source_type,
            "source_ref": memory.source_ref,
            "access_count": memory.access_count,
            # Optional fields
            "status": memory.status,
            "due_date": memory.due_date.isoformat() if memory.due_date else None,
            "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
            # Embedding as list (for vector backends)
            "embedding": memory.embedding,
            # Extra properties dict (rarely used)
            "extra": memory.properties if memory.properties else None,
        }
        
        await self.graph_db.create_nodes([node_data], memory.user_id)
        
        # Create links if provided
        if links:
            for link in links:
                await self.create_link(link, memory.user_id)
        
        logger.info(f"Created {memory.type} memory '{memory.title}' for user {memory.user_id}")
        return memory
    
    async def create_link(self, link: MemoryLink, user_id: str) -> None:
        """Create a link between two memories."""
        relationship = {
            "source": str(link.source_id),
            "target": str(link.target_id),
            "relation": link.relation,
            **link.properties
        }
        await self.graph_db.create_relationships([relationship], user_id)
    
    async def get(self, memory_id: UUID, user_id: str) -> Optional[Memory]:
        """Retrieve a single memory by ID."""
        node_data = await self.graph_db.get_node(str(memory_id), user_id)
        
        if not node_data:
            return None
        
        return self._node_to_memory(node_data, user_id)
    
    async def get_by_type(
        self, 
        memory_type: str, 
        user_id: str,
        limit: int = 50
    ) -> List[Memory]:
        """Get all memories of a specific type."""
        all_nodes = await self.graph_db.get_all_nodes(user_id)
        
        memories = [
            self._node_to_memory(n, user_id)
            for n in all_nodes
            if n.get('type') == memory_type
        ]
        
        # Sort by timestamp descending (most recent first)
        memories.sort(key=lambda m: m.timestamp, reverse=True)
        return memories[:limit]
    
    async def get_by_day(self, day_id: str, user_id: str) -> List[Memory]:
        """Get all memories for a specific day."""
        all_nodes = await self.graph_db.get_all_nodes(user_id)
        
        memories = []
        for node in all_nodes:
            props = node.get('properties', {})
            if props.get('day_id') == day_id:
                memories.append(self._node_to_memory(node, user_id))
        
        memories.sort(key=lambda m: m.timestamp)
        return memories
    
    async def get_recent(
        self, 
        user_id: str, 
        memory_type: Optional[str] = None,
        limit: int = 20
    ) -> List[Memory]:
        """Get recent memories, optionally filtered by type."""
        all_nodes = await self.graph_db.get_all_nodes(user_id)
        
        memories = []
        for node in all_nodes:
            if memory_type and node.get('type') != memory_type:
                continue
            memories.append(self._node_to_memory(node, user_id))
        
        memories.sort(key=lambda m: m.timestamp, reverse=True)
        return memories[:limit]
    
    async def get_most_recent_episode(self, user_id: str) -> Optional[Memory]:
        """Get the most recent episode (for temporal chain linking)."""
        episodes = await self.get_by_type("episode", user_id, limit=1)
        return episodes[0] if episodes else None
    
    async def link_temporal_chain(
        self, 
        new_memory: Memory, 
        previous_memory: Memory
    ) -> None:
        """Create PREVIOUS/NEXT links between episodes."""
        # New → Previous
        await self.create_link(
            MemoryLink(
                source_id=new_memory.id,
                target_id=previous_memory.id,
                relation="PREVIOUS"
            ),
            new_memory.user_id
        )
        # Previous → New  
        await self.create_link(
            MemoryLink(
                source_id=previous_memory.id,
                target_id=new_memory.id,
                relation="NEXT"
            ),
            previous_memory.user_id
        )
    
    def _node_to_memory(self, node: Dict[str, Any], user_id: str) -> Memory:
        """Convert a graph node to a Memory model."""
        props = node.get('properties', {})
        
        # Handle flat properties (new format) vs nested properties (old format)
        # New format: properties are at node level
        # Old format: properties are in a nested 'properties' dict
        if not props and 'title' in node:
            # New flat format
            props = node
        
        return Memory(
            id=UUID(props.get('id', node.get('name'))),
            type=props.get('type', node.get('type', 'episode')),
            title=props.get('title', ''),
            content=props.get('content', ''),
            timestamp=datetime.fromisoformat(props['timestamp']) if props.get('timestamp') else datetime.utcnow(),
            created_at=datetime.fromisoformat(props['created_at']) if props.get('created_at') else datetime.utcnow(),
            day_id=props.get('day_id'),
            status=props.get('status'),
            due_date=datetime.fromisoformat(props['due_date']) if props.get('due_date') else None,
            session_id=props.get('session_id'),
            source_type=props.get('source_type', 'conversation'),
            source_ref=props.get('source_ref'),
            access_count=props.get('access_count', 0),
            last_accessed=datetime.fromisoformat(props['last_accessed']) if props.get('last_accessed') else None,
            user_id=user_id
        )
    
    # ========== Search Methods ==========
    
    async def search_text(
        self,
        user_id: str,
        query: str,
        types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Memory]:
        """
        Keyword-based text search on title and content.
        
        Args:
            user_id: User ID
            query: Search query string
            types: Filter by memory types (episode, psyche, goal)
            limit: Maximum results
        """
        all_nodes = await self.graph_db.get_all_nodes(user_id)
        query_lower = query.lower()
        
        matches = []
        for node in all_nodes:
            if types and node.get('type') not in types:
                continue
            
            # Check title and content for query match
            props = node.get('properties', node)
            title = str(props.get('title', '')).lower()
            content = str(props.get('content', '')).lower()
            
            if query_lower in title or query_lower in content:
                matches.append(self._node_to_memory(node, user_id))
        
        return matches[:limit]
    
    async def search_vector(
        self,
        user_id: str,
        query: str,
        types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Memory]:
        """
        Semantic similarity search using embeddings.
        
        Args:
            user_id: User ID
            query: Search query
            types: Filter by memory types
            limit: Maximum results
        """
        from persona.core.graph_ops import GraphOps
        
        # Use GraphOps for proper vector search
        async with GraphOps() as graph_ops:
            search_results = await graph_ops.text_similarity_search(
                query=query,
                user_id=user_id,
                limit=limit * 2  # Get more for filtering
            )
        
        results = search_results.get('results', [])
        
        # Convert to memories and filter by type
        memories = []
        for result in results:
            node = await self.graph_db.get_node(result['nodeName'], user_id)
            if node:
                if types and node.get('type') not in types:
                    continue
                memories.append(self._node_to_memory(node, user_id))
                if len(memories) >= limit:
                    break
        
        return memories
    
    async def get_connected(
        self,
        memory_id: UUID,
        user_id: str,
        relation: Optional[str] = None
    ) -> List[Memory]:
        """
        Get memories connected to this one via relationships.
        
        Args:
            memory_id: Source memory ID
            user_id: User ID
            relation: Filter by relationship type (DERIVED_FROM, NEXT, etc.)
        """
        relationships = await self.graph_db.get_node_relationships(str(memory_id), user_id)
        
        connected = []
        for rel in relationships:
            # Get the connected node
            target_name = rel.get('target') if rel.get('source') == str(memory_id) else rel.get('source')
            if relation and rel.get('relation') != relation:
                continue
            
            node = await self.graph_db.get_node(target_name, user_id)
            if node:
                connected.append(self._node_to_memory(node, user_id))
        
        return connected
    
    async def get_goal_hierarchy(
        self,
        user_id: str,
        root_id: Optional[UUID] = None
    ) -> List[Memory]:
        """
        Get goals and their subtasks in hierarchy.
        
        If root_id is provided, return that goal and its children.
        Otherwise return all goals.
        """
        goals = await self.get_by_type("goal", user_id, limit=100)
        
        if root_id:
            # Get connected goals via PARENT_OF
            connected = await self.get_connected(root_id, user_id, relation="PARENT_OF")
            return [g for g in goals if g.id == root_id] + connected
        
        return goals
    
    # ========== Mutation Methods ==========
    
    async def update(
        self,
        memory_id: UUID,
        user_id: str,
        updates: Dict[str, Any]
    ) -> Optional[Memory]:
        """
        Update mutable fields of a memory.
        
        Args:
            memory_id: Memory ID to update
            user_id: User ID
            updates: Dict of field:value pairs to update
        
        Allowed fields: title, content, status, properties
        """
        # Get existing memory
        existing = await self.get(memory_id, user_id)
        if not existing:
            logger.warning(f"Memory {memory_id} not found for update")
            return None
        
        # Build update node
        node_data = {
            "name": str(memory_id),
            "type": existing.type,
        }
        
        # Add allowed updates
        allowed_fields = {'title', 'content', 'status', 'due_date'}
        for field, value in updates.items():
            if field in allowed_fields:
                node_data[field] = value
        
        # Apply update
        await self.graph_db.create_nodes([node_data], user_id)
        
        logger.info(f"Updated memory {memory_id}: {list(updates.keys())}")
        return await self.get(memory_id, user_id)

