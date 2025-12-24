"""
Memory Store for Persona v2.

Unified storage for all memory types (episode, psyche, goal).
Handles temporal linking, retrieval, and graph operations.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import UUID

from persona.core.interfaces import GraphDatabase, VectorStore
from persona.models.memory import Memory, MemoryLink, MemoryQueryResponse
from server.logging_config import get_logger

logger = get_logger(__name__)


class MemoryStore:
    """
    Unified store for all memory types.
    
    All memories are stored as nodes in the graph.
    Links between memories are edges.
    """
    
    def __init__(self, graph_db: GraphDatabase, vector_store: Optional[VectorStore] = None):
        self.graph_db = graph_db
        self.vector_store = vector_store

    def _memory_to_node_data(self, memory: Memory) -> Dict[str, Any]:
        # Set day_id if not provided
        if not memory.day_id:
            memory.day_id = memory.timestamp.strftime("%Y-%m-%d")

        node_data = memory.model_dump(exclude={'properties'})
        node_data["name"] = str(memory.id)

        for k, v in node_data.items():
            if isinstance(v, UUID):
                node_data[k] = str(v)
            elif isinstance(v, datetime):
                node_data[k] = v.isoformat()

        if hasattr(memory, 'properties') and memory.properties:
            node_data.update(memory.properties)

        for field in ['timestamp', 'created_at', 'due_date', 'last_accessed']:
            if field in node_data and isinstance(node_data[field], datetime):
                node_data[field] = node_data[field].isoformat()

        return node_data
    
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
        # Create the memory node with FLAT properties (not nested JSON)
        # This is backend-agnostic: each field becomes a native property
        node_data = self._memory_to_node_data(memory)
        
        await self.graph_db.create_nodes([node_data], memory.user_id)

        if self.vector_store and memory.embedding:
            try:
                await self.vector_store.add_embedding(
                    node_name=node_data["name"],
                    embedding=memory.embedding,
                    user_id=memory.user_id
                )
            except Exception as e:
                logger.warning(
                    f"Failed to persist embedding for memory {node_data['name']}: {e}"
                )
        
        # Create links if provided
        if links:
            for link in links:
                await self.create_link(link, memory.user_id)
        
        logger.info(f"Created {memory.type} memory '{memory.title}' for user {memory.user_id}")
        return memory

    async def create_many(
        self,
        memories: List[Memory],
        links: Optional[List[MemoryLink]],
        user_id: str
    ) -> None:
        if not memories:
            return

        node_data = [self._memory_to_node_data(memory) for memory in memories]
        await self.graph_db.create_nodes(node_data, user_id)

        if self.vector_store:
            rows = [
                {"node_name": str(memory.id), "embedding": memory.embedding}
                for memory in memories
                if memory.embedding
            ]
            if rows:
                try:
                    await self.vector_store.add_embeddings(rows, user_id)
                except Exception as e:
                    logger.warning(
                        f"Failed to persist embeddings for batch: {e}"
                    )

        if links:
            relationships = [
                {
                    "source": str(link.source_id),
                    "target": str(link.target_id),
                    "relation": link.relation
                }
                for link in links
            ]
            await self.graph_db.create_relationships(relationships, user_id)
    
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
        """Convert a graph node to the correct polymorphic Memory model."""
        from pydantic import TypeAdapter, ValidationError
        import json
        
        props = node.get('properties', {})
        # Handle flat properties (new format) vs nested properties (old format)
        if not props and 'title' in node:
            props = node
        
        # Deserialization: Parse JSON strings if any
        # This handles nested dicts/lists that were JSON-serialized
        processed_props = {}
        for k, v in props.items():
            if isinstance(v, str) and (v.startswith('{') or v.startswith('[')):
                try:
                    processed_props[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    processed_props[k] = v
            else:
                processed_props[k] = v
        
        # Ensure 'id' is a valid UUID string
        if 'id' not in processed_props and 'name' in node:
            processed_props['id'] = node['name']
        
        # Ensure user_id is set
        processed_props['user_id'] = user_id
        
        try:
            # Pydantic's TypeAdapter handles the discriminated union automatically!
            # It looks at the 'type' field and instantiates Episode/Psyche/Goal accordingly.
            return TypeAdapter(Memory).validate_python(processed_props)
        except ValidationError as e:
            logger.error(f"Failed to reconstruct memory {processed_props.get('id')}: {e}")
            # Fallback to generic Memory-like structure if validation fails
            # This prevents crashing on bad data
            from persona.models.memory import EpisodeMemory
            return EpisodeMemory(
                **{k: v for k, v in processed_props.items() if k in EpisodeMemory.model_fields},
                type='episode', # Force valid type
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
