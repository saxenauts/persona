from persona.core.interfaces import GraphDatabase, VectorStore
from persona.llm.embeddings import generate_embeddings, generate_embeddings_async
from persona.models.schema import (
    NodeModel, 
    RelationshipModel, 
    GraphUpdateModel, 
    NodesAndRelationshipsResponse, 
    CommunityStructure, 
    Subgraph, 
    Node, 
    Relationship, 
    GraphSchema
)
from typing import List, Dict, Any, Optional
import asyncio
import json
from persona.llm.llm_graph import detect_communities
from collections import defaultdict
from server.logging_config import get_logger

logger = get_logger(__name__)


class GraphOps:
    """
    Graph operations layer that abstracts database backends.
    Uses GraphDatabase for graph operations and VectorStore for similarity search.
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
    # Node Operations
    # -------------------------------------------------------------------------

    async def add_nodes(self, nodes: List[NodeModel], user_id: str):
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot add nodes.")
            return

        node_dicts = [
            {
                "name": node.name,
                "type": node.type or "",
                "properties": node.properties or {}
            } 
            for node in nodes
        ]
        
        await self.graph_db.create_nodes(node_dicts, user_id)
        await self.add_nodes_batch_embeddings(nodes, user_id)

    async def add_nodes_batch_embeddings(self, nodes: List[NodeModel], user_id: str):
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot add embeddings.")
            return

        embed_texts = [self._embedding_text_for_node(node) for node in nodes]
        embeddings = await generate_embeddings_async(embed_texts)
        
        for node_obj, embedding in zip(nodes, embeddings):
            if embedding:
                await self.vector_store.add_embedding(node_obj.name, embedding, user_id)
            else:
                logger.error(f"Failed to generate embedding for node: {node_obj.name}")

    def _embedding_text_for_node(self, node: NodeModel) -> str:
        """Build text for embeddings from name + type + key properties."""
        parts = [node.name]
        
        if getattr(node, 'type', None):
            parts.append(f"type:{node.type}")
        
        props = getattr(node, 'properties', {}) or {}
        interesting_keys = [
            'date', 'timestamp', 'entity', 'location', 
            'quantity', 'unit', 'count', 'source', 
            'category', 'tags', 'polarity'
        ]
        
        for k in interesting_keys:
            if k in props and props[k] not in (None, ""):
                parts.append(f"{k}:{props[k]}")
        
        return " | ".join(map(str, parts))

    async def get_node_data(self, node_name: str, user_id: str) -> NodeModel:
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot get node data.")
            return NodeModel(name=node_name)

        node_data = await self.graph_db.get_node(node_name, user_id)
        
        if node_data:
            # Neo4j returns flat dict with all properties at top level.
            # Extract everything except reserved/system fields as properties.
            reserved_keys = {"name", "UserId", "embedding", "elementId"}
            properties = {k: v for k, v in node_data.items() if k not in reserved_keys}
            
            return NodeModel(
                name=node_data["name"],
                type=node_data.get("type"),
                properties=properties  # Now includes content, title, etc.
            )
        
        return NodeModel(name=node_name)

    async def get_all_nodes(self, user_id: str) -> List[NodeModel]:
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot get all nodes.")
            return []

        nodes = await self.graph_db.get_all_nodes(user_id)
        reserved_keys = {"name", "UserId", "embedding", "elementId"}
        
        return [
            NodeModel(
                name=node['name'],
                type=node.get('type'),
                properties={k: v for k, v in node.items() if k not in reserved_keys}
            ) 
            for node in nodes
        ]

    # -------------------------------------------------------------------------
    # Relationship Operations
    # -------------------------------------------------------------------------

    async def add_relationships(self, relationships: List[RelationshipModel], user_id: str):
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot add relationships.")
            return

        relationship_dicts = [rel.dict() for rel in relationships]
        await self.graph_db.create_relationships(relationship_dicts, user_id)

    async def get_node_relationships(self, node_name: str, user_id: str) -> List[RelationshipModel]:
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot get node relationships.")
            return []

        relationships = await self.graph_db.get_node_relationships(node_name, user_id)
        
        return [
            RelationshipModel(
                source=rel["source"], 
                target=rel["target"], 
                relation=rel["relation"]
            ) 
            for rel in relationships
        ]

    async def get_all_relationships(self, user_id: str) -> List[RelationshipModel]:
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot get all relationships.")
            return []

        relationships = await self.graph_db.get_all_relationships(user_id)
        
        return [
            RelationshipModel(
                source=rel['source'], 
                target=rel['target'], 
                relation=rel['relation']
            ) 
            for rel in relationships
        ]

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

    # -------------------------------------------------------------------------
    # Graph Update
    # -------------------------------------------------------------------------

    async def update_graph(self, graph_update: NodesAndRelationshipsResponse, user_id: str):
        """Update the graph with new nodes and relationships."""
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot update graph.")
            return

        if graph_update.nodes:
            await self.add_nodes(graph_update.nodes, user_id)
        
        if graph_update.relationships:
            await self.add_relationships(graph_update.relationships, user_id)
        
        if not graph_update.nodes and not graph_update.relationships:
            logger.debug("No nodes or relationships to update.")

    # -------------------------------------------------------------------------
    # Subgraph & Community Detection
    # -------------------------------------------------------------------------

    async def get_ranked_subgraphs(self, user_id: str) -> List[Subgraph]:
        """Get all subgraphs in the graph, ranked by size and influence."""
        subgraphs = []
        visited_nodes = set()

        all_nodes = await self.graph_db.get_all_nodes(user_id)

        for node in all_nodes:
            node_name = node['name']
            
            if node_name not in visited_nodes:
                relationships = await self.graph_db.get_node_relationships(node_name, user_id)

                connected_nodes = {node_name}
                for rel in relationships:
                    connected_nodes.add(rel['source'])
                    connected_nodes.add(rel['target'])

                visited_nodes.update(connected_nodes)
                central_nodes = await self._get_central_nodes(list(connected_nodes), relationships)

                subgraphs.append(Subgraph(
                    id=len(subgraphs),
                    nodes=list(connected_nodes),
                    relationships=relationships,
                    size=len(connected_nodes),
                    central_nodes=central_nodes
                ))

        subgraphs.sort(key=lambda sg: sg.size, reverse=True)
        return subgraphs

    async def _get_central_nodes(self, nodes: List[str], relationships: List[Dict]) -> List[str]:
        """Calculate central nodes based on degree centrality."""
        if not nodes:
            return []
            
        degree_count = defaultdict(int)
        for rel in relationships:
            degree_count[rel['source']] += 1
            degree_count[rel['target']] += 1
        
        if degree_count:
            sorted_nodes = sorted(degree_count.items(), key=lambda x: x[1], reverse=True)
            return [sorted_nodes[0][0]]
        
        return [nodes[0]]

    async def format_subgraphs_for_llm(self, subgraphs: List[Subgraph]) -> str:
        """Format subgraphs into a string for LLM input."""
        formatted = "# Graph Structure Analysis\n\n"
        
        for subgraph in subgraphs:
            formatted += f"\n## Subgraph {subgraph.id} (Size: {subgraph.size})\n"
            formatted += f"Central Nodes: {', '.join(subgraph.central_nodes)}\n"
            formatted += "\nNodes:\n"
            
            for node in subgraph.nodes:
                formatted += f"- {node}\n"
            
            formatted += "\nRelationships:\n"
            for rel in subgraph.relationships:
                formatted += f"- {rel['source']} {rel['relation']} {rel['target']}\n"
        
        return formatted

    async def make_communities(
        self, 
        user_id: str, 
        community_structure: CommunityStructure, 
        subgraphs: List[Subgraph]
    ) -> None:
        """Create community structure in the graph."""
        for header in community_structure.communityHeaders:
            # Create header node
            await self.graph_db.create_nodes(
                [{'name': header.header, 'type': 'CommunityHeader'}], 
                user_id
            )
            
            for subheader in header.subheaders:
                # Create subheader node
                await self.graph_db.create_nodes(
                    [{'name': subheader.subheader, 'type': 'CommunitySubheader'}], 
                    user_id
                )
                
                # Connect header to subheader
                await self.graph_db.create_relationships(
                    [{
                        'source': header.header, 
                        'target': subheader.subheader, 
                        'relation': 'HAS_SUBHEADER'
                    }], 
                    user_id
                )
                
                # Connect representative nodes from subgraphs
                for subgraph_id in subheader.subgraph_ids:
                    if subgraph_id < len(subgraphs):
                        subgraph = subgraphs[subgraph_id]
                        for central_node in subgraph.central_nodes:
                            await self.graph_db.create_relationships(
                                [{
                                    'source': central_node, 
                                    'target': subheader.subheader, 
                                    'relation': 'BELONGS_TO'
                                }], 
                                user_id
                            )

    async def community_detection(self, user_id) -> None:
        """Main orchestrator for community detection process."""
        subgraphs = await self.get_ranked_subgraphs(user_id)
        subgraphs_text = await self.format_subgraphs_for_llm(subgraphs)
        community_structure = await detect_communities(subgraphs_text)
        await self.make_communities(user_id, community_structure, subgraphs)



# ============================================================================
# LEGACY: GraphContextRetriever has been removed.
# Use persona.core.retrieval.Retriever instead.
# ============================================================================

