import json
from typing import List, Tuple, Dict, Any
from persona.llm.prompts import GET_NODES, GET_RELATIONSHIPS, GENERATE_COMMUNITIES, GENERATE_STRUCTURED_INSIGHTS
from persona.models.schema import EntityExtractionResponse, NodesAndRelationshipsResponse, CommunityStructure, AskResponse, AskRequest, create_dynamic_schema
from pydantic import BaseModel, Field
from persona.utils.instructions_reader import INSTRUCTIONS
from server.logging_config import get_logger
from .client_factory import get_chat_client
from .providers.base import ChatMessage

logger = get_logger(__name__)

class Node(BaseModel):
    name: str = Field(..., description="The node content - can be a simple label (e.g., 'Techno Music') or a narrative fragment (e.g., 'Deeply moved by classical music in empty spaces')")
    type: str = Field(..., description="The type/category of the node (e.g., 'Identity', 'Belief', 'Preference', 'Goal', 'Event', 'Relationship', etc.)")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Optional schemaless properties for the node (e.g., date, count, source, location)")

class Relationship(BaseModel):
    source: str
    relation: str
    target: str

class RelationshipWithID(BaseModel):
    source_id: str = Field(..., description="Temporary ID of the source node (e.g., 'Node1')")
    relation: str = Field(..., description="Type of relationship")
    target_id: str = Field(..., description="Temporary ID of the target node (e.g., 'Node2')")

class GraphResponse(BaseModel):
    nodes: List[Node] = Field(..., description="List of nodes in the graph")
    relationships: List[Relationship] = Field(default_factory=list, description="List of relationships between nodes")

async def get_nodes(text: str, graph_context: str) -> List[Node]:
    """
    Extract nodes from provided text using the configured LLM service.
    Returns nodes that can be either simple labels or narrative fragments.
    """
    try:
        combined_instructions = f"App Objective: {INSTRUCTIONS}\n\nExisting Graph Context: {graph_context}\n\nNode Extraction Task: {GET_NODES}"
        
        messages = [
            ChatMessage(role="system", content=combined_instructions),
            ChatMessage(role="user", content=text)
        ]
        
        client = get_chat_client()
        response = await client.chat(
            messages=messages,
            temperature=0.5,
            response_format={"type": "json_object"}
        )
        
        # Parse JSON response
        json_data = json.loads(response.content)

        # Validate and convert to Node objects
        if "nodes" in json_data:
            nodes = []
            for node_data in json_data["nodes"]:
                # Ensure properties defaults to {} if missing
                if "properties" not in node_data or node_data["properties"] is None:
                    node_data["properties"] = {}
                try:
                    nodes.append(Node(**node_data))
                except Exception as e:
                    logger.warning(f"Skipping malformed node from LLM response: {e}; data={node_data}")
            return nodes
        else:
            logger.warning("No 'nodes' key found in LLM response")
            return []
            
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON in get_nodes: {e}")
        return []
    except Exception as e:
        logger.error(f"Error while extracting nodes: {e}")
        return []

async def get_relationships(nodes: List[Node], graph_context: str) -> Tuple[List[Relationship], Dict[str, str]]:
    """
    Generate relationships based on the list of nodes and existing graph context using the configured LLM service.
    Returns a tuple of (relationships, id_mapping) where id_mapping maps temporary IDs to node names.
    """
    if not nodes:
        return [], {}
    
    # Create temporary ID mapping
    id_mapping = {}
    nodes_with_ids = []
    
    for i, node in enumerate(nodes):
        temp_id = f"Node{i+1}"
        id_mapping[temp_id] = node.name
        nodes_with_ids.append(f'{temp_id}: "{node.name}"')
    
    # Format nodes for the prompt with temporary IDs
    nodes_str = '\n'.join(nodes_with_ids)
    combined_instructions = f"App Objective: {INSTRUCTIONS}\n\nRelationships Generation Task: {GET_RELATIONSHIPS}"
    
    try:
        messages = [
            ChatMessage(role="system", content=combined_instructions),
            ChatMessage(role="user", content=f"Nodes:\n{nodes_str}\n\nExisting Graph Context:\n{graph_context}")
        ]
        
        client = get_chat_client()
        response = await client.chat(
            messages=messages,
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        # Parse JSON response
        json_data = json.loads(response.content)
        
        # Validate and convert to RelationshipWithID objects
        relationships_with_ids = []
        if "relationships" in json_data:
            for rel_data in json_data["relationships"]:
                relationships_with_ids.append(RelationshipWithID(**rel_data))
        
        # Convert RelationshipWithID back to Relationship using the mapping
        converted_relationships = []
        for rel_with_id in relationships_with_ids:
            source_name = id_mapping.get(rel_with_id.source_id)
            target_name = id_mapping.get(rel_with_id.target_id)
            
            if source_name and target_name:
                converted_relationships.append(Relationship(
                    source=source_name,
                    relation=rel_with_id.relation,
                    target=target_name
                ))
            else:
                logger.warning(f"Invalid relationship with IDs: {rel_with_id.source_id} -> {rel_with_id.target_id}")
        
        return converted_relationships, id_mapping
        
    except Exception as e:
        logger.error(f"Error while generating relationships: {e}")
        return [], {}

async def generate_response_with_context(query: str, context: str) -> str:
    """Generate a response based on query and context using the configured LLM service."""
    prompt = f"""
    Given the following context from a knowledge graph and a query, provide a detailed answer:

    Context:
    {context}

    Query: {query}

    Please provide a comprehensive answer based on the given context:
    """

    try:
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant that answers queries about a user based on the provided context from their graph."),
            ChatMessage(role="user", content=prompt)
        ]
        
        client = get_chat_client()
        response = await client.chat(messages=messages, temperature=0.7)
        
        return response.content
        
    except Exception as e:
        logger.error(f"Error generating response with context: {e}")
        return "I apologize, but I encountered an error while processing your request."

async def detect_communities(subgraphs_text: str) -> CommunityStructure:
    """
    Use LLM to detect communities in the graph and organize them into headers/subheaders
    """
    try:
        messages = [
            ChatMessage(role="system", content=GENERATE_COMMUNITIES),
            ChatMessage(role="user", content=subgraphs_text)
        ]
        
        client = get_chat_client()
        response = await client.chat(
            messages=messages,
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        # Parse JSON response
        json_data = json.loads(response.content)
        
        # Validate and convert to CommunityStructure
        community_structure = CommunityStructure(**json_data)
        logger.debug(f"Community Detection Response: {community_structure}")
        return community_structure
        
    except Exception as e:
        logger.error(f"Error in community detection: {str(e)}")
        return CommunityStructure(communityHeaders=[])

async def generate_structured_insights(ask_request: AskRequest, context: str) -> Dict[str, Any]:
    """
    Generate structured insights based on the provided context and query using the configured LLM service
    """
    prompt = f"""
    Based on this context from the knowledge graph:
    {context}
    
    Answer this query about the user: {ask_request.query}
    
    Provide your response following the example structure:
    {json.dumps(ask_request.output_schema, indent=2)}
    """

    logger.debug(f"Structured insights prompt: {prompt}")

    try:
        messages = [
            ChatMessage(role="system", content=GENERATE_STRUCTURED_INSIGHTS),
            ChatMessage(role="user", content=prompt)
        ]
        
        client = get_chat_client()
        response = await client.chat(
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.content)
        
    except Exception as e:
        logger.error(f"Error in generate_structured_insights: {e}")
        return {k: [] if isinstance(v, list) else {} for k, v in ask_request.output_schema.items()}
