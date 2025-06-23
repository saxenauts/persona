"""
Schema and Pydantic Models for the Graph Library Ops."
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from instructor import OpenAISchema

class UnstructuredData(BaseModel):
    title: str
    content: str
    metadata: Optional[Dict[str, str]] = {}

class Node(OpenAISchema):
    name: str = Field(..., description="The node content - can be a simple label (e.g., 'Techno Music') or a narrative fragment (e.g., 'Deeply moved by classical music in empty spaces')")

class Relationship(OpenAISchema):
    source: str
    target: str
    relation: str

class NodeModel(BaseModel):
    name: str = Field(..., description="The node content - can be a simple label or narrative fragment")
    perspective: Optional[str] = Field(None, description="Additional perspective or context for the node")
    properties: Optional[Dict[str, str]] = Field(default_factory=dict)
    embedding: Optional[List[float]] = Field(None, description="Embedding vector for the node, if applicable")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Finds peace in early morning solitude",
                "perspective": "This reflects a desire for quiet contemplation",
                "properties": {},
                "embedding": [0.1, 0.2, 0.3]
            }
        }


class RelationshipModel(BaseModel):
    source: str
    target: str
    relation: str

    class Config:
        json_schema_extra = {
            "example": {
                "source": "Quantum Computing",
                "target": "AI",
                "relation": "RELATED_TO"
            }
        }



class GraphUpdateModel(BaseModel):
    nodes: List[NodeModel]
    relationships: List[RelationshipModel]

    class Config:
        json_schema_extra = {
            "example": {
                "nodes": [
                    {"name": "Node1", "properties": {"frequency": 1}, "embedding": [0.1, 0.2, 0.3]},
                    {"name": "Node2", "properties": {"frequency": 1}, "embedding": [0.1, 0.2, 0.3]}
                ],
                "relationships": [
                    {"source": "Node1", "target": "Node2", "relation": "CONNECTED_TO"}
                ]
            }
        }


class EntityExtractionResponse(BaseModel):
    entities: List[str] = Field(..., example=["Blockchain", "Quantum Computing", "Indie Games", "Sustainable Farming", "Virtual Reality"])


class NodesAndRelationshipsResponse(BaseModel):
    nodes: List[NodeModel] = Field(..., example=[
        {"name": "Finds peace in early morning solitude"},
        {"name": "Techno Music"},
        {"name": "Values deep conversations"},
        {"name": "Real Madrid"},
        {"name": "Anxious about future of AI"}
    ])
    relationships: List[RelationshipModel] = Field(..., example=[
        {"source": "Finds peace in early morning solitude", "relation": "CONTRASTS_WITH", "target": "Anxious about future of AI"},
        {"source": "Techno Music", "relation": "ENHANCES", "target": "Finds peace in early morning solitude"},
        {"source": "Values deep conversations", "relation": "REFLECTS", "target": "Anxious about future of AI"}
    ])

class UserCreate(BaseModel):
    user_id: str

class RAGQuery(BaseModel):
    query: str

class RAGResponse(BaseModel):
    answer: str

class Subgraph(OpenAISchema):
    id: int
    nodes: List[str]
    relationships: List[Dict[str, str]]
    size: int
    central_nodes: List[str]  # nodes with highest degree/influence

class CommunitySubheader(OpenAISchema):
    subheader: str
    subgraph_ids: List[int]

class CommunityHeader(OpenAISchema):
    header: str
    subheaders: List[CommunitySubheader]

class CommunityStructure(OpenAISchema):
    communityHeaders: List[CommunityHeader]

# BYOA - Learn Anything Ask Anything Personalize Anything

class GraphSchema(BaseModel):
    name: str
    description: str
    attributes: List[str]
    relationships: List[str]
    is_seed: bool = False
    created_at: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Gaming Preferences",
                "description": "Schema for tracking user gaming preferences",
                "attributes": ["CURRENT_GAME", "FAVORITE_GENRE", "PLAYTIME"],
                "relationships": ["PLAYS", "PREFERS", "COMPLETED"],
                "is_seed": False
            }
        }

class LearnRequest(BaseModel):
    user_id: str
    graph_schema: GraphSchema
    description: str

class LearnResponse(BaseModel):
    status: str
    schema_id: str
    details: str

class AskRequest(BaseModel):
    query: str
    output_schema: Dict[str, Any] = Field(..., description="Expected output structure with example values")

class AskResponse(BaseModel):
    result: Dict[str, Any]

class AskResponseInstructor(OpenAISchema):
    result: Dict[str, Any]


def create_dynamic_schema(output_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a JSON schema based on the provided output schema for OpenAI structured output
    """
    def create_property_schema(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            properties = {
                k: create_property_schema(v) for k, v in value.items()
            }
            return {
                "type": "object",
                "properties": properties,
                "required": list(value.keys())
            }
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                return {
                    "type": "array",
                    "items": create_property_schema(value[0])
                }
            return {
                "type": "array",
                "items": {"type": "string"}
            }
        return {"type": "string"}

    schema = {
        "type": "object",
        "properties": {
            k: create_property_schema(v) for k, v in output_schema.items()
        },
        "required": list(output_schema.keys())
    }
    
    return schema

class CustomNodeData(BaseModel):
    name: str
    perspective: Optional[str] = None
    properties: Optional[Dict[str, str]] = Field(default_factory=dict)
    embedding: Optional[List[float]] = Field(None, description="Embedding vector for the node, if applicable")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Quantum Computing",
                "properties": {
                    "current_context": "Research",
                    "frequency": 10
                },
                "embedding": [0.1, 0.2, 0.3]
            }
        }

class CustomRelationshipData(BaseModel):
    source: str
    target: str
    relation_type: str
    data: Dict[str, Any] = Field(default_factory=dict)

class CustomGraphUpdate(BaseModel):
    nodes: List[CustomNodeData]
    relationships: List[CustomRelationshipData]

    class Config:
        json_schema_extra = {
            "example": {
                "nodes": [
                    {
                        "name": "Favorite Movie",
                        "properties": {"genre": "Sci-Fi", "rating": "9/10"},
                        "perspective": "user preference"
                    }
                ],
                "relationships": [
                    {
                        "source": "Favorite Movie",
                        "target": "Interstellar",
                        "relation_type": "IS",
                        "data": {"confidence": 0.9}
                    }
                ]
            }
        }