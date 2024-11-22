"""
Schema and Pydantic Models for the Graph Library Ops. TODO put in an apt folder"
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from instructor import OpenAISchema

class UnstructuredData(BaseModel):
    title: str
    content: str
    metadata: Optional[Dict[str, str]] = {}

class Node(OpenAISchema):
    name: str
    perspective: Optional[str] = None

class Relationship(OpenAISchema):
    source: str
    target: str
    relation: str

class NodeModel(BaseModel):
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
                "embedding": [0.1, 0.2, ...]  # Simplified example
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
                    {"name": "Node1", "properties": {"frequency": 1}, "embedding": [0.1, 0.2, ...]},
                    {"name": "Node2", "properties": {"frequency": 1}, "embedding": [0.1, 0.2, ...]}
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
        {"id": "Blockchain", "label": "Technology"},
        {"id": "Quantum Computing", "label": "Science"},
        {"id": "Indie Games", "label": "Entertainment"},
        {"id": "Sustainable Farming", "label": "Agriculture"},
        {"id": "Virtual Reality", "label": "Technology"}
    ])
    relationships: List[RelationshipModel] = Field(..., example=[
        {"source": "Technology", "relation": "includes", "target": "Blockchain"},
        {"source": "Science", "relation": "includes", "target": "Quantum Computing"},
        {"source": "Entertainment", "relation": "includes", "target": "Indie Games"},
        {"source": "Agriculture", "relation": "includes", "target": "Sustainable Farming"},
        {"source": "Technology", "relation": "includes", "target": "Virtual Reality"}
    ])

class UserCreate(BaseModel):
    user_id: str

class IngestData(BaseModel):
    content: str

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

    @classmethod
    def model_validate(cls, obj):
        if isinstance(obj, dict) and 'created_at' in obj:
            obj['created_at'] = str(obj['created_at'])
        return super().model_validate(obj)

class LearnRequest(BaseModel):
    user_id: str
    schema: GraphSchema
    description: str

class LearnResponse(BaseModel):
    status: str
    schema_id: str
    details: str

class AskRequest(BaseModel):
    user_id: str
    query: str
    output_schema: Dict[str, Any] = Field(..., description="Expected output structure with example values")

class AskResponse(BaseModel):
    result: Dict[str, Any]

class AskResponseInstructor(OpenAISchema):
    result: Dict[str, Any]

class PersonalizeRequest(BaseModel):
    user_id: str
    personalization_criteria: Dict[str, Any]

class PersonalizeResponse(BaseModel):
    status: str
    details: str

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