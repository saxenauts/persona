from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class UnstructuredData(BaseModel):
    title: str
    content: str
    metadata: Optional[Dict[str, str]] = {}

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