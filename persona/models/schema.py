"""
Schema and Pydantic Models for Persona API.

This file contains models used by the API layer. For memory-related models,
see persona.models.memory instead.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any


# =============================================================================
# API Request/Response Models
# =============================================================================

class UserCreate(BaseModel):
    user_id: str

class RAGQuery(BaseModel):
    query: str

class RAGResponse(BaseModel):
    answer: str

class AskRequest(BaseModel):
    query: str
    output_schema: Dict[str, Any] = Field(..., description="Expected output structure with example values")

class AskResponse(BaseModel):
    result: Dict[str, Any]


# =============================================================================
# Dynamic Schema Helper
# =============================================================================

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