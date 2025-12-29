"""Tool schemas - the hippocampal index protocol.

LLM provides structured parameters directly (dates, types, filters).
No natural language parsing on our side - LLM does the translation.
"""

from typing import List, Dict, Any


RECALL_TOOL = {
    "type": "function",
    "function": {
        "name": "recall",
        "description": "Search user's memory with optional filters. Returns matching memories ranked by relevance. Call multiple times in parallel for different slices (time periods, topics, types).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Semantic search query. What to look for in memory content.",
                },
                "date_start": {
                    "type": "string",
                    "description": "Start date filter (ISO 8601: YYYY-MM-DD). Only return memories from this date onward.",
                },
                "date_end": {
                    "type": "string",
                    "description": "End date filter (ISO 8601: YYYY-MM-DD). Only return memories up to this date.",
                },
                "memory_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["episode", "psyche", "note"]},
                    "description": "Filter by memory type. episode=events, psyche=traits/preferences, note=goals/tasks.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return. Default: 10.",
                },
            },
            "required": ["query"],
        },
    },
}

RECORD_TOOL = {
    "type": "function",
    "function": {
        "name": "record",
        "description": "Store information to user's memory. System infers memory type and creates links automatically.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The information to store, in natural language.",
                },
            },
            "required": ["text"],
        },
    },
}

EXPAND_NEIGHBORS_TOOL = {
    "type": "function",
    "function": {
        "name": "expand_neighbors",
        "description": "Explore graph connections from a memory node. Returns neighbors linked by relationships. Use after recall() to explore interesting connections.",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "UUID of the memory to expand from (get this from recall() results).",
                },
                "relationship_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "LED_TO",
                            "CAUSED_BY",
                            "NEXT",
                            "PREVIOUS",
                            "RELATES_TO",
                        ],
                    },
                    "description": "Filter by relationship types. If omitted, returns all.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum neighbors to return. Default: 10.",
                },
            },
            "required": ["memory_id"],
        },
    },
}

FOLLOW_RELATIONSHIP_TOOL = {
    "type": "function",
    "function": {
        "name": "follow_relationship",
        "description": "Trace a specific relationship chain from a memory. More targeted than expand_neighbors - follows one relationship type.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "UUID of the starting memory.",
                },
                "relation_type": {
                    "type": "string",
                    "enum": ["LED_TO", "CAUSED_BY", "NEXT", "PREVIOUS", "RELATES_TO"],
                    "description": "The relationship type to follow.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum connected memories to return. Default: 5.",
                },
            },
            "required": ["source_id", "relation_type"],
        },
    },
}


MEMORY_TOOLS = [
    RECALL_TOOL,
    RECORD_TOOL,
    EXPAND_NEIGHBORS_TOOL,
    FOLLOW_RELATIONSHIP_TOOL,
]


def get_tool_by_name(name: str) -> Dict[str, Any]:
    for tool in MEMORY_TOOLS:
        if tool["function"]["name"] == name:
            return tool
    raise ValueError(f"Unknown tool: {name}")


def get_all_tools() -> List[Dict[str, Any]]:
    return MEMORY_TOOLS
