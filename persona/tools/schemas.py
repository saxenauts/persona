from typing import List, Dict, Any


RECALL_TOOL = {
    "type": "function",
    "function": {
        "name": "recall",
        "description": "Ask the user's memory in natural language. Call multiple times in parallel for different queries (time slice, topic, related memories).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Free-form request. Include time/topic/type cues in plain English.",
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
        "description": "Store information to user's memory. System infers memory type and links automatically.",
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
        "description": "Expand from a memory to find connected memories via graph relationships. Use after recall() to explore interesting connections from a specific memory ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "UUID of the memory to expand from (get this from recall() results).",
                },
                "relationship_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by relationship types (e.g., ['LED_TO', 'CAUSED_BY', 'NEXT']). If omitted, returns all relationships.",
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
        "description": "Follow a specific relationship type from a memory to trace causal chains or thematic connections. More targeted than expand_neighbors.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "UUID of the starting memory.",
                },
                "relation_type": {
                    "type": "string",
                    "description": "The relationship type to follow (e.g., 'LED_TO', 'CAUSED_BY', 'NEXT', 'PREVIOUS').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of connected memories to return. Default: 5.",
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
