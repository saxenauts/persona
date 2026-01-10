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
                            "MENTIONS",
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
                    "enum": [
                        "LED_TO",
                        "CAUSED_BY",
                        "NEXT",
                        "PREVIOUS",
                        "RELATES_TO",
                        "MENTIONS",
                    ],
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

BROWSE_TOOL = {
    "type": "function",
    "function": {
        "name": "browse",
        "description": "Time-ordered listing of memories. Unlike recall (similarity search), browse returns memories sorted by event_time. Use for temporal questions like 'what happened last week' or 'show me memories from June 2023'.",
        "parameters": {
            "type": "object",
            "properties": {
                "date_start": {
                    "type": "string",
                    "description": "Start date (ISO 8601: YYYY-MM-DD). Only return memories from this date onward.",
                },
                "date_end": {
                    "type": "string",
                    "description": "End date (ISO 8601: YYYY-MM-DD). Only return memories up to this date.",
                },
                "memory_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["episode", "psyche", "note", "entity"],
                    },
                    "description": "Filter by memory type.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return. Default: 20.",
                },
                "order": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "description": "Sort order by event_time. 'asc' = oldest first, 'desc' = newest first. Default: desc.",
                },
            },
            "required": [],
        },
    },
}

GET_MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "get_memory",
        "description": "Fetch a single memory by ID. Returns full content (not just a snippet). Use after recall/browse to get complete details of a specific memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "UUID of the memory to fetch (from recall/browse results).",
                },
            },
            "required": ["memory_id"],
        },
    },
}

UPDATE_MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "update_memory",
        "description": "Update fields on an existing memory. Use to mark tasks complete, change due dates, edit content, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "UUID of the memory to update.",
                },
                "updates": {
                    "type": "object",
                    "description": "Fields to update. Allowed: title, content, status (for notes: 'active', 'completed', 'cancelled'), due_date (ISO format), importance (0.0-1.0).",
                    "properties": {
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["active", "completed", "cancelled"],
                        },
                        "due_date": {"type": "string"},
                        "importance": {"type": "number"},
                    },
                },
            },
            "required": ["memory_id", "updates"],
        },
    },
}


MEMORY_TOOLS = [
    RECALL_TOOL,
    RECORD_TOOL,
    BROWSE_TOOL,
    GET_MEMORY_TOOL,
    UPDATE_MEMORY_TOOL,
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
