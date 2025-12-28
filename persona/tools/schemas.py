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


MEMORY_TOOLS = [RECALL_TOOL, RECORD_TOOL]


def get_tool_by_name(name: str) -> Dict[str, Any]:
    for tool in MEMORY_TOOLS:
        if tool["function"]["name"] == name:
            return tool
    raise ValueError(f"Unknown tool: {name}")


def get_all_tools() -> List[Dict[str, Any]]:
    return MEMORY_TOOLS
