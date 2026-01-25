"""Tool schemas - the hippocampal index protocol.

LLM provides structured parameters directly (dates, types, filters).
No natural language parsing on our side - LLM does the translation.
"""

from typing import List, Dict, Any


RECALL_TOOL = {
    "type": "function",
    "function": {
        "name": "recall",
        "description": """Search user's memory. REQUIRED before answering questions about user's life, preferences, or experiences.

WHEN TO USE: Before answering ANY user-specific question. If you answer without searching, you WILL be wrong.

INTERPRETING RESULTS: 
- Results are ranked by relevance (most relevant first)
- Read ALL results to understand full context—user's state may have evolved
- Check timestamps (event_time, age_days) to identify chronological order
- If results show evolution: Look for explicit change signals ("changed", "now prefer", "switched to")
- If results conflict without resolution: Note both, cite the more recent, ask if clarification needed
- Use specific details from results in your response

IF RESULTS ARE THIN OR MIXED:
- Try broader query (remove specific terms)
- Try related concepts or entity names from initial results
- Use expand_neighbors() on promising results to find connections""",
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
                    "items": {
                        "type": "string",
                        "enum": ["episode", "psyche", "note", "entity"],
                    },
                    "description": "Filter by memory types. Use to narrow retrieval when the question is clearly about one kind of memory (e.g., preferences = psyche, tasks = note).",
                },
                "exclude_transcripts": {
                    "type": "boolean",
                    "description": "If true, exclude transcript-like memories from results (default behavior). Set to false only when you explicitly need conversational transcripts.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return. Default: 10.",
                },
                "order": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "description": "Sort by event_time instead of relevance. 'asc' = oldest first, 'desc' = newest first. Omit for relevance ranking.",
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
        "description": """Store information to user's memory. System infers memory type and creates links automatically.

WHEN TO USE:
- User states a preference, value, or belief ("I prefer X", "I believe Y", "I'm interested in Z")
- User provides biographical facts ("My birthday is...", "I work at...", "I live in...")
- User commits to a task or goal ("I want to...", "I need to...", "I'm planning to...")
- User explicitly asks to remember something ("Remember that...", "Save this...", "Don't forget...")
- Information is durable and worth long-term retention (not ephemeral chat)

WHEN NOT TO USE:
- NEVER record secrets: API keys, passwords, credentials, tokens, private keys
- NEVER record system text: tool outputs, error messages, internal logs, code snippets
- NEVER record ephemeral chat: casual conversation, temporary thoughts, one-off comments
- NEVER record if user says "don't save this" or "off the record"
- NEVER record raw transcripts or unprocessed conversation dumps
- NEVER record sensitive personal data without explicit consent

INTERPRETING RESULTS:
- Success: Returns list of memory IDs (e.g., [{"id": "uuid", "type": "psyche"}])
- Each stored item has an id (UUID) and type (episode/psyche/note/entity)
- If empty list: Information was not stored (may be filtered as ephemeral or invalid)
- If record fails: Check that text is not empty and doesn't contain secrets
- Confirmation: After recording, you can recall() the same topic to verify it was saved""",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The information to store, in natural language. Should be durable, meaningful information (not ephemeral chat).",
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
        "description": """Time-ordered listing of memories. Unlike recall (semantic search), browse returns memories sorted by time.

WHEN TO USE: 
- "What happened last week/month?" 
- "Show me memories from [date range]"
- When you need chronological understanding of how something evolved
- When a question asks for ordering or sequence (use order='asc' to get oldest-first)

INTERPRETING RESULTS: Results ordered by time (default: newest first). Use 'asc' order to see oldest first when tracking evolution.""",
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
                    "description": "Sort order by event_time. 'asc' = oldest first (use for ordering questions), 'desc' = newest first. Default: desc.",
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
        "description": """Fetch full content of a specific memory by ID.

WHEN TO USE: After recall/browse when snippet is insufficient for accurate answer. Use when you need exact details (names, dates, specific quotes).""",
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
        "description": """Modify fields on an existing memory. Use to mark tasks complete, edit content, update due dates, or correct mistakes.

WHEN TO USE:
- Mark a note/task as done: status='completed' when user finishes something
- Mark a note/task as cancelled: status='cancelled' when user abandons it
- Edit memory content/title: Fix mistakes or add clarifications to existing memories
- Update importance: Adjust priority (0.0-1.0) for notes/tasks
- Update due_date: Change deadline for notes/tasks (ISO format: YYYY-MM-DD)

WORKFLOW (CRITICAL - ALWAYS FOLLOW):
1. FIND the memory: Use recall() or browse() to locate the memory
2. VERIFY the memory: Use get_memory(memory_id) to confirm correct memory before editing
3. UPDATE: Call update_memory() with memory_id and updates
4. NEVER update based on recall() snippets alone - always get_memory() first

PILLAR-SPECIFIC SEMANTICS:
- NOTES (tasks/goals): Can update status (active/completed/cancelled), due_date, importance, content
- EPISODES (events): Can update content/title only (append-only principle - prefer creating new memories)
- PSYCHE (traits/beliefs): Can update content/title only (consolidate via new memories, not edits)
- ENTITY (people/places/things): Can update content/title only (facts evolve via new memories)

SAFETY NOTES:
- Prefer creating new memories over editing old ones (append-only principle for Episodes/Psyche/Entity)
- Only Notes should have status changes - Episodes/Psyche/Entity don't have status
- Always verify memory_id before updating - wrong ID will corrupt unrelated memory

INTERPRETING RESULTS:
- Success: Returns {success: true, memory_id: "...", updated_fields: ["field1", "field2"]}
- Failure: Returns {success: false} - check that memory_id exists and is valid UUID
- Verify changes: After update, you can get_memory(memory_id) to confirm changes took effect""",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "UUID of the memory to update (get this from recall/browse/get_memory results).",
                },
                "updates": {
                    "type": "object",
                    "description": "Fields to update. Allowed: title, content, status (for notes: 'active', 'completed', 'cancelled'), due_date (ISO format: YYYY-MM-DD), importance (0.0-1.0).",
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


RESOLVE_DATE_RANGE_TOOL = {
    "type": "function",
    "function": {
        "name": "resolve_date_range",
        "description": """Convert natural-language time reference into ISO date range for recall()/browse().

WHEN TO USE: User references relative time ("last week", "past 3 days", "in January", "before X event").
OUTPUT: { "date_start": "YYYY-MM-DD", "date_end": "YYYY-MM-DD", "explanation": "..." }""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The time reference to resolve, e.g. 'last week', 'January 2024'.",
                },
            },
            "required": ["query"],
        },
    },
}

TIMELINE_TOOL = {
    "type": "function",
    "function": {
        "name": "timeline",
        "description": """Trace a subject through time in CHRONOLOGICAL ORDER (oldest first).

WHEN TO USE:
- Questions about ORDER or SEQUENCE: "In what order did I...", "What came first..."
- Questions about EVOLUTION: "How did X change over time...", "When did I first/last..."
- Any question needing chronological understanding of events

DIFFERENCE FROM recall():
- recall() ranks by RELEVANCE (best match first)
- timeline() sorts by TIME (oldest first) - use this when order matters

INTERPRETING RESULTS: Items are sorted oldest-to-newest. The sequence reflects when events occurred.""",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "What to trace through time. Can be a topic, person, project, concept.",
                },
                "date_start": {
                    "type": "string",
                    "description": "Start date filter (ISO 8601: YYYY-MM-DD).",
                },
                "date_end": {
                    "type": "string",
                    "description": "End date filter (ISO 8601: YYYY-MM-DD).",
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
            },
            "required": ["subject"],
        },
    },
}

MEMORY_TOOLS = [
    RECALL_TOOL,
    RECORD_TOOL,
    RESOLVE_DATE_RANGE_TOOL,
    TIMELINE_TOOL,
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
