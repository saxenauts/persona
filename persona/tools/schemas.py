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
        "description": """Explore graph connections from a memory node. Returns neighbors linked by relationships. Use after recall() to explore interesting connections.

WHEN TO USE: After finding a relevant memory via recall(), use to discover related context through graph connections. Useful for understanding causality chains, temporal sequences, or entity mentions.

RELATIONSHIP TYPES:
- **LED_TO**: Causal chain (X led to Y happening). Example: "Started exercising" LED_TO "Improved energy levels"
- **CAUSED_BY**: Reverse causal (Y was caused by X). Example: "Missed deadline" CAUSED_BY "Unexpected illness"
- **NEXT**: Temporal sequence (what happened after). Example: "Dinner with Sarah" NEXT "Watched movie together"
- **PREVIOUS**: Temporal sequence (what happened before). Example: "Job interview" PREVIOUS "Received offer"
- **RELATES_TO**: General association (connected but not causal/temporal). Example: "AI research" RELATES_TO "Philosophy of mind"
- **MENTIONS**: Entity reference (memory mentions a person/place/thing). Example: "Team meeting" MENTIONS "Sarah"

INTERPRETING RESULTS:
- Results show neighbors with their relationship type and content snippet
- Use relationship type to understand connection: causal (LED_TO/CAUSED_BY), temporal (NEXT/PREVIOUS), associative (RELATES_TO), or reference (MENTIONS)
- Graph expansions can be noisy—filter by specific relationship_types if you need focused results
- For narrative continuity: Use NEXT/PREVIOUS to trace chronological chains
- For causality: Use LED_TO/CAUSED_BY to understand cause-effect relationships
- For entity context: Use MENTIONS to find all memories referencing a person/place/thing
- If results are too broad, try follow_relationship() instead (more targeted, single relationship type)""",
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
                    "description": "Filter by relationship types. If omitted, returns all. Use to narrow results: e.g., ['NEXT', 'PREVIOUS'] for temporal chains, ['LED_TO', 'CAUSED_BY'] for causality.",
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
        "description": """Trace a specific relationship chain from a memory. More targeted than expand_neighbors - follows one relationship type.

WHEN TO USE: When you need to trace a specific chain (e.g., "what happened next?" or "what caused this?"). More focused than expand_neighbors because it follows only one relationship type.

RELATIONSHIP TYPES:
- **LED_TO**: Causal chain (X led to Y happening). Example: "Started exercising" LED_TO "Improved energy levels" LED_TO "Joined gym"
- **CAUSED_BY**: Reverse causal (Y was caused by X). Example: "Missed deadline" CAUSED_BY "Unexpected illness" CAUSED_BY "Flu outbreak"
- **NEXT**: Temporal sequence (what happened after). Example: "Dinner with Sarah" NEXT "Watched movie" NEXT "Went home"
- **PREVIOUS**: Temporal sequence (what happened before). Example: "Job interview" PREVIOUS "Prepared resume" PREVIOUS "Updated LinkedIn"
- **RELATES_TO**: General association (connected but not causal/temporal). Example: "AI research" RELATES_TO "Philosophy of mind" RELATES_TO "Ethics"
- **MENTIONS**: Entity reference (memory mentions a person/place/thing). Example: "Team meeting" MENTIONS "Sarah" MENTIONS "Project Alpha"

INTERPRETING RESULTS:
- Results show a chain of memories connected by the specified relationship type
- Order reflects the chain direction: LED_TO shows forward causality, CAUSED_BY shows backward causality
- NEXT/PREVIOUS show temporal sequences (use NEXT for "what happened after", PREVIOUS for "what came before")
- MENTIONS chains show all memories referencing the same entity
- Use this when you need a focused narrative chain (e.g., "trace the cause-effect chain" or "show me the sequence of events")
- Compare to expand_neighbors: expand_neighbors shows all connections at once (broader), follow_relationship traces one type (deeper)

COMMON PATTERNS:
- Narrative continuity: follow_relationship(source_id, "NEXT") to trace "what happened next"
- Causality chains: follow_relationship(source_id, "LED_TO") to trace "what did this lead to"
- Entity mentions: follow_relationship(source_id, "MENTIONS") to find all memories about a person/place""",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "UUID of the starting memory (get this from recall() or expand_neighbors() results).",
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
                    "description": "The relationship type to follow. Choose based on what you're tracing: LED_TO/CAUSED_BY for causality, NEXT/PREVIOUS for temporal sequence, MENTIONS for entity references.",
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
        "description": """Fetch full content of a specific memory by ID. Returns complete memory with all metadata, attributes, and relationships.

WHEN TO USE: After recall/browse when snippet is insufficient for accurate answer. Use when you need exact details (names, dates, specific quotes, full context, or relationship information).

INTERPRETING RESULTS:
- Returns full memory object with id, type, title, content, timestamps, and type-specific fields
- For NOTES: Includes status (active/completed/cancelled), due_date, note_type, importance
- For ENTITIES: Includes entity_type, canonical_name, aliases, description, and attributes (key-value facts with update timestamps)
- For EPISODES/PSYCHE: Includes title, content, event_time, observed_at
- event_time = when the event/fact occurred; observed_at = when it was recorded
- Use full content to cite exact details in your response (quotes, specific facts, dates)
- Use attributes on entities to answer "what do we know about X?" questions
- Use status on notes to understand task state (active = pending, completed = done, cancelled = abandoned)

WHEN TO STOP vs CONTINUE:
- STOP (have enough info): You have exact details needed to answer the question accurately
- CONTINUE (need more context): Memory references other entities/events → use expand_neighbors() or follow_relationship() to trace connections
- EXAMPLE: User asks "What did Sarah say about the project?" → get_memory() shows Sarah mentioned, but content is vague → use expand_neighbors(MENTIONS) to find all Sarah-related memories

COMMON WORKFLOWS:
1. After recall() snippet insufficient: get_memory(id) to get full content for accurate citation
2. Before update_memory(): get_memory(id) to verify correct memory and see current state
3. To understand entity context: get_memory(entity_id) to see all attributes and relationships
4. To trace narrative: get_memory(id) then expand_neighbors() to find connected events""",
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
