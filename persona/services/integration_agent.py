"""Integration Agent Service: Background agent for connecting memories to the graph.

Runs asynchronously after ingestion to:
- Find unprocessed memories
- Link them to entities they mention
- Create causal chains (LED_TO, CAUSED_BY)
- Flag contradictions for review
- Merge duplicate entities
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from persona.core.graph_ops import GraphOps
from persona.core.memory_store import MemoryStore
from persona.llm.client_factory import get_chat_client
from persona.llm.providers.base import ChatMessage, ToolCall, ToolResult
from persona.tools.integration import (
    IntegrationContext,
    INTEGRATION_HANDLERS,
    GraphPatch,
    GraphPatchResult,
)
from persona.services.consolidation_service import maybe_run_consolidation
from server.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class IntegrationAgentConfig:
    """Configuration for integration agent runs."""

    max_turns: int = 10
    max_tool_calls: int = 50
    tool_timeout: float = 30.0
    temperature: float = 0.3  # Lower temp for more deterministic linking
    max_tokens: int = 4096


# =============================================================================
# System Prompt
# =============================================================================


INTEGRATION_AGENT_PROMPT = """You are a background integration agent for a personal memory graph.

Your job: Connect new memories to the existing graph through semantic relationships.

## WORKFLOW

1. Call get_unintegrated() to see memories needing processing
2. For each memory, call recall() to find related existing memories
3. Use expand_neighbors() to explore graph context when useful
4. Apply connections via commit_patch()

## CONNECTION TYPES

| Type | When to Use | Example |
|------|-------------|---------|
| MENTIONS | Episode/Note references an Entity | "Met with Sarah" → Entity:Sarah |
| LED_TO | Causal forward: A caused B | "Got promotion" → "Celebrated" |
| CAUSED_BY | Causal backward: B caused by A | "Burnout" ← "3 months crunch" |
| NEXT/PREVIOUS | Temporal sequence | Meeting Monday → Follow-up Tuesday |
| RELATES_TO | Thematic association | Both about "career" |
| CONTRADICTS | Information conflicts | "Loves coffee" vs "Hates coffee" |
| SAME_AS | Entity dedup | "Sam" = "Samuel Chen" |

## ENTITY RESOLUTION (merge operation)

Detect when different names refer to the same person/thing:

EXAMPLE: New memory mentions "my wife Sarah" but existing entity is "Sarah Chen"
→ recall("Sarah") finds Entity:Sarah Chen (id: abc-123)
→ Compare: same person (wife, name matches)
→ commit_patch with merge operation linking the reference

SIGNALS for same entity:
- Same name or nickname ("Sam" / "Samuel" / "Sammy")
- Same role description ("my manager" / "boss" / "Alex the PM")
- Context confirms ("Sarah from Google" = "Sarah Chen" if she works at Google)

BE CONSERVATIVE: Only merge when confident. If unsure, use RELATES_TO instead.

## CONTRADICTION DETECTION (flag operation)

Detect when new info conflicts with existing:

EXAMPLE: New psyche says "Prefers remote work" but existing says "Loves office environment"
→ These contradict - flag both with CONTRADICTS relationship
→ Use flag operation with flag_type: "contradiction"

SIGNALS for contradiction:
- Opposite preferences/values
- Conflicting facts about same entity
- Timeline inconsistencies

DON'T flag as contradiction:
- Evolution over time ("Used to like X, now prefers Y" - that's change, not contradiction)
- Different contexts ("Likes coffee at work, tea at home")

## TEMPORAL LINKING (LED_TO, NEXT/PREVIOUS)

Connect events in causal or temporal chains:

EXAMPLE: Episode "Had argument with manager about timeline" 
→ recall("manager timeline") finds "Updated resume" from next day
→ Likely LED_TO relationship (argument caused resume update)

SIGNALS for LED_TO:
- Explicit causation ("because", "led to", "resulted in")
- Temporal proximity + logical connection
- Decision followed by action

## COMMIT_PATCH FORMAT

```json
{
  "items": [
    {"operation": "link", "source_id": "uuid", "target_id": "uuid", "relation_type": "LED_TO", "reason": "Brief why"},
    {"operation": "merge", "source_id": "uuid", "target_id": "uuid", "reason": "Same person"},
    {"operation": "flag", "source_id": "uuid", "properties": {"flag_type": "contradiction"}, "reason": "Conflicts with X"},
    {"operation": "mark_integrated", "source_id": "uuid"}
  ],
  "dry_run": false
}
```

## GUIDELINES

1. Start with get_unintegrated() - process oldest first (causal ordering)
2. For each memory, search for potential connections with recall()
3. Quality over quantity - don't over-link. 2-3 strong links > 10 weak ones
4. Always mark_integrated after processing
5. Batch operations in single commit_patch when possible
6. Stop when all memories are processed - report summary

When done, summarize: "Processed N memories. Created X links, Y merges, Z flags.\""""


# =============================================================================
# Tool Schemas
# =============================================================================


INTEGRATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Search for existing memories semantically related to a query. Use to find potential connections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in existing memories.",
                    },
                    "date_start": {
                        "type": "string",
                        "description": "Optional start date filter (YYYY-MM-DD).",
                    },
                    "date_end": {
                        "type": "string",
                        "description": "Optional end date filter (YYYY-MM-DD).",
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
                        "description": "Max results. Default: 10.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand_neighbors",
            "description": "Explore graph connections from a memory node. Returns linked memories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "UUID of memory to expand from.",
                    },
                    "relationship_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by relationship types (LED_TO, MENTIONS, etc.).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max neighbors. Default: 10.",
                    },
                },
                "required": ["memory_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_unintegrated",
            "description": "Get memories that haven't been integrated yet. Call this first to see work to do.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max memories to return. Default: 20.",
                    },
                    "memory_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["episode", "psyche", "note", "entity"],
                        },
                        "description": "Filter by memory type.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "commit_patch",
            "description": "Apply graph mutations. Create links, merge entities, flag contradictions, mark as integrated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patch_json": {
                        "type": "string",
                        "description": "JSON string of GraphPatch with items array. See system prompt for format.",
                    },
                },
                "required": ["patch_json"],
            },
        },
    },
]


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class IntegrationResult:
    """Result of an integration agent run."""

    success: bool
    memories_processed: int = 0
    links_created: int = 0
    flags_raised: int = 0
    merges_performed: int = 0
    turns: int = 0
    tool_calls_made: int = 0
    duration_ms: float = 0.0
    summary: str = ""
    errors: List[str] = field(default_factory=list)

    # For resumption if interrupted
    state: Optional[str] = None
    can_resume: bool = False


# =============================================================================
# Integration Tool Registry
# =============================================================================


class IntegrationToolRegistry:
    """Tool registry for integration agent with IntegrationContext."""

    def __init__(self):
        self.handlers = INTEGRATION_HANDLERS

    async def execute(self, tool_call: ToolCall, ctx: IntegrationContext) -> ToolResult:
        handler = self.handlers.get(tool_call.name)
        if not handler:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=json.dumps({"error": f"Unknown tool: {tool_call.name}"}),
            )

        try:
            args = json.loads(tool_call.arguments)
            result = await handler(ctx, **args)

            # Serialize result
            if hasattr(result, "__dict__"):
                content = json.dumps(result.__dict__, default=str)
            elif isinstance(result, (dict, list)):
                content = json.dumps(result, default=str)
            else:
                content = str(result)

            return ToolResult(tool_call_id=tool_call.id, content=content)

        except json.JSONDecodeError as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=json.dumps({"error": f"Invalid arguments JSON: {e}"}),
            )
        except Exception as e:
            logger.error(f"Integration tool error for {tool_call.name}: {e}")
            return ToolResult(
                tool_call_id=tool_call.id,
                content=json.dumps({"error": str(e)}),
            )


# =============================================================================
# Main Entry Point
# =============================================================================


async def run_integration_agent(
    user_id: str,
    trigger_ids: List[str],
    session_id: Optional[str] = None,
    graph_ops: Optional[GraphOps] = None,
    config: Optional[IntegrationAgentConfig] = None,
) -> IntegrationResult:
    """Run the integration agent to connect new memories to the graph.

    Args:
        user_id: User whose memories to integrate
        trigger_ids: Memory IDs that triggered this run (from recent ingestion)
        session_id: If provided, only process memories from this session
        graph_ops: Optional GraphOps instance (will create one if not provided)
        config: Optional configuration overrides

    Returns:
        IntegrationResult with statistics and any errors
    """
    config = config or IntegrationAgentConfig()
    run_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    scope_msg = (
        f"session={session_id}" if session_id else f"triggers={len(trigger_ids)}"
    )
    logger.info(
        f"Starting integration agent run {run_id} for user {user_id}, {scope_msg}"
    )

    # Build context
    own_graph_ops = graph_ops is None
    if own_graph_ops:
        graph_ops = GraphOps()
        await graph_ops.__aenter__()

    try:
        store = MemoryStore(graph_ops.graph_db)

        ctx = IntegrationContext(
            user_id=user_id,
            graph_ops=graph_ops,
            store=store,
            trigger_ids=trigger_ids,
            run_id=run_id,
            session_id=session_id,
        )

        # Initialize LLM and registry
        llm = get_chat_client()
        registry = IntegrationToolRegistry()

        # Build initial messages
        messages = [
            ChatMessage(role="system", content=INTEGRATION_AGENT_PROMPT),
            ChatMessage(
                role="user",
                content=f"New memories have been ingested for integration. "
                f"Trigger memory IDs: {trigger_ids[:5]}{'...' if len(trigger_ids) > 5 else ''}. "
                f"Start by checking get_unintegrated() to see what needs processing.",
            ),
        ]

        # Run agent loop
        total_tool_calls = 0
        turns = 0
        links_created = 0
        flags_raised = 0
        merges_performed = 0
        memories_processed = 0
        errors: List[str] = []

        while turns < config.max_turns and total_tool_calls < config.max_tool_calls:
            turns += 1

            try:
                response = await llm.chat(
                    messages=messages,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    tools=INTEGRATION_TOOLS,
                )
            except Exception as e:
                logger.error(f"LLM call failed in integration agent: {e}")
                errors.append(f"LLM error: {e}")
                break

            # Check if agent is done (no tool calls)
            if response.stop_reason != "tool_use" or not response.tool_calls:
                logger.info(f"Integration agent completed after {turns} turns")
                break

            # Add assistant message
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )

            # Execute tool calls
            for tool_call in response.tool_calls:
                total_tool_calls += 1

                result = await registry.execute(tool_call, ctx)
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=result.content,
                        tool_call_id=tool_call.id,
                    )
                )

                # Track metrics from commit_patch results
                if tool_call.name == "commit_patch":
                    try:
                        result_data = json.loads(result.content)
                        if result_data.get("success"):
                            applied = result_data.get("applied", 0)
                            links_created += applied

                            # Parse the patch to count specific operations
                            args = json.loads(tool_call.arguments)
                            patch_json = args.get("patch_json", "{}")
                            patch_data = json.loads(patch_json)
                            for item in patch_data.get("items", []):
                                op = item.get("operation")
                                if op == "flag":
                                    flags_raised += 1
                                elif op == "merge":
                                    merges_performed += 1
                                elif op == "mark_integrated":
                                    memories_processed += 1
                    except (json.JSONDecodeError, KeyError):
                        pass

        duration_ms = (time.time() - start_time) * 1000

        summary = (
            f"Integration run {run_id}: {memories_processed} memories processed, "
            f"{links_created} operations applied, {flags_raised} flags, {merges_performed} merges. "
            f"{turns} turns, {total_tool_calls} tool calls, {duration_ms:.0f}ms"
        )
        logger.info(summary)

        if memories_processed > 0:
            try:
                await maybe_run_consolidation(user_id, graph_ops)
            except Exception as e:
                logger.warning(f"Consolidation after integration failed: {e}")

        return IntegrationResult(
            success=len(errors) == 0,
            memories_processed=memories_processed,
            links_created=links_created,
            flags_raised=flags_raised,
            merges_performed=merges_performed,
            turns=turns,
            tool_calls_made=total_tool_calls,
            duration_ms=duration_ms,
            summary=summary,
            errors=errors,
        )

    finally:
        if own_graph_ops and graph_ops:
            await graph_ops.__aexit__(None, None, None)


# =============================================================================
# Utility: Trigger Integration After Ingestion
# =============================================================================


async def maybe_trigger_integration(
    user_id: str,
    new_memory_ids: List[str],
    threshold: int = 5,
    graph_ops: Optional[GraphOps] = None,
) -> Optional[IntegrationResult]:
    """Conditionally trigger integration based on batch size.

    Call this after ingestion. Integration runs if:
    - Number of new memories exceeds threshold, OR
    - Contains high-importance memories (entities, notes)

    Args:
        user_id: User ID
        new_memory_ids: IDs of newly ingested memories
        threshold: Minimum memories before triggering
        graph_ops: Optional GraphOps to reuse

    Returns:
        IntegrationResult if integration ran, None otherwise
    """
    if len(new_memory_ids) < threshold:
        logger.debug(
            f"Skipping integration: {len(new_memory_ids)} < {threshold} threshold"
        )
        return None

    logger.info(f"Triggering integration for {len(new_memory_ids)} new memories")
    return await run_integration_agent(
        user_id=user_id,
        trigger_ids=new_memory_ids,
        graph_ops=graph_ops,
    )
