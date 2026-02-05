"""Integration Tools: Background agent tools for connecting new memories to the graph.

Different context than chat tools - operates on unprocessed memories with explicit
graph patching. Reuses recall/expand from memory.py, adds integration-specific tools.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum
from pydantic import BaseModel, Field

from persona.tools.memory import recall_handler, expand_neighbors_handler, MemoryHit
from persona.models.memory import MemoryLink
from server.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# GraphPatch Models (Declarative Graph Mutations)
# =============================================================================


class PatchOperation(str, Enum):
    """Operations that can be applied to the memory graph."""

    # Relationship operations
    LINK = "link"  # Create relationship between memories
    UNLINK = "unlink"  # Remove relationship

    # Entity operations
    MERGE = "merge"  # Merge two entities (SAME_AS)
    UPDATE_ATTR = "update_attr"  # Update entity attribute

    # Memory state operations
    FLAG = "flag"  # Flag for review (contradictions, etc.)
    MARK_INTEGRATED = "mark_integrated"  # Mark memory as processed


class GraphPatchItem(BaseModel):
    """A single graph mutation operation."""

    operation: PatchOperation
    source_id: str = Field(..., description="UUID of source memory/entity")
    target_id: Optional[str] = Field(None, description="UUID of target (for links)")
    relation_type: Optional[str] = Field(
        None,
        description="Relationship type: LED_TO, MENTIONS, CONTRADICTS, SAME_AS, etc.",
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Additional properties for the operation"
    )
    reason: str = Field(
        default="", description="Brief explanation for this change (for audit)"
    )


class GraphPatch(BaseModel):
    """Batch of graph mutations to apply atomically."""

    items: List[GraphPatchItem] = Field(default_factory=list)
    dry_run: bool = Field(default=False, description="Validate only, don't apply")

    def add_link(
        self, source_id: str, target_id: str, relation_type: str, reason: str = ""
    ) -> "GraphPatch":
        """Fluent API: Add a link operation."""
        self.items.append(
            GraphPatchItem(
                operation=PatchOperation.LINK,
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                reason=reason,
            )
        )
        return self

    def add_flag(
        self, memory_id: str, flag_type: str, reason: str = ""
    ) -> "GraphPatch":
        """Fluent API: Flag a memory for review."""
        self.items.append(
            GraphPatchItem(
                operation=PatchOperation.FLAG,
                source_id=memory_id,
                properties={"flag_type": flag_type},
                reason=reason,
            )
        )
        return self

    def mark_integrated(self, memory_id: str) -> "GraphPatch":
        """Fluent API: Mark memory as integrated."""
        self.items.append(
            GraphPatchItem(
                operation=PatchOperation.MARK_INTEGRATED,
                source_id=memory_id,
            )
        )
        return self


class GraphPatchResult(BaseModel):
    """Result of applying a GraphPatch."""

    success: bool
    applied_count: int = 0
    failed_count: int = 0
    errors: List[str] = Field(default_factory=list)
    dry_run: bool = False


# =============================================================================
# Integration Context
# =============================================================================


@dataclass
class IntegrationContext:
    """Context for integration agent - extends ToolContext with integration-specific data.

    Unlike ToolContext (which handles user-facing chat), IntegrationContext
    is for background graph maintenance operations.
    """

    user_id: str
    graph_ops: Any  # GraphOps
    store: Any  # MemoryStore
    trigger_ids: List[str]  # Memory IDs that triggered this run
    run_id: str
    session_id: Optional[str] = None  # If set, scope to this session only
    checkpoint: Optional[Dict[str, Any]] = None

    # Execution tracking
    _subtasks: list = field(default_factory=list)

    def track_subtask(self, name: str, status: str = "pending") -> int:
        idx = len(self._subtasks)
        self._subtasks.append({"name": name, "status": status})
        return idx

    def complete_subtask(self, idx: int, status: str = "done") -> None:
        if 0 <= idx < len(self._subtasks):
            self._subtasks[idx]["status"] = status

    @property
    def subtask_summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for t in self._subtasks:
            counts[t["status"]] = counts.get(t["status"], 0) + 1
        return counts


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class UnintegratedResult:
    """Result from get_unintegrated tool."""

    items: List[MemoryHit] = field(default_factory=list)
    count: int = 0
    total_unprocessed: int = 0


@dataclass
class CommitResult:
    """Result from commit_patch tool."""

    success: bool = True
    applied: int = 0
    failed: int = 0
    errors: List[str] = field(default_factory=list)


# =============================================================================
# Tool Handlers
# =============================================================================


async def get_unintegrated_handler(
    ctx: IntegrationContext,
    limit: int = 20,
    memory_types: Optional[List[str]] = None,
) -> UnintegratedResult:
    """Fetch memories that haven't been integrated (integrated_at is null)."""
    try:
        all_nodes = await ctx.graph_ops.graph_db.get_all_nodes(ctx.user_id)

        unprocessed = []
        for node in all_nodes:
            node_type = node.get("type")
            if node_type in ("memeplex", "usercard"):
                continue

            if memory_types and node_type not in memory_types:
                continue

            if ctx.session_id:
                session_filter = ctx.session_id.lower()
                if session_filter not in {"all", "*"} and node.get("session_id") != ctx.session_id:
                    continue

            integrated_at = node.get("integrated_at")
            if integrated_at is None:
                unprocessed.append(node)

        # Sort by event_time (oldest first for causal ordering)
        def get_event_time(n: Dict) -> str:
            return n.get("event_time", "") or ""

        unprocessed.sort(key=get_event_time)

        # Convert to MemoryHit format
        items = []
        for node in unprocessed[:limit]:
            content = node.get("content", "") or ""
            items.append(
                MemoryHit(
                    id=node.get("name", ""),
                    type=node.get("type", "unknown"),
                    title=node.get("title", ""),
                    snippet=content[:200] + "..." if len(content) > 200 else content,
                    event_time=node.get("event_time", ""),
                    score=0.0,
                )
            )

        return UnintegratedResult(
            items=items,
            count=len(items),
            total_unprocessed=len(unprocessed),
        )

    except Exception as e:
        logger.error(f"get_unintegrated failed: {e}")
        return UnintegratedResult()


async def commit_patch_handler(
    ctx: IntegrationContext,
    patch_json: str,
) -> CommitResult:
    """Validate and apply a GraphPatch to the memory graph.

    The LLM provides patch_json as a JSON string representing GraphPatch.
    This handler validates the patch, applies operations, and returns results.
    """
    import json

    try:
        # Parse and validate the patch
        if isinstance(patch_json, str):
            patch_data = json.loads(patch_json)
        elif isinstance(patch_json, dict):
            patch_data = patch_json
        else:
            raise TypeError(f"patch_json must be str or dict, got {type(patch_json).__name__}")
        patch = GraphPatch.model_validate(patch_data)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid patch JSON: {e}")
        return CommitResult(success=False, errors=[f"Invalid JSON: {e}"])
    except TypeError as e:
        logger.error(f"Invalid patch type: {e}")
        return CommitResult(success=False, errors=[str(e)])
    except Exception as e:
        logger.error(f"Patch validation failed: {e}")
        return CommitResult(success=False, errors=[f"Validation error: {e}"])

    if patch.dry_run:
        logger.info(f"Dry run: {len(patch.items)} operations validated")
        return CommitResult(success=True, applied=0, failed=0)

    # Apply each operation
    applied = 0
    failed = 0
    errors = []

    for item in patch.items:
        try:
            if item.operation == PatchOperation.LINK:
                await _apply_link(ctx, item)
                applied += 1

            elif item.operation == PatchOperation.UNLINK:
                await _apply_unlink(ctx, item)
                applied += 1

            elif item.operation == PatchOperation.MERGE:
                await _apply_merge(ctx, item)
                applied += 1

            elif item.operation == PatchOperation.FLAG:
                await _apply_flag(ctx, item)
                applied += 1

            elif item.operation == PatchOperation.MARK_INTEGRATED:
                await _apply_mark_integrated(ctx, item)
                applied += 1

            elif item.operation == PatchOperation.UPDATE_ATTR:
                await _apply_update_attr(ctx, item)
                applied += 1

            else:
                errors.append(f"Unknown operation: {item.operation}")
                failed += 1

        except Exception as e:
            logger.error(f"Failed to apply {item.operation} on {item.source_id}: {e}")
            errors.append(f"{item.operation} on {item.source_id}: {e}")
            failed += 1

    success = failed == 0
    logger.info(f"Patch applied: {applied} succeeded, {failed} failed")

    return CommitResult(
        success=success,
        applied=applied,
        failed=failed,
        errors=errors,
    )


# =============================================================================
# Patch Operation Implementations
# =============================================================================


async def _apply_link(ctx: IntegrationContext, item: GraphPatchItem) -> None:
    """Create a relationship between two memories."""
    if not item.target_id:
        raise ValueError("LINK requires target_id")
    if not item.relation_type:
        raise ValueError("LINK requires relation_type")

    link = MemoryLink(
        source_id=UUID(item.source_id),
        target_id=UUID(item.target_id),
        relation=item.relation_type,
        properties=item.properties,
    )
    await ctx.store.create_link(link, ctx.user_id)
    logger.debug(
        f"Created link: {item.source_id} -[{item.relation_type}]-> {item.target_id}"
    )


async def _apply_unlink(ctx: IntegrationContext, item: GraphPatchItem) -> None:
    """Remove a relationship between two memories."""
    if not item.target_id:
        raise ValueError("UNLINK requires target_id")
    if not item.relation_type:
        raise ValueError("UNLINK requires relation_type")

    # Use graph_db directly for deletion
    await ctx.graph_ops.graph_db.delete_relationship(
        source=item.source_id,
        target=item.target_id,
        relation=item.relation_type,
        user_id=ctx.user_id,
    )
    logger.debug(
        f"Removed link: {item.source_id} -[{item.relation_type}]-> {item.target_id}"
    )


async def _apply_merge(ctx: IntegrationContext, item: GraphPatchItem) -> None:
    """Merge two entities via SAME_AS relationship.

    Creates bidirectional SAME_AS links. Actual entity consolidation
    happens during query-time resolution.
    """
    if not item.target_id:
        raise ValueError("MERGE requires target_id")

    # Create bidirectional SAME_AS
    await ctx.store.create_link(
        MemoryLink(
            source_id=UUID(item.source_id),
            target_id=UUID(item.target_id),
            relation="SAME_AS",
            properties={
                "merged_at": datetime.utcnow().isoformat(),
                "reason": item.reason,
            },
        ),
        ctx.user_id,
    )
    await ctx.store.create_link(
        MemoryLink(
            source_id=UUID(item.target_id),
            target_id=UUID(item.source_id),
            relation="SAME_AS",
            properties={
                "merged_at": datetime.utcnow().isoformat(),
                "reason": item.reason,
            },
        ),
        ctx.user_id,
    )
    logger.debug(f"Merged entities: {item.source_id} <-> {item.target_id}")


async def _apply_flag(ctx: IntegrationContext, item: GraphPatchItem) -> None:
    """Flag a memory for review (contradictions, needs attention, etc.)."""
    flag_type = item.properties.get("flag_type", "needs_review")

    await ctx.graph_ops.graph_db.create_nodes(
        [
            {
                "name": item.source_id,
                "flagged": True,
                "flag_type": flag_type,
                "flag_reason": item.reason,
                "flagged_at": datetime.utcnow().isoformat(),
            }
        ],
        ctx.user_id,
    )
    logger.debug(f"Flagged memory {item.source_id}: {flag_type}")


async def _apply_mark_integrated(ctx: IntegrationContext, item: GraphPatchItem) -> None:
    """Mark a memory as integrated (processed by integration agent)."""
    await ctx.graph_ops.graph_db.create_nodes(
        [
            {
                "name": item.source_id,
                "integrated_at": datetime.utcnow().isoformat(),
                "integration_run": ctx.run_id,
            }
        ],
        ctx.user_id,
    )
    logger.debug(f"Marked integrated: {item.source_id}")


async def _apply_update_attr(ctx: IntegrationContext, item: GraphPatchItem) -> None:
    """Update an entity attribute."""
    key = item.properties.get("key")
    value = item.properties.get("value")

    if not key or value is None:
        raise ValueError("UPDATE_ATTR requires key and value in properties")

    await ctx.store.upsert_entity_attribute(
        entity_id=UUID(item.source_id),
        user_id=ctx.user_id,
        key=key,
        value=str(value),
        confidence=item.properties.get("confidence", 1.0),
    )
    logger.debug(f"Updated attribute {key}={value} on entity {item.source_id}")


# =============================================================================
# Wrapped Handlers for IntegrationContext
# =============================================================================

# Integration agent can reuse recall/expand, but needs context adaptation


async def integration_recall_handler(
    ctx: IntegrationContext,
    query: str,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    memory_types: Optional[List[str]] = None,
    limit: int = 10,
):
    """Wrapper for recall_handler using IntegrationContext."""

    # Create a minimal ToolContext-like object
    class _MinimalToolCtx:
        def __init__(self, user_id: str, graph_ops, store):
            self.user_id = user_id
            self.graph_ops = graph_ops
            self.store = store

    tool_ctx = _MinimalToolCtx(ctx.user_id, ctx.graph_ops, ctx.store)
    return await recall_handler(
        tool_ctx,  # type: ignore
        query=query,
        date_start=date_start,
        date_end=date_end,
        memory_types=memory_types,
        limit=limit,
    )


async def integration_expand_handler(
    ctx: IntegrationContext,
    memory_id: str,
    relationship_types: Optional[List[str]] = None,
    limit: int = 10,
):
    """Wrapper for expand_neighbors_handler using IntegrationContext."""

    class _MinimalToolCtx:
        def __init__(self, user_id: str, graph_ops, store):
            self.user_id = user_id
            self.graph_ops = graph_ops
            self.store = store

    tool_ctx = _MinimalToolCtx(ctx.user_id, ctx.graph_ops, ctx.store)
    return await expand_neighbors_handler(
        tool_ctx,  # type: ignore
        memory_id=memory_id,
        relationship_types=relationship_types,
        limit=limit,
    )


# =============================================================================
# Static Handler Registry
# =============================================================================

INTEGRATION_HANDLERS = {
    "recall": integration_recall_handler,
    "expand_neighbors": integration_expand_handler,
    "get_unintegrated": get_unintegrated_handler,
    "commit_patch": commit_patch_handler,
}
