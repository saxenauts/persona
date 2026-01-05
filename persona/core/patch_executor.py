"""
Patch Executor for Memory Integration.

Applies GraphPatch operations to Neo4j, handling links, merges,
derived memories, and conflict marking. Designed to be idempotent
and fault-tolerant (collects errors but continues execution).
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4
import json

from persona.models.graph_patch import (
    GraphPatch,
    GraphPatchResult,
    LinkOp,
    MergeOp,
    DerivedOp,
    ConflictOp,
    OperationResult,
)
from persona.core.memory_store import MemoryStore
from persona.models.memory import (
    MemoryLink,
    EpisodeMemory,
    PsycheMemory,
    EntityMemory,
    NoteMemory,
)
from server.logging_config import get_logger

logger = get_logger(__name__)


class PatchExecutor:
    """
    Executes GraphPatch operations against the memory store.

    Operations are applied in order: links -> merges -> derived -> conflicts.
    Each operation is independent; failures are logged but don't stop execution.
    """

    def __init__(self, store: MemoryStore):
        """
        Initialize PatchExecutor with a MemoryStore.

        Args:
            store: MemoryStore instance for database operations.
        """
        self.store = store

    async def apply_patch(self, patch: GraphPatch, user_id: str) -> GraphPatchResult:
        """
        Apply all operations in a GraphPatch to the memory graph.

        Args:
            patch: The GraphPatch containing operations to apply.
            user_id: User ID for the operations.

        Returns:
            GraphPatchResult with counts and any errors encountered.
        """
        result = GraphPatchResult(
            success=True,  # Will be set to False if any errors
            run_id=patch.run_id,
            user_id=user_id,
        )

        logger.info(
            f"Applying patch {patch.run_id}: "
            f"{len(patch.links)} links, {len(patch.merges)} merges, "
            f"{len(patch.derived)} derived, {len(patch.conflicts)} conflicts"
        )

        # Apply operations in sequence
        await self._apply_links(patch.links, user_id, result)
        await self._apply_merges(patch.merges, user_id, result)
        await self._apply_derived(patch.derived, user_id, result)
        await self._apply_conflicts(patch.conflicts, user_id, result)

        # Mark processed memories
        if patch.processed_ids:
            await self._mark_processed(
                patch.processed_ids, patch.run_id, user_id, result
            )

        result.success = len(result.errors) == 0

        logger.info(
            f"Patch {patch.run_id} complete: "
            f"{result.total_operations} operations, {len(result.errors)} errors"
        )

        return result

    async def _apply_links(
        self, links: List[LinkOp], user_id: str, result: GraphPatchResult
    ) -> None:
        """
        Create MemoryLink for each LinkOp.

        Handles duplicates gracefully (idempotent).
        """
        for link_op in links:
            try:
                memory_link = MemoryLink(
                    source_id=link_op.source_id,
                    target_id=link_op.target_id,
                    relation=link_op.relation,
                    properties={
                        **link_op.properties,
                        "confidence": link_op.confidence,
                        "created_by_integration": True,
                    },
                )

                await self.store.create_link(memory_link, user_id)

                result.links_created += 1
                result.operation_results.append(
                    OperationResult(
                        operation_type="link",
                        success=True,
                        details=f"{link_op.source_id} --{link_op.relation}--> {link_op.target_id}",
                    )
                )
                logger.debug(
                    f"Created link: {link_op.source_id} --{link_op.relation}--> {link_op.target_id}"
                )

            except Exception as e:
                error_msg = f"Failed to create link {link_op.source_id} -> {link_op.target_id}: {e}"
                result.errors.append(error_msg)
                result.operation_results.append(
                    OperationResult(
                        operation_type="link",
                        success=False,
                        error=str(e),
                    )
                )
                logger.warning(error_msg)

    async def _apply_merges(
        self, merges: List[MergeOp], user_id: str, result: GraphPatchResult
    ) -> None:
        """
        Merge duplicate entities into canonical ones.

        For each merge:
        1. Add SAME_AS link from duplicate to canonical
        2. Update duplicate's aliases to include canonical reference
        3. Update any MENTIONS links to point to canonical
        """
        for merge_op in merges:
            try:
                # Get both entities
                duplicate = await self.store.get(merge_op.duplicate_id, user_id)
                canonical = await self.store.get(merge_op.canonical_id, user_id)

                if not duplicate:
                    raise ValueError(
                        f"Duplicate entity {merge_op.duplicate_id} not found"
                    )
                if not canonical:
                    raise ValueError(
                        f"Canonical entity {merge_op.canonical_id} not found"
                    )

                # 1. Create SAME_AS link from duplicate to canonical
                same_as_link = MemoryLink(
                    source_id=merge_op.duplicate_id,
                    target_id=merge_op.canonical_id,
                    relation="SAME_AS",
                    properties={
                        "reason": merge_op.reason,
                        "merged_at": datetime.utcnow().isoformat(),
                    },
                )
                await self.store.create_link(same_as_link, user_id)

                # 2. Update duplicate's aliases to reference canonical
                if hasattr(duplicate, "aliases") and hasattr(
                    canonical, "canonical_name"
                ):
                    duplicate_aliases = getattr(duplicate, "aliases", []) or []
                    canonical_name = getattr(canonical, "canonical_name", "")

                    if canonical_name and canonical_name not in duplicate_aliases:
                        updated_aliases = list(duplicate_aliases) + [canonical_name]

                        # Update via graph_db directly for aliases
                        await self.store.graph_db.create_nodes(
                            [
                                {
                                    "name": str(merge_op.duplicate_id),
                                    "type": "entity",
                                    "aliases": json.dumps(updated_aliases),
                                    "merged_into": str(merge_op.canonical_id),
                                    "is_deprecated": True,
                                }
                            ],
                            user_id,
                        )

                # 3. Redirect MENTIONS relationships (complex - get all relationships)
                await self._redirect_mentions(
                    merge_op.duplicate_id, merge_op.canonical_id, user_id
                )

                result.merges_applied += 1
                result.operation_results.append(
                    OperationResult(
                        operation_type="merge",
                        success=True,
                        details=f"Merged {merge_op.duplicate_id} -> {merge_op.canonical_id}",
                    )
                )
                logger.info(
                    f"Merged entity {merge_op.duplicate_id} into {merge_op.canonical_id}"
                )

            except Exception as e:
                error_msg = f"Failed to merge {merge_op.duplicate_id} -> {merge_op.canonical_id}: {e}"
                result.errors.append(error_msg)
                result.operation_results.append(
                    OperationResult(
                        operation_type="merge",
                        success=False,
                        error=str(e),
                    )
                )
                logger.warning(error_msg)

    async def _redirect_mentions(
        self, from_id: UUID, to_id: UUID, user_id: str
    ) -> None:
        """
        Redirect MENTIONS relationships from one entity to another.

        This creates new links to the canonical entity while preserving
        the original links (for audit trail).
        """
        try:
            # Get relationships pointing to the duplicate
            relationships = await self.store.graph_db.get_node_relationships(
                str(from_id), user_id
            )

            for rel in relationships:
                if rel.get("relation") == "MENTIONS":
                    source = rel.get("source")
                    # If this memory mentions the duplicate, also link to canonical
                    if source and source != str(from_id):
                        redirect_link = MemoryLink(
                            source_id=UUID(source),
                            target_id=to_id,
                            relation="MENTIONS",
                            properties={
                                "redirected_from": str(from_id),
                                "redirected_at": datetime.utcnow().isoformat(),
                            },
                        )
                        await self.store.create_link(redirect_link, user_id)
                        logger.debug(f"Redirected MENTIONS from {source} to {to_id}")

        except Exception as e:
            logger.warning(f"Failed to redirect mentions for {from_id}: {e}")
            # Don't raise - this is a secondary operation

    async def _apply_derived(
        self, derived: List[DerivedOp], user_id: str, result: GraphPatchResult
    ) -> None:
        """
        Create new memories derived from existing ones.

        For each derived memory:
        1. Create the new memory with source_type = "integration"
        2. Add DERIVED_FROM links to all source memories
        """
        for derived_op in derived:
            try:
                # Create the appropriate memory type
                memory_id = uuid4()
                base_fields = {
                    "id": memory_id,
                    "user_id": user_id,
                    "title": derived_op.title,
                    "content": derived_op.content,
                    "source_type": "integration",
                    "properties": {
                        **derived_op.properties,
                        "derived_from_count": len(derived_op.source_ids),
                    },
                }

                if derived_op.memory_type == "episode":
                    memory = EpisodeMemory(**base_fields)
                elif derived_op.memory_type == "psyche":
                    memory = PsycheMemory(**base_fields)
                elif derived_op.memory_type == "note":
                    memory = NoteMemory(**base_fields)
                elif derived_op.memory_type == "entity":
                    # Entity requires additional fields
                    memory = EntityMemory(
                        **base_fields,
                        entity_type=derived_op.properties.get("entity_type", "concept"),
                        canonical_name=derived_op.title,
                    )
                else:
                    raise ValueError(f"Unknown memory type: {derived_op.memory_type}")

                # Create the memory
                await self.store.create(memory)

                # Create DERIVED_FROM links to all sources
                for source_id in derived_op.source_ids:
                    derived_link = MemoryLink(
                        source_id=memory_id,
                        target_id=source_id,
                        relation="DERIVED_FROM",
                        properties={
                            "derived_at": datetime.utcnow().isoformat(),
                        },
                    )
                    await self.store.create_link(derived_link, user_id)

                result.derived_created += 1
                result.operation_results.append(
                    OperationResult(
                        operation_type="derived",
                        success=True,
                        details=f"Created {derived_op.memory_type} '{derived_op.title}' from {len(derived_op.source_ids)} sources",
                    )
                )
                logger.info(
                    f"Created derived {derived_op.memory_type} '{derived_op.title}' "
                    f"from {len(derived_op.source_ids)} sources"
                )

            except Exception as e:
                error_msg = f"Failed to create derived memory '{derived_op.title}': {e}"
                result.errors.append(error_msg)
                result.operation_results.append(
                    OperationResult(
                        operation_type="derived",
                        success=False,
                        error=str(e),
                    )
                )
                logger.warning(error_msg)

    async def _apply_conflicts(
        self, conflicts: List[ConflictOp], user_id: str, result: GraphPatchResult
    ) -> None:
        """
        Mark conflicting memories with CONTRADICTS relationship.

        For each conflict:
        1. Create CONTRADICTS link between memory_a and memory_b
        2. Store reason in link properties
        3. Optionally mark memories with "contested" in properties
        """
        for conflict_op in conflicts:
            try:
                # Create bidirectional CONTRADICTS relationship
                # (A contradicts B and B contradicts A)
                conflict_props = {
                    "reason": conflict_op.reason,
                    "resolution": conflict_op.resolution,
                    "marked_at": datetime.utcnow().isoformat(),
                }

                # A -> B
                conflict_link_ab = MemoryLink(
                    source_id=conflict_op.memory_a,
                    target_id=conflict_op.memory_b,
                    relation="CONTRADICTS",
                    properties=conflict_props,
                )
                await self.store.create_link(conflict_link_ab, user_id)

                # B -> A (bidirectional for easy querying)
                conflict_link_ba = MemoryLink(
                    source_id=conflict_op.memory_b,
                    target_id=conflict_op.memory_a,
                    relation="CONTRADICTS",
                    properties=conflict_props,
                )
                await self.store.create_link(conflict_link_ba, user_id)

                # Mark both memories as contested
                for memory_id in [conflict_op.memory_a, conflict_op.memory_b]:
                    memory = await self.store.get(memory_id, user_id)
                    if memory:
                        await self.store.graph_db.create_nodes(
                            [
                                {
                                    "name": str(memory_id),
                                    "type": memory.type,
                                    "contested": True,
                                    "contested_reason": conflict_op.reason,
                                }
                            ],
                            user_id,
                        )

                result.conflicts_flagged += 1
                result.operation_results.append(
                    OperationResult(
                        operation_type="conflict",
                        success=True,
                        details=f"Marked conflict: {conflict_op.memory_a} <-> {conflict_op.memory_b}",
                    )
                )
                logger.info(
                    f"Marked conflict between {conflict_op.memory_a} and {conflict_op.memory_b}: "
                    f"{conflict_op.reason}"
                )

            except Exception as e:
                error_msg = f"Failed to mark conflict {conflict_op.memory_a} <-> {conflict_op.memory_b}: {e}"
                result.errors.append(error_msg)
                result.operation_results.append(
                    OperationResult(
                        operation_type="conflict",
                        success=False,
                        error=str(e),
                    )
                )
                logger.warning(error_msg)

    async def _mark_processed(
        self,
        memory_ids: List[UUID],
        run_id: str,
        user_id: str,
        result: GraphPatchResult,
    ) -> None:
        """
        Mark memories as processed by this integration run.

        Updates each memory with:
        - integrated_at: timestamp
        - integrated_by: run_id
        """
        processed_count = 0

        for memory_id in memory_ids:
            try:
                memory = await self.store.get(memory_id, user_id)
                if not memory:
                    logger.debug(f"Memory {memory_id} not found for marking processed")
                    continue

                # Update via graph_db directly
                await self.store.graph_db.create_nodes(
                    [
                        {
                            "name": str(memory_id),
                            "type": memory.type,
                            "integrated_at": datetime.utcnow().isoformat(),
                            "integrated_by": run_id,
                        }
                    ],
                    user_id,
                )
                processed_count += 1

            except Exception as e:
                logger.warning(f"Failed to mark memory {memory_id} as processed: {e}")
                # Don't add to errors - this is metadata, not critical

        result.memories_marked_integrated = processed_count
        logger.debug(
            f"Marked {processed_count}/{len(memory_ids)} memories as processed"
        )
