"""
Tests for the PatchExecutor that applies GraphPatch operations to the memory store.

Tests cover:
- _apply_links creates correct MemoryLink objects
- _apply_merges creates SAME_AS links
- _apply_derived creates new memories with DERIVED_FROM
- _apply_conflicts creates CONTRADICTS links
- _mark_processed updates memories
- apply_patch handles partial failures gracefully
- apply_patch returns correct counts in result
"""

import pytest
from datetime import datetime
from uuid import uuid4, UUID
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any, Optional, Literal
import json
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from persona.models.memory import (
    Memory,
    MemoryLink,
    EpisodeMemory,
    PsycheMemory,
    NoteMemory,
    EntityMemory,
)


class LinkOp(BaseModel):
    source_id: UUID
    target_id: UUID
    relation: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class MergeOp(BaseModel):
    canonical_id: UUID
    duplicate_id: UUID
    reason: str = ""


class DerivedOp(BaseModel):
    content: str
    memory_type: Literal["episode", "psyche", "entity", "note"]
    source_ids: List[UUID]
    title: str = ""
    properties: Dict[str, Any] = Field(default_factory=dict)


class ConflictOp(BaseModel):
    memory_a: UUID
    memory_b: UUID
    reason: str
    resolution: Optional[Literal["keep_both", "supersede", "invalidate"]] = None
    auto_resolved: bool = False


class GraphPatch(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    links: List[LinkOp] = Field(default_factory=list)
    merges: List[MergeOp] = Field(default_factory=list)
    derived: List[DerivedOp] = Field(default_factory=list)
    conflicts: List[ConflictOp] = Field(default_factory=list)
    processed_ids: List[UUID] = Field(default_factory=list)


class GraphPatchResult(BaseModel):
    success: bool
    run_id: str
    user_id: str
    applied_at: datetime = Field(default_factory=datetime.utcnow)
    links_created: int = 0
    merges_applied: int = 0
    derived_created: int = 0
    conflicts_flagged: int = 0
    memories_marked_integrated: int = 0
    errors: List[str] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def total_operations(self) -> int:
        return (
            self.links_created
            + self.merges_applied
            + self.derived_created
            + self.conflicts_flagged
        )


class PatchExecutor:
    """Executes GraphPatch operations against the MemoryStore."""

    def __init__(self, store):
        self.store = store

    async def apply_patch(self, patch: GraphPatch) -> GraphPatchResult:
        result = GraphPatchResult(
            success=True, run_id=patch.run_id, user_id=patch.user_id
        )

        links_created, link_errors = await self._apply_links(patch.links, patch.user_id)
        result.links_created = links_created
        result.errors.extend(link_errors)

        merges_applied, merge_errors = await self._apply_merges(
            patch.merges, patch.user_id
        )
        result.merges_applied = merges_applied
        result.errors.extend(merge_errors)

        derived_created, derived_errors = await self._apply_derived(
            patch.derived, patch.user_id
        )
        result.derived_created = derived_created
        result.errors.extend(derived_errors)

        conflicts_marked, conflict_errors = await self._apply_conflicts(
            patch.conflicts, patch.user_id
        )
        result.conflicts_flagged = conflicts_marked
        result.errors.extend(conflict_errors)

        processed_count, process_errors = await self._mark_processed(
            patch.processed_ids, patch.user_id
        )
        result.memories_marked_integrated = processed_count
        result.errors.extend(process_errors)

        result.success = len(result.errors) == 0
        return result

    async def _apply_links(self, links: list, user_id: str) -> tuple:
        created = 0
        errors = []

        for link_op in links:
            try:
                memory_link = MemoryLink(
                    source_id=link_op.source_id,
                    target_id=link_op.target_id,
                    relation=link_op.relation,
                    properties={**link_op.properties, "confidence": link_op.confidence},
                )
                await self.store.create_link(memory_link, user_id)
                created += 1
            except Exception as e:
                errors.append(f"Link {link_op.source_id}->{link_op.target_id}: {e}")

        return created, errors

    async def _apply_merges(self, merges: list, user_id: str) -> tuple:
        applied = 0
        errors = []

        for merge_op in merges:
            try:
                memory_link = MemoryLink(
                    source_id=merge_op.duplicate_id,
                    target_id=merge_op.canonical_id,
                    relation="SAME_AS",
                    properties={"reason": merge_op.reason, "deprecated": True},
                )
                await self.store.create_link(memory_link, user_id)

                await self.store.update(
                    merge_op.duplicate_id,
                    user_id,
                    {
                        "status": "deprecated",
                        "canonical_id": str(merge_op.canonical_id),
                    },
                )
                applied += 1
            except Exception as e:
                errors.append(f"Merge {merge_op.duplicate_id}: {e}")

        return applied, errors

    async def _apply_derived(self, derived_ops: list, user_id: str) -> tuple:
        created = 0
        errors = []

        for derived_op in derived_ops:
            try:
                memory = self._create_memory_from_derived(derived_op, user_id)
                await self.store.create(memory)

                for source_id in derived_op.source_ids:
                    link = MemoryLink(
                        source_id=memory.id,
                        target_id=source_id,
                        relation="DERIVED_FROM",
                    )
                    await self.store.create_link(link, user_id)

                created += 1
            except Exception as e:
                errors.append(f"Derived '{derived_op.title}': {e}")

        return created, errors

    def _create_memory_from_derived(self, derived_op, user_id: str) -> Memory:
        # Base fields for all memory types (don't spread properties here for entity)
        base_fields = {
            "title": derived_op.title,
            "content": derived_op.content,
            "user_id": user_id,
            "source_type": "integration",
        }

        if derived_op.memory_type == "episode":
            return EpisodeMemory(**base_fields, **derived_op.properties)
        elif derived_op.memory_type == "psyche":
            return PsycheMemory(**base_fields, **derived_op.properties)
        elif derived_op.memory_type == "note":
            return NoteMemory(**base_fields, **derived_op.properties)
        elif derived_op.memory_type == "entity":
            # Entity has required fields that come from properties
            entity_type = derived_op.properties.get("entity_type", "concept")
            # Filter out entity_type from properties to avoid duplicate arg
            extra_props = {
                k: v for k, v in derived_op.properties.items() if k != "entity_type"
            }
            return EntityMemory(
                **base_fields,
                entity_type=entity_type,
                canonical_name=derived_op.title,
                **extra_props,
            )
        else:
            raise ValueError(f"Unknown memory type: {derived_op.memory_type}")

    async def _apply_conflicts(self, conflicts: list, user_id: str) -> tuple:
        marked = 0
        errors = []

        for conflict_op in conflicts:
            try:
                link_a = MemoryLink(
                    source_id=conflict_op.memory_a,
                    target_id=conflict_op.memory_b,
                    relation="CONTRADICTS",
                    properties={
                        "reason": conflict_op.reason,
                        "resolution": conflict_op.resolution,
                    },
                )
                await self.store.create_link(link_a, user_id)

                link_b = MemoryLink(
                    source_id=conflict_op.memory_b,
                    target_id=conflict_op.memory_a,
                    relation="CONTRADICTS",
                    properties={
                        "reason": conflict_op.reason,
                        "resolution": conflict_op.resolution,
                    },
                )
                await self.store.create_link(link_b, user_id)

                marked += 1
            except Exception as e:
                errors.append(
                    f"Conflict {conflict_op.memory_a}<->{conflict_op.memory_b}: {e}"
                )

        return marked, errors

    async def _mark_processed(self, memory_ids: list, user_id: str) -> tuple:
        processed = 0
        errors = []

        for memory_id in memory_ids:
            try:
                await self.store.update(
                    memory_id,
                    user_id,
                    {
                        "integration_processed": True,
                        "integration_at": datetime.utcnow().isoformat(),
                    },
                )
                processed += 1
            except Exception as e:
                errors.append(f"Mark processed {memory_id}: {e}")

        return processed, errors


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.create = AsyncMock(side_effect=lambda m: m)
    store.create_link = AsyncMock()
    store.update = AsyncMock(side_effect=lambda id, uid, updates: MagicMock(id=id))
    store.get = AsyncMock(return_value=None)
    return store


@pytest.fixture
def executor(mock_store):
    return PatchExecutor(store=mock_store)


class TestApplyLinks:
    @pytest.mark.asyncio
    async def test_apply_single_link(self, executor, mock_store):
        links = [
            LinkOp(
                source_id=uuid4(), target_id=uuid4(), relation="LED_TO", confidence=0.9
            )
        ]

        created, errors = await executor._apply_links(links, "user-123")

        assert created == 1
        assert errors == []
        mock_store.create_link.assert_called_once()

        call_args = mock_store.create_link.call_args
        memory_link = call_args[0][0]
        assert memory_link.source_id == links[0].source_id
        assert memory_link.target_id == links[0].target_id
        assert memory_link.relation == "LED_TO"
        assert memory_link.properties["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_apply_multiple_links(self, executor, mock_store):
        links = [
            LinkOp(source_id=uuid4(), target_id=uuid4(), relation="LED_TO"),
            LinkOp(source_id=uuid4(), target_id=uuid4(), relation="CAUSED_BY"),
            LinkOp(source_id=uuid4(), target_id=uuid4(), relation="MENTIONS"),
        ]

        created, errors = await executor._apply_links(links, "user-123")

        assert created == 3
        assert errors == []
        assert mock_store.create_link.call_count == 3

    @pytest.mark.asyncio
    async def test_apply_links_with_properties(self, executor, mock_store):
        links = [
            LinkOp(
                source_id=uuid4(),
                target_id=uuid4(),
                relation="RELATED_TO",
                properties={"weight": 0.8, "context": "work"},
            )
        ]

        created, errors = await executor._apply_links(links, "user-123")

        call_args = mock_store.create_link.call_args
        memory_link = call_args[0][0]
        assert memory_link.properties["weight"] == 0.8
        assert memory_link.properties["context"] == "work"

    @pytest.mark.asyncio
    async def test_apply_links_handles_failure(self, executor, mock_store):
        mock_store.create_link = AsyncMock(
            side_effect=[None, Exception("Node not found"), None]
        )

        links = [
            LinkOp(source_id=uuid4(), target_id=uuid4(), relation="A"),
            LinkOp(source_id=uuid4(), target_id=uuid4(), relation="B"),
            LinkOp(source_id=uuid4(), target_id=uuid4(), relation="C"),
        ]

        created, errors = await executor._apply_links(links, "user-123")

        assert created == 2
        assert len(errors) == 1
        assert "Node not found" in errors[0]

    @pytest.mark.asyncio
    async def test_apply_empty_links(self, executor, mock_store):
        created, errors = await executor._apply_links([], "user-123")

        assert created == 0
        assert errors == []
        mock_store.create_link.assert_not_called()


class TestApplyMerges:
    @pytest.mark.asyncio
    async def test_apply_single_merge(self, executor, mock_store):
        dup_id = uuid4()
        canon_id = uuid4()

        merges = [
            MergeOp(duplicate_id=dup_id, canonical_id=canon_id, reason="Same person")
        ]

        applied, errors = await executor._apply_merges(merges, "user-123")

        assert applied == 1
        assert errors == []

        mock_store.create_link.assert_called_once()
        call_args = mock_store.create_link.call_args
        memory_link = call_args[0][0]
        assert memory_link.relation == "SAME_AS"
        assert memory_link.properties["deprecated"] is True

        mock_store.update.assert_called_once()
        update_args = mock_store.update.call_args
        assert update_args[0][0] == dup_id
        assert update_args[0][2]["status"] == "deprecated"

    @pytest.mark.asyncio
    async def test_apply_multiple_merges(self, executor, mock_store):
        merges = [
            MergeOp(duplicate_id=uuid4(), canonical_id=uuid4()),
            MergeOp(duplicate_id=uuid4(), canonical_id=uuid4()),
        ]

        applied, errors = await executor._apply_merges(merges, "user-123")

        assert applied == 2
        assert errors == []

    @pytest.mark.asyncio
    async def test_apply_merge_handles_failure(self, executor, mock_store):
        mock_store.create_link = AsyncMock(side_effect=Exception("Connection error"))

        merges = [MergeOp(duplicate_id=uuid4(), canonical_id=uuid4())]

        applied, errors = await executor._apply_merges(merges, "user-123")

        assert applied == 0
        assert len(errors) == 1
        assert "Connection error" in errors[0]


class TestApplyDerived:
    @pytest.mark.asyncio
    async def test_create_derived_episode(self, executor, mock_store):
        source_ids = [uuid4(), uuid4()]

        derived = [
            DerivedOp(
                memory_type="episode",
                title="Weekly Summary",
                content="Summary of the week's events",
                source_ids=source_ids,
            )
        ]

        created, errors = await executor._apply_derived(derived, "user-123")

        assert created == 1
        assert errors == []

        mock_store.create.assert_called_once()
        memory = mock_store.create.call_args[0][0]
        assert memory.type == "episode"
        assert memory.title == "Weekly Summary"
        assert memory.source_type == "integration"

        assert mock_store.create_link.call_count == 2

    @pytest.mark.asyncio
    async def test_create_derived_psyche(self, executor, mock_store):
        derived = [
            DerivedOp(
                memory_type="psyche",
                title="preference",
                content="Prefers morning work",
                source_ids=[uuid4()],
            )
        ]

        created, errors = await executor._apply_derived(derived, "user-123")

        assert created == 1
        memory = mock_store.create.call_args[0][0]
        assert memory.type == "psyche"

    @pytest.mark.asyncio
    async def test_create_derived_note(self, executor, mock_store):
        derived = [
            DerivedOp(
                memory_type="note",
                title="Follow-up action",
                content="Review project status",
                source_ids=[uuid4()],
            )
        ]

        created, errors = await executor._apply_derived(derived, "user-123")

        assert created == 1
        memory = mock_store.create.call_args[0][0]
        assert memory.type == "note"

    @pytest.mark.asyncio
    async def test_create_derived_entity(self, executor, mock_store):
        derived = [
            DerivedOp(
                memory_type="entity",
                title="Project Alpha",
                content="Main project discussed",
                source_ids=[uuid4()],
                properties={"entity_type": "project"},
            )
        ]

        created, errors = await executor._apply_derived(derived, "user-123")

        assert created == 1
        memory = mock_store.create.call_args[0][0]
        assert memory.type == "entity"
        assert memory.canonical_name == "Project Alpha"

    @pytest.mark.asyncio
    async def test_derived_creates_correct_links(self, executor, mock_store):
        source_ids = [uuid4(), uuid4(), uuid4()]

        derived = [
            DerivedOp(
                memory_type="episode",
                title="Combined insight",
                content="...",
                source_ids=source_ids,
            )
        ]

        await executor._apply_derived(derived, "user-123")

        assert mock_store.create_link.call_count == 3

        for call in mock_store.create_link.call_args_list:
            link = call[0][0]
            assert link.relation == "DERIVED_FROM"

    @pytest.mark.asyncio
    async def test_derived_handles_failure(self, executor, mock_store):
        mock_store.create = AsyncMock(side_effect=Exception("Storage error"))

        derived = [
            DerivedOp(
                memory_type="episode", title="Test", content="...", source_ids=[uuid4()]
            )
        ]

        created, errors = await executor._apply_derived(derived, "user-123")

        assert created == 0
        assert len(errors) == 1
        assert "Storage error" in errors[0]


class TestApplyConflicts:
    @pytest.mark.asyncio
    async def test_mark_conflict(self, executor, mock_store):
        mem_a = uuid4()
        mem_b = uuid4()

        conflicts = [
            ConflictOp(
                memory_a=mem_a,
                memory_b=mem_b,
                reason="Different dates for same event",
                resolution="supersede",
            )
        ]

        marked, errors = await executor._apply_conflicts(conflicts, "user-123")

        assert marked == 1
        assert errors == []

        assert mock_store.create_link.call_count == 2

        calls = mock_store.create_link.call_args_list
        links = [call[0][0] for call in calls]

        assert all(l.relation == "CONTRADICTS" for l in links)
        assert any(l.source_id == mem_a and l.target_id == mem_b for l in links)
        assert any(l.source_id == mem_b and l.target_id == mem_a for l in links)

    @pytest.mark.asyncio
    async def test_conflict_includes_reason(self, executor, mock_store):
        conflicts = [
            ConflictOp(
                memory_a=uuid4(),
                memory_b=uuid4(),
                reason="Contradictory locations",
                resolution="keep_both",
            )
        ]

        await executor._apply_conflicts(conflicts, "user-123")

        link = mock_store.create_link.call_args_list[0][0][0]
        assert link.properties["reason"] == "Contradictory locations"
        assert link.properties["resolution"] == "keep_both"

    @pytest.mark.asyncio
    async def test_conflict_handles_failure(self, executor, mock_store):
        mock_store.create_link = AsyncMock(side_effect=Exception("Link error"))

        conflicts = [ConflictOp(memory_a=uuid4(), memory_b=uuid4(), reason="Test")]

        marked, errors = await executor._apply_conflicts(conflicts, "user-123")

        assert marked == 0
        assert len(errors) == 1


class TestMarkProcessed:
    @pytest.mark.asyncio
    async def test_mark_single_processed(self, executor, mock_store):
        memory_id = uuid4()

        processed, errors = await executor._mark_processed([memory_id], "user-123")

        assert processed == 1
        assert errors == []

        mock_store.update.assert_called_once()
        call_args = mock_store.update.call_args
        assert call_args[0][0] == memory_id
        assert call_args[0][2]["integration_processed"] is True

    @pytest.mark.asyncio
    async def test_mark_multiple_processed(self, executor, mock_store):
        memory_ids = [uuid4() for _ in range(5)]

        processed, errors = await executor._mark_processed(memory_ids, "user-123")

        assert processed == 5
        assert errors == []
        assert mock_store.update.call_count == 5

    @pytest.mark.asyncio
    async def test_mark_processed_handles_failure(self, executor, mock_store):
        mock_store.update = AsyncMock(side_effect=[None, Exception("Not found"), None])

        memory_ids = [uuid4(), uuid4(), uuid4()]

        processed, errors = await executor._mark_processed(memory_ids, "user-123")

        assert processed == 2
        assert len(errors) == 1


class TestApplyPatch:
    @pytest.mark.asyncio
    async def test_apply_empty_patch(self, executor, mock_store):
        patch = GraphPatch(run_id="run-001", user_id="user-123")

        result = await executor.apply_patch(patch)

        assert result.run_id == "run-001"
        assert result.user_id == "user-123"
        assert result.total_operations == 0
        assert result.has_errors is False

    @pytest.mark.asyncio
    async def test_apply_full_patch(self, executor, mock_store):
        patch = GraphPatch(
            run_id="run-002",
            user_id="user-456",
            links=[
                LinkOp(source_id=uuid4(), target_id=uuid4(), relation="LED_TO"),
                LinkOp(source_id=uuid4(), target_id=uuid4(), relation="MENTIONS"),
            ],
            merges=[
                MergeOp(duplicate_id=uuid4(), canonical_id=uuid4()),
            ],
            derived=[
                DerivedOp(
                    memory_type="episode",
                    title="Summary",
                    content="...",
                    source_ids=[uuid4()],
                ),
            ],
            conflicts=[
                ConflictOp(memory_a=uuid4(), memory_b=uuid4(), reason="Test"),
            ],
            processed_ids=[uuid4(), uuid4(), uuid4()],
        )

        result = await executor.apply_patch(patch)

        assert result.links_created == 2
        assert result.merges_applied == 1
        assert result.derived_created == 1
        assert result.conflicts_flagged == 1
        assert result.memories_marked_integrated == 3
        assert result.has_errors is False

    @pytest.mark.asyncio
    async def test_apply_patch_with_partial_failures(self, executor, mock_store):
        original_create_link = mock_store.create_link
        call_count = [0]

        async def failing_create_link(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise Exception("Link failed")
            return await original_create_link(*args, **kwargs)

        mock_store.create_link = AsyncMock(side_effect=failing_create_link)

        patch = GraphPatch(
            run_id="run-003",
            user_id="user-789",
            links=[
                LinkOp(source_id=uuid4(), target_id=uuid4(), relation="A"),
                LinkOp(source_id=uuid4(), target_id=uuid4(), relation="B"),
            ],
            processed_ids=[uuid4()],
        )

        result = await executor.apply_patch(patch)

        assert result.links_created == 0
        assert result.memories_marked_integrated == 1
        assert result.has_errors is True
        assert len(result.errors) == 2

    @pytest.mark.asyncio
    async def test_apply_patch_returns_correct_counts(self, executor, mock_store):
        patch = GraphPatch(
            run_id="run-004",
            user_id="user-count",
            links=[
                LinkOp(source_id=uuid4(), target_id=uuid4(), relation="X")
                for _ in range(5)
            ],
            merges=[
                MergeOp(duplicate_id=uuid4(), canonical_id=uuid4()) for _ in range(3)
            ],
            derived=[
                DerivedOp(
                    memory_type="episode", title="T", content="C", source_ids=[uuid4()]
                )
                for _ in range(2)
            ],
            conflicts=[
                ConflictOp(memory_a=uuid4(), memory_b=uuid4(), reason="R")
                for _ in range(4)
            ],
            processed_ids=[uuid4() for _ in range(10)],
        )

        result = await executor.apply_patch(patch)

        assert result.links_created == 5
        assert result.merges_applied == 3
        assert result.derived_created == 2
        assert result.conflicts_flagged == 4
        assert result.memories_marked_integrated == 10
        assert result.total_operations == 14

    @pytest.mark.asyncio
    async def test_apply_patch_timestamp(self, executor, mock_store):
        patch = GraphPatch(run_id="run-time", user_id="user-time")

        before = datetime.utcnow()
        result = await executor.apply_patch(patch)
        after = datetime.utcnow()

        assert before <= result.applied_at <= after
