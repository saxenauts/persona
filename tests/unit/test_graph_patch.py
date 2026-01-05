"""
Tests for the GraphPatch schema and related models.

Tests cover:
- Operation models (LinkOp, MergeOp, DerivedOp, ConflictOp)
- GraphPatch batch operations
- Serialization/deserialization
- IntegrationJob status transitions
- GraphPatchResult aggregation
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4, UUID
import json

from typing import Any

from persona.models.graph_patch import (
    LinkOp,
    MergeOp,
    DerivedOp,
    ConflictOp,
    GraphPatch,
    GraphPatchResult,
    OperationResult,
)


# ============================================================================
# LinkOp Tests
# ============================================================================


class TestLinkOp:
    """Tests for LinkOp model creation and validation."""

    def test_create_basic_link(self):
        """Test creating a basic link between memories."""
        source = uuid4()
        target = uuid4()

        link = LinkOp(source_id=source, target_id=target, relation="LED_TO")

        assert link.source_id == source
        assert link.target_id == target
        assert link.relation == "LED_TO"
        assert link.confidence == 1.0
        assert link.properties == {}

    def test_link_with_properties(self):
        """Test creating a link with custom properties."""
        link = LinkOp(
            source_id=uuid4(),
            target_id=uuid4(),
            relation="CAUSED_BY",
            properties={"weight": 0.8, "context": "work"},
        )

        assert link.properties["weight"] == 0.8
        assert link.properties["context"] == "work"

    def test_link_with_confidence(self):
        """Test creating a link with custom confidence."""
        link = LinkOp(
            source_id=uuid4(), target_id=uuid4(), relation="MENTIONS", confidence=0.75
        )

        assert link.confidence == 0.75

    def test_link_confidence_bounds(self):
        """Test that confidence is bounded 0-1."""
        # Valid bounds
        link_min = LinkOp(
            source_id=uuid4(), target_id=uuid4(), relation="X", confidence=0.0
        )
        link_max = LinkOp(
            source_id=uuid4(), target_id=uuid4(), relation="X", confidence=1.0
        )

        assert link_min.confidence == 0.0
        assert link_max.confidence == 1.0

        # Invalid bounds should raise validation error
        with pytest.raises(ValueError):
            LinkOp(source_id=uuid4(), target_id=uuid4(), relation="X", confidence=-0.1)

        with pytest.raises(ValueError):
            LinkOp(source_id=uuid4(), target_id=uuid4(), relation="X", confidence=1.1)

    def test_link_serialization(self):
        """Test LinkOp serialization roundtrip."""
        original = LinkOp(
            source_id=uuid4(),
            target_id=uuid4(),
            relation="DERIVED_FROM",
            properties={"key": "value"},
            confidence=0.9,
        )

        data = original.model_dump()
        restored = LinkOp.model_validate(data)

        assert restored.source_id == original.source_id
        assert restored.target_id == original.target_id
        assert restored.relation == original.relation
        assert restored.properties == original.properties
        assert restored.confidence == original.confidence


# ============================================================================
# MergeOp Tests
# ============================================================================


class TestMergeOp:
    """Tests for MergeOp model creation and validation."""

    def test_create_merge(self):
        """Test creating a merge operation."""
        dup = uuid4()
        canonical = uuid4()

        merge = MergeOp(
            duplicate_id=dup,
            canonical_id=canonical,
            reason="Same entity with different names",
        )

        assert merge.duplicate_id == dup
        assert merge.canonical_id == canonical
        assert merge.reason == "Same entity with different names"

    def test_merge_default_reason(self):
        """Test merge with default empty reason."""
        merge = MergeOp(duplicate_id=uuid4(), canonical_id=uuid4())

        assert merge.reason == ""

    def test_merge_serialization(self):
        """Test MergeOp serialization roundtrip."""
        original = MergeOp(
            duplicate_id=uuid4(), canonical_id=uuid4(), reason="Duplicate entity"
        )

        data = original.model_dump()
        restored = MergeOp.model_validate(data)

        assert restored.duplicate_id == original.duplicate_id
        assert restored.canonical_id == original.canonical_id
        assert restored.reason == original.reason

    def test_merge_with_uuid_strings(self):
        """Test creating merge with UUID strings (for API interop)."""
        dup_str = str(uuid4())
        canon_str = str(uuid4())

        merge = MergeOp(
            duplicate_id=dup_str,  # type: ignore - testing string input
            canonical_id=canon_str,  # type: ignore
        )

        assert isinstance(merge.duplicate_id, UUID)
        assert isinstance(merge.canonical_id, UUID)


# ============================================================================
# DerivedOp Tests
# ============================================================================


class TestDerivedOp:
    """Tests for DerivedOp model creation and validation."""

    def test_create_derived_episode(self):
        """Test creating a derived episode memory."""
        sources = [uuid4(), uuid4(), uuid4()]

        derived = DerivedOp(
            memory_type="episode",
            title="Weekly Summary",
            content="This week focused on project deadlines.",
            source_ids=sources,
        )

        assert derived.memory_type == "episode"
        assert derived.title == "Weekly Summary"
        assert derived.content == "This week focused on project deadlines."
        assert len(derived.source_ids) == 3
        assert derived.properties == {}

    def test_create_derived_psyche(self):
        """Test creating a derived psyche (trait/preference)."""
        derived = DerivedOp(
            memory_type="psyche",
            title="preference",
            content="Prefers working in the morning",
            source_ids=[uuid4()],
        )

        assert derived.memory_type == "psyche"

    def test_create_derived_note(self):
        """Test creating a derived note."""
        derived = DerivedOp(
            memory_type="note",
            title="Action item from meeting",
            content="Follow up with team on deliverables",
            source_ids=[uuid4(), uuid4()],
        )

        assert derived.memory_type == "note"

    def test_create_derived_entity(self):
        """Test creating a derived entity."""
        derived = DerivedOp(
            memory_type="entity",
            title="Project Alpha",
            content="Main project mentioned across conversations",
            source_ids=[uuid4()],
            properties={"entity_type": "project", "canonical_name": "Project Alpha"},
        )

        assert derived.memory_type == "entity"
        assert derived.properties["entity_type"] == "project"

    def test_derived_with_properties(self):
        """Test derived op with additional properties."""
        derived = DerivedOp(
            memory_type="episode",
            title="Pattern",
            content="Weekly exercise pattern",
            source_ids=[uuid4()],
            properties={"pattern_type": "behavioral", "confidence": 0.85},
        )

        assert derived.properties["pattern_type"] == "behavioral"

    def test_derived_invalid_memory_type(self):
        """Test that invalid memory types are rejected."""
        with pytest.raises(ValueError):
            DerivedOp(
                memory_type="invalid",  # type: ignore
                title="Test",
                content="Test",
                source_ids=[uuid4()],
            )

    def test_derived_serialization(self):
        """Test DerivedOp serialization roundtrip."""
        original = DerivedOp(
            memory_type="psyche",
            title="Insight",
            content="Values authenticity",
            source_ids=[uuid4(), uuid4()],
            properties={"derived": True},
        )

        data = original.model_dump()
        restored = DerivedOp.model_validate(data)

        assert restored.memory_type == original.memory_type
        assert restored.title == original.title
        assert restored.content == original.content
        assert restored.source_ids == original.source_ids
        assert restored.properties == original.properties


# ============================================================================
# ConflictOp Tests
# ============================================================================


class TestConflictOp:
    """Tests for ConflictOp model creation and validation."""

    def test_create_conflict(self):
        """Test creating a conflict marker."""
        mem_a = uuid4()
        mem_b = uuid4()

        conflict = ConflictOp(
            memory_a=mem_a, memory_b=mem_b, reason="Contradictory dates for same event"
        )

        assert conflict.memory_a == mem_a
        assert conflict.memory_b == mem_b
        assert conflict.reason == "Contradictory dates for same event"
        assert conflict.resolution is None

    def test_conflict_with_resolution(self):
        """Test conflict with provided resolution."""
        conflict = ConflictOp(
            memory_a=uuid4(),
            memory_b=uuid4(),
            reason="Different job titles",
            resolution="supersede",
        )

        assert conflict.resolution == "supersede"

    def test_conflict_resolution_literals(self):
        """Test all valid resolution literal values."""
        for resolution in ["keep_both", "supersede", "invalidate"]:
            conflict = ConflictOp(
                memory_a=uuid4(),
                memory_b=uuid4(),
                reason="Test",
                resolution=resolution,  # type: ignore
            )
            assert conflict.resolution == resolution

    def test_conflict_serialization(self):
        """Test ConflictOp serialization roundtrip."""
        original = ConflictOp(
            memory_a=uuid4(),
            memory_b=uuid4(),
            reason="Conflicting locations",
            resolution="keep_both",
        )

        data = original.model_dump()
        restored = ConflictOp.model_validate(data)

        assert restored.memory_a == original.memory_a
        assert restored.memory_b == original.memory_b
        assert restored.reason == original.reason
        assert restored.resolution == original.resolution


# ============================================================================
# GraphPatch Tests
# ============================================================================


class TestGraphPatch:
    """Tests for GraphPatch batch model."""

    def test_create_empty_patch(self):
        """Test creating an empty patch."""
        patch = GraphPatch(run_id="run-001", user_id="user-123")

        assert patch.run_id == "run-001"
        assert patch.user_id == "user-123"
        assert patch.links == []
        assert patch.merges == []
        assert patch.derived == []
        assert patch.conflicts == []
        assert patch.processed_ids == []
        assert isinstance(patch.created_at, datetime)

    def test_create_patch_with_all_operations(self):
        """Test creating a patch with all operation types."""
        link = LinkOp(source_id=uuid4(), target_id=uuid4(), relation="LED_TO")
        merge = MergeOp(duplicate_id=uuid4(), canonical_id=uuid4())
        derived = DerivedOp(
            memory_type="episode", title="Summary", content="...", source_ids=[uuid4()]
        )
        conflict = ConflictOp(
            memory_a=uuid4(), memory_b=uuid4(), reason="Contradiction"
        )

        patch = GraphPatch(
            run_id="run-002",
            user_id="user-456",
            links=[link],
            merges=[merge],
            derived=[derived],
            conflicts=[conflict],
            processed_ids=[uuid4(), uuid4()],
        )

        assert len(patch.links) == 1
        assert len(patch.merges) == 1
        assert len(patch.derived) == 1
        assert len(patch.conflicts) == 1
        assert len(patch.processed_ids) == 2

    def test_patch_with_multiple_links(self):
        """Test patch with multiple links."""
        links = [
            LinkOp(source_id=uuid4(), target_id=uuid4(), relation="LED_TO"),
            LinkOp(source_id=uuid4(), target_id=uuid4(), relation="CAUSED_BY"),
            LinkOp(source_id=uuid4(), target_id=uuid4(), relation="MENTIONS"),
        ]

        patch = GraphPatch(run_id="run-003", user_id="user-789", links=links)

        assert len(patch.links) == 3

    def test_patch_serialization(self):
        """Test GraphPatch serialization roundtrip."""
        original = GraphPatch(
            run_id="run-serial",
            user_id="user-serial",
            links=[LinkOp(source_id=uuid4(), target_id=uuid4(), relation="X")],
            merges=[MergeOp(duplicate_id=uuid4(), canonical_id=uuid4())],
            derived=[
                DerivedOp(
                    memory_type="psyche", title="T", content="C", source_ids=[uuid4()]
                )
            ],
            conflicts=[ConflictOp(memory_a=uuid4(), memory_b=uuid4(), reason="R")],
            processed_ids=[uuid4()],
        )

        data = original.model_dump()
        restored = GraphPatch.model_validate(data)

        assert restored.run_id == original.run_id
        assert restored.user_id == original.user_id
        assert len(restored.links) == 1
        assert len(restored.merges) == 1
        assert len(restored.derived) == 1
        assert len(restored.conflicts) == 1
        assert len(restored.processed_ids) == 1

    def test_patch_json_serialization(self):
        """Test GraphPatch JSON serialization."""
        patch = GraphPatch(
            run_id="run-json",
            user_id="user-json",
            links=[LinkOp(source_id=uuid4(), target_id=uuid4(), relation="Y")],
        )

        json_str = patch.model_dump_json()
        data = json.loads(json_str)
        restored = GraphPatch.model_validate(data)

        assert restored.run_id == patch.run_id
        assert len(restored.links) == 1


# ============================================================================
# GraphPatchResult Tests
# ============================================================================


class TestGraphPatchResult:
    """Tests for GraphPatchResult aggregation and properties."""

    def test_create_empty_result(self):
        """Test creating an empty result."""
        result = GraphPatchResult(success=True, run_id="run-001", user_id="user-123")

        assert result.run_id == "run-001"
        assert result.user_id == "user-123"
        assert result.links_created == 0
        assert result.merges_applied == 0
        assert result.derived_created == 0
        assert result.conflicts_flagged == 0
        assert result.memories_marked_integrated == 0
        assert result.errors == []
        assert result.operation_results == []

    def test_result_with_counts(self):
        """Test result with operation counts."""
        result = GraphPatchResult(
            success=True,
            run_id="run-002",
            user_id="user-456",
            links_created=5,
            merges_applied=2,
            derived_created=1,
            conflicts_flagged=1,
            memories_marked_integrated=10,
        )

        assert result.links_created == 5
        assert result.merges_applied == 2
        assert result.derived_created == 1
        assert result.conflicts_flagged == 1
        assert result.memories_marked_integrated == 10

    def test_result_total_operations(self):
        """Test total_operations property calculation."""
        result = GraphPatchResult(
            success=True,
            run_id="run-003",
            user_id="user-789",
            links_created=3,
            merges_applied=2,
            derived_created=1,
            conflicts_flagged=1,
        )

        assert result.total_operations == 7

    def test_result_has_errors_false(self):
        """Test has_errors is False when no errors."""
        result = GraphPatchResult(success=True, run_id="r", user_id="u")

        assert result.has_errors is False

    def test_result_has_errors_true(self):
        """Test has_errors is True when errors exist."""
        result = GraphPatchResult(
            success=False,
            run_id="r",
            user_id="u",
            errors=["Failed to create link: node not found"],
        )

        assert result.has_errors is True

    def test_result_with_operation_results(self):
        """Test result with detailed operation results."""
        ops = [
            OperationResult(
                operation_type="link", success=True, details="Created LED_TO"
            ),
            OperationResult(
                operation_type="merge", success=False, error="Node not found"
            ),
            OperationResult(
                operation_type="derived", success=True, details="Created insight"
            ),
        ]

        result = GraphPatchResult(
            success=True,
            run_id="run-ops",
            user_id="user-ops",
            links_created=1,
            derived_created=1,
            operation_results=ops,
        )

        assert len(result.operation_results) == 3
        assert result.operation_results[0].success is True
        assert result.operation_results[1].success is False
        assert result.operation_results[1].error == "Node not found"

    def test_result_serialization(self):
        """Test GraphPatchResult serialization roundtrip."""
        original = GraphPatchResult(
            success=True,
            run_id="run-serial",
            user_id="user-serial",
            links_created=2,
            errors=["error1"],
            operation_results=[OperationResult(operation_type="link", success=True)],
        )

        data = original.model_dump()
        restored = GraphPatchResult.model_validate(data)

        assert restored.run_id == original.run_id
        assert restored.links_created == 2
        assert len(restored.errors) == 1
        assert len(restored.operation_results) == 1


# ============================================================================
# OperationResult Tests
# ============================================================================


class TestOperationResult:
    """Tests for individual OperationResult model."""

    def test_create_success_result(self):
        """Test creating a success result."""
        result = OperationResult(
            operation_type="link", success=True, details="Created LED_TO relationship"
        )

        assert result.operation_type == "link"
        assert result.success is True
        assert result.details == "Created LED_TO relationship"
        assert result.error is None

    def test_create_failure_result(self):
        """Test creating a failure result."""
        result = OperationResult(
            operation_type="merge",
            success=False,
            error="Duplicate node not found in graph",
        )

        assert result.operation_type == "merge"
        assert result.success is False
        assert result.error == "Duplicate node not found in graph"

    def test_result_all_operation_types(self):
        """Test result for all operation types."""
        types = ["link", "merge", "derived", "conflict", "mark_processed"]

        for op_type in types:
            result = OperationResult(operation_type=op_type, success=True)
            assert result.operation_type == op_type


# ============================================================================
# IntegrationJob Tests (Status Transitions)
# ============================================================================


class IntegrationJob:
    """Model for tracking integration job status (to be implemented).

    This is a placeholder for the IntegrationJob model that will track
    background integration runs. Tests define expected behavior.
    """

    VALID_STATUSES = ["pending", "running", "completed", "failed", "cancelled"]
    VALID_TRANSITIONS = {
        "pending": ["running", "cancelled"],
        "running": ["completed", "failed", "cancelled"],
        "completed": [],
        "failed": [],
        "cancelled": [],
    }

    def __init__(self, job_id: str, user_id: str, status: str = "pending"):
        self.job_id = job_id
        self.user_id = user_id
        self.status = status
        self.created_at = datetime.utcnow()
        self.started_at = None
        self.completed_at = None
        self.result: Any = None
        self.error = None
        self.checkpoint = {}

    def can_transition_to(self, new_status: str) -> bool:
        return new_status in self.VALID_TRANSITIONS.get(self.status, [])

    def transition(self, new_status: str) -> bool:
        if not self.can_transition_to(new_status):
            return False

        if new_status == "running":
            self.started_at = datetime.utcnow()
        elif new_status in ["completed", "failed"]:
            self.completed_at = datetime.utcnow()

        self.status = new_status
        return True


class TestIntegrationJob:
    """Tests for IntegrationJob status transitions."""

    def test_create_job(self):
        """Test creating a new integration job."""
        job = IntegrationJob(job_id="job-001", user_id="user-123")

        assert job.job_id == "job-001"
        assert job.user_id == "user-123"
        assert job.status == "pending"
        assert job.created_at is not None

    def test_valid_transition_pending_to_running(self):
        """Test valid transition: pending -> running."""
        job = IntegrationJob(job_id="job-002", user_id="user-123")

        assert job.can_transition_to("running") is True
        assert job.transition("running") is True
        assert job.status == "running"
        assert job.started_at is not None

    def test_valid_transition_running_to_completed(self):
        """Test valid transition: running -> completed."""
        job = IntegrationJob(job_id="job-003", user_id="user-123", status="running")

        assert job.can_transition_to("completed") is True
        assert job.transition("completed") is True
        assert job.status == "completed"
        assert job.completed_at is not None

    def test_valid_transition_running_to_failed(self):
        """Test valid transition: running -> failed."""
        job = IntegrationJob(job_id="job-004", user_id="user-123", status="running")

        assert job.can_transition_to("failed") is True
        assert job.transition("failed") is True
        assert job.status == "failed"

    def test_valid_transition_pending_to_cancelled(self):
        """Test valid transition: pending -> cancelled."""
        job = IntegrationJob(job_id="job-005", user_id="user-123")

        assert job.can_transition_to("cancelled") is True
        assert job.transition("cancelled") is True
        assert job.status == "cancelled"

    def test_invalid_transition_pending_to_completed(self):
        """Test invalid transition: pending -> completed."""
        job = IntegrationJob(job_id="job-006", user_id="user-123")

        assert job.can_transition_to("completed") is False
        assert job.transition("completed") is False
        assert job.status == "pending"

    def test_invalid_transition_completed_to_anything(self):
        """Test that completed is a terminal state."""
        job = IntegrationJob(job_id="job-007", user_id="user-123", status="completed")

        for status in ["pending", "running", "failed", "cancelled"]:
            assert job.can_transition_to(status) is False

    def test_invalid_transition_failed_to_anything(self):
        """Test that failed is a terminal state."""
        job = IntegrationJob(job_id="job-008", user_id="user-123", status="failed")

        for status in ["pending", "running", "completed", "cancelled"]:
            assert job.can_transition_to(status) is False

    def test_checkpoint_storage(self):
        """Test storing checkpoint data on job."""
        job = IntegrationJob(job_id="job-009", user_id="user-123")
        job.checkpoint = {
            "step": 5,
            "processed_ids": ["id1", "id2"],
            "last_memory_id": "mem-100",
        }

        assert job.checkpoint["step"] == 5
        assert len(job.checkpoint["processed_ids"]) == 2

    def test_job_lifecycle(self):
        """Test complete job lifecycle: pending -> running -> completed."""
        job = IntegrationJob(job_id="job-010", user_id="user-123")

        # Start the job
        job.transition("running")
        assert job.status == "running"

        # Simulate work with checkpoint
        job.checkpoint["memories_processed"] = 50

        # Complete the job
        job.result = GraphPatchResult(
            success=True, run_id="job-010", user_id="user-123", links_created=10
        )
        job.transition("completed")

        assert job.status == "completed"
        assert job.result.links_created == 10
