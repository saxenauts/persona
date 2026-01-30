"""
Regression test for H3 status case mismatch bug.

Bug: NoteMemory.status defaults to lowercase "active", but filtering code
checked uppercase "COMPLETED", causing completed notes to leak into active context.

Fix: Normalize status comparison to use .lower() in:
1. persona/core/context.py line 120: _format_notes_prose()
2. persona/services/consolidation_service.py line 60: UserCardService.generate()

Test verifies:
1. Notes with status="completed" (lowercase) do NOT appear in format_working_memory_prose()
2. Notes with status="COMPLETED" (uppercase) do NOT appear appear in format_working_memory_prose()
3. Notes with status="active" (lowercase) DO appear in format_working_memory_prose()
4. Notes with status="ACTIVE" (uppercase) DO appear in format_working_memory_prose()
"""

from uuid import uuid4

import pytest

from persona.models.memory import NoteMemory, UserCard
from persona.core.context import format_working_memory_prose


@pytest.fixture
def user_card():
    """Provide a basic UserCard for testing."""
    return UserCard(
        user_id=str(uuid4()),
        timezone="UTC",
        identity_prose="Test user",
    )


def test_completed_note_lowercase_excluded(user_card):
    """
    Verify that notes with status="completed" (lowercase) are excluded from prose.

    This is the primary bug case - status defaults to lowercase "active",
    so completed notes would also be lowercase "completed".
    """
    completed_note = NoteMemory(
        user_id=user_card.user_id,
        title="Completed Task",
        content="This task is done",
        status="completed",  # lowercase - the default case
    )

    prose = format_working_memory_prose(user_card, [], [], [completed_note])

    # Completed note should NOT appear in prose
    assert "Completed Task" not in prose
    assert "This task is done" not in prose


def test_completed_note_uppercase_excluded(user_card):
    """
    Verify that notes with status="COMPLETED" (uppercase) are also excluded.

    This tests the edge case where status might be uppercase.
    """
    completed_note = NoteMemory(
        user_id=user_card.user_id,
        title="Completed Task",
        content="This task is done",
        status="COMPLETED",  # uppercase - edge case
    )

    prose = format_working_memory_prose(user_card, [], [], [completed_note])

    # Completed note should NOT appear in prose
    assert "Completed Task" not in prose
    assert "This task is done" not in prose


def test_active_note_lowercase_included(user_card):
    """
    Verify that notes with status="active" (lowercase) are included in prose.

    This is the default case - status defaults to "active".
    """
    active_note = NoteMemory(
        user_id=user_card.user_id,
        title="Active Task",
        content="This task is pending",
        status="active",  # lowercase - the default
    )

    prose = format_working_memory_prose(user_card, [], [], [active_note])

    # Active note SHOULD appear in prose
    assert "Active Task" in prose


def test_active_note_uppercase_included(user_card):
    """
    Verify that notes with status="ACTIVE" (uppercase) are also included.

    This tests the edge case where status might be uppercase.
    """
    active_note = NoteMemory(
        user_id=user_card.user_id,
        title="Active Task",
        content="This task is pending",
        status="ACTIVE",  # uppercase - edge case
    )

    prose = format_working_memory_prose(user_card, [], [], [active_note])

    # Active note SHOULD appear in prose
    assert "Active Task" in prose


def test_mixed_status_notes(user_card):
    """
    Verify that mixed status notes are filtered correctly.

    Scenario: 3 notes - 2 active, 1 completed.
    Expected: Only active notes appear in prose.
    """
    active_note_1 = NoteMemory(
        user_id=user_card.user_id,
        title="Task 1",
        content="First active task",
        status="active",
    )

    completed_note = NoteMemory(
        user_id=user_card.user_id,
        title="Task 2",
        content="Completed task",
        status="completed",
    )

    active_note_2 = NoteMemory(
        user_id=user_card.user_id,
        title="Task 3",
        content="Second active task",
        status="active",
    )

    notes = [active_note_1, completed_note, active_note_2]
    prose = format_working_memory_prose(user_card, [], [], notes)

    # Active notes should appear
    assert "Task 1" in prose
    assert "Task 3" in prose

    # Completed note should NOT appear
    assert "Task 2" not in prose
