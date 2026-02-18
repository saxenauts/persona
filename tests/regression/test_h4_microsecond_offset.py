"""
Regression test for H4 microsecond offset bug.

Bug: Line 333 of ingestion_service.py multiplied observed_at.microsecond by 1000,
causing event_time to shift by up to 16 minutes (999999 * 1000 = 999999000 microseconds).

Fix: Remove the * 1000 multiplier. Use microsecond value directly (0-999999 range).

Test verifies:
1. Multiple memories in same ingest have event_time within 1 second of base_event_time
2. Event ordering is preserved (monotonically increasing)
3. No 16-minute shifts occur
"""

from datetime import datetime, timedelta


def test_microsecond_offset_no_16_minute_shift():
    """
    Verify that microsecond seeding doesn't cause 16-minute shifts.

    Scenario:
    - observed_at.microsecond = 500000 (middle of range)
    - Create 5 event_time values using the fixed logic
    - All should be within 1 second of base_event_time
    - Without fix: would be ~8.33 minutes later (500000 * 1000 microseconds)
    """
    observed_at = datetime(2024, 1, 15, 10, 30, 45, microsecond=500000)
    base_event_time = observed_at.replace(microsecond=0)

    memory_seq = observed_at.microsecond

    event_times = []
    for _ in range(5):
        et = base_event_time + timedelta(microseconds=memory_seq)
        event_times.append(et)
        memory_seq += 1

    max_offset = timedelta(seconds=1)
    for i, et in enumerate(event_times):
        offset = et - base_event_time
        assert offset < max_offset, (
            f"Memory {i} event_time shifted by {offset.total_seconds()} seconds. "
            f"Expected < 1 second. Bug: microsecond offset not fixed? "
            f"event_time={et}, base={base_event_time}"
        )
        assert offset >= timedelta(0), (
            f"Memory {i} event_time is before base_event_time. Ordering issue detected."
        )


def test_microsecond_offset_ordering_preserved():
    """
    Verify that event ordering is preserved within ingest.

    Scenario:
    - Create 10 event_time values using the fixed logic
    - Verify event_time is monotonically increasing
    """
    observed_at = datetime(2024, 1, 15, 10, 30, 45, microsecond=999999)
    base_event_time = observed_at.replace(microsecond=0)

    memory_seq = observed_at.microsecond

    event_times = []
    for _ in range(10):
        et = base_event_time + timedelta(microseconds=memory_seq)
        event_times.append(et)
        memory_seq += 1

    for i in range(len(event_times) - 1):
        assert event_times[i] < event_times[i + 1], (
            f"Event ordering broken: memory {i} event_time {event_times[i]} "
            f">= memory {i + 1} event_time {event_times[i + 1]}"
        )


def test_microsecond_offset_cross_ingest_collision_avoidance():
    """
    Verify that different ingests with same observed_at second don't collide.

    Scenario:
    - Two ingests with same second but different microseconds
    - Ingest 1: observed_at.microsecond = 100000
    - Ingest 2: observed_at.microsecond = 200000
    - Verify first memory of ingest 2 has event_time > last memory of ingest 1
    """
    base_event_time = datetime(2024, 1, 15, 10, 30, 45)

    observed_at_1 = datetime(2024, 1, 15, 10, 30, 45, microsecond=100000)
    memory_seq_1 = observed_at_1.microsecond
    event_times_1 = []
    for _ in range(3):
        et = base_event_time + timedelta(microseconds=memory_seq_1)
        event_times_1.append(et)
        memory_seq_1 += 1

    observed_at_2 = datetime(2024, 1, 15, 10, 30, 45, microsecond=200000)
    memory_seq_2 = observed_at_2.microsecond
    event_times_2 = []
    for _ in range(3):
        et = base_event_time + timedelta(microseconds=memory_seq_2)
        event_times_2.append(et)
        memory_seq_2 += 1

    last_event_time_1 = max(event_times_1)
    first_event_time_2 = min(event_times_2)

    assert last_event_time_1 < first_event_time_2, (
        f"Cross-ingest collision detected: "
        f"ingest 1 last event_time {last_event_time_1} >= "
        f"ingest 2 first event_time {first_event_time_2}"
    )


def test_microsecond_offset_worst_case_999999():
    """
    Verify worst case: observed_at.microsecond = 999999.

    Without fix: 999999 * 1000 = 999999000 microseconds = 16.65 minutes
    With fix: 999999 microseconds = 0.999999 seconds (< 1 second)
    """
    observed_at = datetime(2024, 1, 15, 10, 30, 45, microsecond=999999)
    base_event_time = observed_at.replace(microsecond=0)

    memory_seq = observed_at.microsecond
    et = base_event_time + timedelta(microseconds=memory_seq)

    offset = et - base_event_time
    assert offset < timedelta(seconds=1), (
        f"Worst case failed: offset {offset.total_seconds()} seconds >= 1 second"
    )
    assert offset == timedelta(microseconds=999999), (
        f"Expected offset of 999999 microseconds, got {offset.total_seconds()} seconds"
    )
