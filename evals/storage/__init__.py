"""
SQLite-based EventStore implementation.

Design Principles:
- Append-only event storage for auditability
- Materialized views for fast queries
- Schema versioning for migrations
- Thread-safe via connection-per-call pattern
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

from ..core.interfaces import Event, EventStore
from ..core.models import EvalResult, MetricResult


class SQLiteEventStore:
    """
    SQLite implementation of EventStore.

    Schema:
        events: Append-only event log
        runs: Materialized view of run metadata
        results: Materialized view of individual results
        metrics: Materialized view of metric results

    Usage:
        store = SQLiteEventStore("evals/results/eval.db")
        store.append(Event(...))

        # After run completes
        store.materialize_views(run_id="abc123")

        # Query
        runs = store.list_runs()
        results = store.get_results(run_id="abc123")
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def schema_version(self) -> int:
        return self.SCHEMA_VERSION

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """Get a connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._conn() as conn:
            conn.executescript("""
                -- Schema version tracking
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    migrated_at TEXT NOT NULL
                );
                
                -- Append-only event log
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    payload_version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                
                CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id);
                CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
                
                -- Materialized: Run summaries
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    benchmark TEXT,
                    adapter TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    total_cases INTEGER,
                    passed_cases INTEGER,
                    failed_cases INTEGER,
                    error_cases INTEGER,
                    pass_rate REAL,
                    metadata TEXT  -- JSON
                );
                
                -- Materialized: Individual results
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    test_case_id TEXT NOT NULL,
                    benchmark TEXT,
                    question_type TEXT,
                    adapter TEXT,
                    answer TEXT,
                    reference_answer TEXT,
                    passed INTEGER,
                    error TEXT,
                    latency_ms REAL,
                    started_at TEXT,
                    finished_at TEXT,
                    artifacts TEXT,  -- JSON
                    UNIQUE(run_id, test_case_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_results_run_id ON results(run_id);
                CREATE INDEX IF NOT EXISTS idx_results_question_type ON results(question_type);
                
                -- Materialized: Metric results
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    test_case_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    kind TEXT,
                    score_type TEXT,
                    score REAL,
                    passed INTEGER,
                    reason TEXT,
                    threshold REAL,
                    UNIQUE(run_id, test_case_id, metric_name)
                );
                
                CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON metrics(run_id);
                CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name);
            """)

            # Record schema version
            existing = conn.execute(
                "SELECT version FROM schema_version WHERE version = ?",
                (self.SCHEMA_VERSION,),
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO schema_version (version, migrated_at) VALUES (?, ?)",
                    (self.SCHEMA_VERSION, datetime.utcnow().isoformat()),
                )

            conn.commit()

    def append(self, event: Event) -> None:
        """Append an event to the store."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO events (type, occurred_at, run_id, payload, payload_version)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.type,
                    event.occurred_at.isoformat(),
                    event.run_id,
                    json.dumps(event.payload, default=str),
                    event.payload_version,
                ),
            )
            conn.commit()

    def iter_events(
        self,
        *,
        run_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> Sequence[Event]:
        """Iterate events with optional filters."""
        query = "SELECT * FROM events WHERE 1=1"
        params: list[Any] = []

        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if event_type:
            query += " AND type = ?"
            params.append(event_type)

        query += " ORDER BY id"

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                Event(
                    type=row["type"],
                    occurred_at=datetime.fromisoformat(row["occurred_at"]),
                    run_id=row["run_id"],
                    payload=json.loads(row["payload"]),
                    payload_version=row["payload_version"],
                )
                for row in rows
            ]

    def materialize_views(self, *, run_id: str) -> None:
        """Materialize derived views for a run."""
        events = self.iter_events(run_id=run_id)

        run_started = None
        run_completed = None
        case_results: list[dict] = []

        for event in events:
            if event.type == "run_started":
                run_started = event
            elif event.type == "run_completed":
                run_completed = event
            elif event.type == "case_finished":
                case_results.append(dict(event.payload))

        if not run_started:
            return

        with self._conn() as conn:
            # Insert or update run summary
            payload = run_completed.payload if run_completed else {}
            conn.execute(
                """
                INSERT OR REPLACE INTO runs 
                (run_id, benchmark, adapter, started_at, finished_at, 
                 total_cases, passed_cases, failed_cases, error_cases, pass_rate, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    run_started.payload.get("benchmark"),
                    run_started.payload.get("adapter"),
                    run_started.occurred_at.isoformat(),
                    run_completed.occurred_at.isoformat() if run_completed else None,
                    payload.get("total", 0),
                    payload.get("passed", 0),
                    payload.get("failed", 0),
                    payload.get("errors", 0),
                    payload.get("pass_rate", 0.0),
                    json.dumps(run_started.payload),
                ),
            )

            # Insert results from case_finished events
            for case in case_results:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO results
                    (run_id, test_case_id, benchmark, question_type, adapter,
                     answer, reference_answer, passed, error, latency_ms,
                     started_at, finished_at, artifacts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        case.get("test_case_id"),
                        case.get("benchmark"),
                        case.get("question_type"),
                        case.get("adapter"),
                        case.get("answer"),
                        case.get("reference_answer"),
                        1 if case.get("passed") else 0,
                        case.get("error"),
                        case.get("latency_ms"),
                        case.get("started_at"),
                        case.get("finished_at"),
                        json.dumps(case.get("artifacts", {})),
                    ),
                )

                # Insert metric results
                for metric in case.get("metrics", []):
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO metrics
                        (run_id, test_case_id, metric_name, kind, score_type,
                         score, passed, reason, threshold)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            case.get("test_case_id"),
                            metric.get("metric"),
                            metric.get("kind"),
                            metric.get("score_type"),
                            metric.get("score"),
                            1 if metric.get("passed") else 0,
                            metric.get("reason"),
                            metric.get("threshold"),
                        ),
                    )

            conn.commit()

    # === Convenience Query Methods ===

    def list_runs(self, limit: int = 50) -> list[dict]:
        """List recent runs."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM runs 
                ORDER BY started_at DESC 
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_run(self, run_id: str) -> Optional[dict]:
        """Get a single run by ID."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_results(
        self,
        run_id: str,
        *,
        question_type: Optional[str] = None,
        passed: Optional[bool] = None,
    ) -> list[dict]:
        """Get results for a run with optional filters."""
        query = "SELECT * FROM results WHERE run_id = ?"
        params: list[Any] = [run_id]

        if question_type:
            query += " AND question_type = ?"
            params.append(question_type)
        if passed is not None:
            query += " AND passed = ?"
            params.append(1 if passed else 0)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_metrics_for_case(self, run_id: str, test_case_id: str) -> list[dict]:
        """Get all metric results for a specific test case."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM metrics 
                WHERE run_id = ? AND test_case_id = ?
                """,
                (run_id, test_case_id),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_aggregate_by_type(self, run_id: str) -> dict[str, dict]:
        """Get pass rates grouped by question type."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT 
                    question_type,
                    COUNT(*) as total,
                    SUM(passed) as passed_count,
                    AVG(passed) as pass_rate
                FROM results
                WHERE run_id = ?
                GROUP BY question_type
                """,
                (run_id,),
            ).fetchall()
            return {
                row["question_type"]: {
                    "total": row["total"],
                    "passed": row["passed_count"],
                    "pass_rate": row["pass_rate"],
                }
                for row in rows
            }

    def get_aggregate_by_metric(self, run_id: str) -> dict[str, dict]:
        """Get pass rates grouped by metric."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT 
                    metric_name,
                    COUNT(*) as total,
                    SUM(passed) as passed_count,
                    AVG(passed) as pass_rate,
                    AVG(score) as avg_score
                FROM metrics
                WHERE run_id = ?
                GROUP BY metric_name
                """,
                (run_id,),
            ).fetchall()
            return {
                row["metric_name"]: {
                    "total": row["total"],
                    "passed": row["passed_count"],
                    "pass_rate": row["pass_rate"],
                    "avg_score": row["avg_score"],
                }
                for row in rows
            }

    # === Result Recording ===

    def record_eval_result(self, result: EvalResult) -> None:
        """
        Record an evaluation result directly (bypass event log).

        Useful for incremental recording during long runs.
        """
        event = Event(
            type="case_finished",
            occurred_at=result.finished_at or datetime.utcnow(),
            run_id=result.run_id,
            payload={
                "test_case_id": result.test_case_id,
                "benchmark": result.benchmark,
                "question_type": result.question_type,
                "adapter": result.adapter,
                "answer": result.query_result.answer,
                "passed": result.passed,
                "error": result.query_result.error,
                "latency_ms": result.query_result.latency.total_ms,
                "started_at": result.started_at.isoformat()
                if result.started_at
                else None,
                "finished_at": result.finished_at.isoformat()
                if result.finished_at
                else None,
                "metrics": list(
                    {
                        "metric": m.metric,
                        "kind": m.kind,
                        "score_type": m.score_type,
                        "score": m.score,
                        "passed": m.passed,
                        "reason": m.reason,
                        "threshold": m.threshold,
                    }
                    for m in result.metric_results
                ),
            },
        )
        self.append(event)
