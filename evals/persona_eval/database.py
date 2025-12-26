"""
SQLite database for eval provenance and analysis.

Stores evaluation results in a queryable format for root cause analysis.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class EvalResult:
    """A single evaluation result."""

    question_id: str
    benchmark: str
    question_type: str
    question_text: str
    gold_answer: str
    generated_answer: str
    correct: bool

    # Retrieval info
    retrieved_node_ids: List[str]
    retrieved_node_count: int
    retrieval_duration_ms: float

    # Ingestion info
    session_count: int
    nodes_created: int
    ingestion_duration_ms: float

    # Failure analysis
    failure_category: Optional[str] = None
    failure_notes: Optional[str] = None

    # Timestamps
    timestamp: str = ""


class EvalDatabase:
    """SQLite database for evaluation results and analysis."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS questions (
        question_id TEXT PRIMARY KEY,
        benchmark TEXT NOT NULL,
        question_type TEXT NOT NULL,
        question_text TEXT NOT NULL,
        gold_answer TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        system_name TEXT NOT NULL,
        notes TEXT,  -- User annotation: "the idea behind this run"
        config TEXT,  -- JSON blob of run configuration
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        system_name TEXT NOT NULL,
        generated_answer TEXT NOT NULL,
        correct INTEGER NOT NULL,
        
        -- Retrieval metrics
        retrieved_node_ids TEXT,  -- JSON array
        retrieved_node_count INTEGER,
        retrieval_duration_ms REAL,
        
        -- Ingestion metrics
        session_count INTEGER,
        nodes_created INTEGER,
        ingestion_duration_ms REAL,
        
        -- Failure analysis
        failure_category TEXT,
        failure_notes TEXT,
        
        -- Human annotation
        annotated INTEGER DEFAULT 0,
        annotation_notes TEXT,
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        FOREIGN KEY (question_id) REFERENCES questions(question_id)
    );
    
    CREATE TABLE IF NOT EXISTS retrieved_nodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        result_id INTEGER NOT NULL,
        node_id TEXT NOT NULL,
        node_type TEXT,
        node_content TEXT,
        similarity_score REAL,
        retrieval_rank INTEGER,
        was_relevant INTEGER,  -- Post-hoc annotation: was this node actually relevant?
        
        FOREIGN KEY (result_id) REFERENCES results(id)
    );
    
    CREATE TABLE IF NOT EXISTS failure_taxonomy (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT UNIQUE NOT NULL,
        description TEXT,
        examples TEXT,  -- JSON array of example question_ids
        count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Indexes for fast queries
    CREATE INDEX IF NOT EXISTS idx_results_correct ON results(correct);
    CREATE INDEX IF NOT EXISTS idx_results_system ON results(system_name);
    CREATE INDEX IF NOT EXISTS idx_results_failure ON results(failure_category);
    CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
    CREATE INDEX IF NOT EXISTS idx_questions_type ON questions(question_type);
    CREATE INDEX IF NOT EXISTS idx_questions_benchmark ON questions(benchmark);
    """

    DEFAULT_TAXONOMY = [
        ("cross_session_gap", "Information across sessions not linked", "[]"),
        ("temporal_blindness", "Date ordering not captured", "[]"),
        ("entity_resolution", "Same entity not recognized across mentions", "[]"),
        ("insufficient_context", "Retrieved context too short or sparse", "[]"),
        ("semantic_mismatch", "Query doesn't match node embeddings", "[]"),
        ("ingestion_failure", "Data not indexed properly", "[]"),
        ("hallucination", "Answer contains made-up information", "[]"),
        ("wrong_aggregation", "Counting/summarization error", "[]"),
        ("abstention_failure", "Failed to say 'I don't know' when appropriate", "[]"),
        (
            "format_error",
            "Answer format doesn't match expected (e.g., wrong multiple choice letter)",
            "[]",
        ),
        ("api_error", "API error (rate limit, timeout, network failure)", "[]"),
    ]

    def __init__(self, db_path: str = "eval_results.db"):
        """Initialize the database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)

            # Insert default failure taxonomy
            for category, description, examples in self.DEFAULT_TAXONOMY:
                conn.execute(
                    """INSERT OR IGNORE INTO failure_taxonomy 
                       (category, description, examples) VALUES (?, ?, ?)""",
                    (category, description, examples),
                )
            conn.commit()

    def import_from_jsonl(self, jsonl_path: str, run_id: str, system_name: str) -> int:
        imported = 0
        skipped = 0

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO runs (run_id, system_name) VALUES (?, ?)",
                (run_id, system_name),
            )

            existing = set(
                row[0]
                for row in conn.execute(
                    "SELECT question_id FROM results WHERE run_id = ?", (run_id,)
                ).fetchall()
            )

            with open(jsonl_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        log = json.loads(line)
                        question_id = log["question_id"]

                        if question_id in existing:
                            skipped += 1
                            continue

                        conn.execute(
                            """INSERT OR IGNORE INTO questions 
                               (question_id, benchmark, question_type, question_text, gold_answer)
                               VALUES (?, ?, ?, ?, ?)""",
                            (
                                log["question_id"],
                                log["benchmark"],
                                log["question_type"],
                                log["question"],
                                log["evaluation"]["gold_answer"],
                            ),
                        )

                        # Get retrieved node IDs
                        retrieved_ids = []
                        if "retrieval" in log and "graph_traversal" in log["retrieval"]:
                            retrieved_ids = log["retrieval"]["graph_traversal"].get(
                                "final_ranked_nodes", []
                            )

                        # Insert result
                        cursor = conn.execute(
                            """INSERT INTO results 
                               (question_id, run_id, system_name, generated_answer, correct,
                                retrieved_node_ids, retrieved_node_count, retrieval_duration_ms,
                                session_count, nodes_created, ingestion_duration_ms)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                log["question_id"],
                                run_id,
                                system_name,
                                log["generation"]["answer"],
                                1 if log["evaluation"].get("correct") else 0,
                                json.dumps(retrieved_ids),
                                len(retrieved_ids),
                                log["retrieval"]["duration_ms"],
                                log["ingestion"]["sessions_count"],
                                log["ingestion"]["nodes_created"],
                                log["ingestion"]["duration_ms"],
                            ),
                        )

                        result_id = cursor.lastrowid

                        # Insert retrieved nodes
                        if "retrieval" in log and "vector_search" in log["retrieval"]:
                            for i, seed in enumerate(
                                log["retrieval"]["vector_search"].get("seeds", [])
                            ):
                                conn.execute(
                                    """INSERT INTO retrieved_nodes
                                       (result_id, node_id, node_type, similarity_score, retrieval_rank)
                                       VALUES (?, ?, ?, ?, ?)""",
                                    (
                                        result_id,
                                        seed.get("node_id", ""),
                                        seed.get("node_type", ""),
                                        seed.get("score", 0),
                                        i + 1,
                                    ),
                                )

                        imported += 1

                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Warning: Could not parse line: {e}")
                        continue

            conn.commit()

        if skipped > 0:
            print(f"  Skipped {skipped} already imported questions")
        return imported

    def get_failure_summary(self, system_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of failures by type and category."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Build query with optional system filter
            system_filter = "AND r.system_name = ?" if system_name else ""
            params = (system_name,) if system_name else ()

            # Overall stats
            overall = conn.execute(
                f"""
                SELECT 
                    COUNT(*) as total,
                    SUM(r.correct) as correct,
                    AVG(r.correct) * 100 as accuracy
                FROM results r
                WHERE 1=1 {system_filter}
            """,
                params,
            ).fetchone()

            # By question type
            by_type = conn.execute(
                f"""
                SELECT 
                    q.question_type,
                    COUNT(*) as total,
                    SUM(r.correct) as correct,
                    AVG(r.correct) * 100 as accuracy
                FROM results r
                JOIN questions q ON r.question_id = q.question_id
                WHERE 1=1 {system_filter}
                GROUP BY q.question_type
                ORDER BY accuracy ASC
            """,
                params,
            ).fetchall()

            # By failure category
            by_category = conn.execute(
                f"""
                SELECT 
                    r.failure_category,
                    COUNT(*) as count
                FROM results r
                WHERE r.correct = 0 
                  AND r.failure_category IS NOT NULL
                  {system_filter.replace("AND", "AND" if system_filter else "")}
                GROUP BY r.failure_category
                ORDER BY count DESC
            """,
                params,
            ).fetchall()

            return {
                "overall": {
                    "total": overall["total"],
                    "correct": overall["correct"],
                    "accuracy": round(overall["accuracy"] or 0, 2),
                },
                "by_type": [dict(row) for row in by_type],
                "by_failure_category": [dict(row) for row in by_category],
            }

    def get_failures(
        self,
        system_name: Optional[str] = None,
        question_type: Optional[str] = None,
        failure_category: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get failed questions for analysis."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = """
                SELECT 
                    r.id, r.question_id, r.system_name, r.generated_answer, r.correct,
                    r.retrieved_node_count, r.retrieval_duration_ms,
                    r.session_count, r.nodes_created, r.ingestion_duration_ms,
                    r.failure_category, r.failure_notes, r.annotated,
                    q.question_type, q.question_text, q.gold_answer, q.benchmark
                FROM results r
                JOIN questions q ON r.question_id = q.question_id
                WHERE r.correct = 0
            """
            params = []

            if system_name:
                query += " AND r.system_name = ?"
                params.append(system_name)
            if question_type:
                query += " AND q.question_type = ?"
                params.append(question_type)
            if failure_category:
                query += " AND r.failure_category = ?"
                params.append(failure_category)

            query += f" ORDER BY r.id DESC LIMIT {limit}"

            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def annotate_failure(
        self, result_id: int, failure_category: str, notes: Optional[str] = None
    ):
        """Annotate a failure with its category."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE results 
                   SET failure_category = ?, failure_notes = ?, annotation_notes = ?, annotated = 1
                   WHERE id = ?""",
                (failure_category, notes, notes, result_id),
            )

            # Update category count
            conn.execute(
                """UPDATE failure_taxonomy 
                   SET count = count + 1 
                   WHERE category = ?""",
                (failure_category,),
            )
            conn.commit()

    def get_taxonomy(self) -> List[Dict[str, Any]]:
        """Get the failure taxonomy with counts."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT category, description, count 
                   FROM failure_taxonomy 
                   ORDER BY count DESC"""
            ).fetchall()
            return [dict(row) for row in rows]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            return dict(row) if row else None

    def update_run_notes(self, run_id: str, notes: str) -> None:
        """Update notes for a run."""
        with sqlite3.connect(self.db_path) as conn:
            # Upsert: insert if not exists, update if exists
            conn.execute(
                """INSERT INTO runs (run_id, system_name, notes, updated_at)
                   VALUES (?, '', ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(run_id) DO UPDATE SET 
                   notes = excluded.notes, updated_at = CURRENT_TIMESTAMP""",
                (run_id, notes),
            )
            conn.commit()

    def ensure_run_exists(self, run_id: str, system_name: str) -> None:
        """Ensure a run record exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO runs (run_id, system_name)
                   VALUES (?, ?)""",
                (run_id, system_name),
            )
            conn.commit()

    def get_unannotated_failures(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get failures that haven't been annotated yet."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT 
                    r.id, r.question_id, r.system_name, r.generated_answer,
                    r.retrieved_node_count, r.retrieval_duration_ms,
                    q.question_type, q.question_text, q.gold_answer
                FROM results r
                JOIN questions q ON r.question_id = q.question_id
                WHERE r.correct = 0 AND r.annotated = 0
                ORDER BY RANDOM()
                LIMIT ?
            """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]


# Quick test
if __name__ == "__main__":
    db = EvalDatabase("test_eval.db")
    print("Database initialized successfully!")
    print(f"Taxonomy: {db.get_taxonomy()}")
