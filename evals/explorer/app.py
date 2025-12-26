"""
Eval Explorer v2 - Flask app for browsing evaluation results.

Features:
- System-agnostic: Works with Graphiti, Persona, and other adapters
- Pipeline-centric: Shows ingestion → retrieval → generation stages with timing
- Dynamic field detection: Shows whatever fields exist in the data

Run with: python -m evals.explorer.app
"""

import json
import sqlite3
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent / "data"
STAGE_LOGS_DIR = RESULTS_DIR / "graphiti_stage_logs"


def get_db_path() -> Path:
    return RESULTS_DIR / "eval_analysis.db"


def get_db():
    db = sqlite3.connect(get_db_path())
    db.row_factory = sqlite3.Row
    return db


def load_stage_log(question_id: str) -> Dict[str, Any]:
    """Load stage log for a question from graphiti_stage_logs directory."""
    if not STAGE_LOGS_DIR.exists():
        return {}

    for f in STAGE_LOGS_DIR.iterdir():
        if (
            f.suffix == ".jsonl"
            and question_id.replace("personamem_32k_", "") in f.stem
        ):
            stages = {}
            with open(f) as fp:
                for line in fp:
                    if line.strip():
                        data = json.loads(line)
                        stage_name = data.get("stage", "unknown")
                        stages[stage_name] = data
            return stages
    return {}


def load_deep_log(question_id: str, run_id: Optional[str] = None) -> Optional[Dict]:
    """Load deep log for a question, checking multiple sources."""
    if not run_id:
        return None

    for results_subdir in RESULTS_DIR.iterdir():
        if not results_subdir.is_dir():
            continue

        deep_logs_file = results_subdir / "deep_logs.jsonl"
        if deep_logs_file.exists():
            with open(deep_logs_file) as f:
                for line in f:
                    if line.strip():
                        try:
                            log = json.loads(line)
                            if log.get("question_id") == question_id:
                                return log
                        except json.JSONDecodeError:
                            continue
    return None


def load_shared_contexts(benchmark: str, variant: str = "32k") -> Dict[str, List[Dict]]:
    """Load shared context sessions for PersonaMem benchmark."""
    if benchmark == "personamem":
        path = DATA_DIR / "personamem" / f"shared_contexts_{variant}.jsonl"
        if not path.exists():
            return {}
        contexts = {}
        with open(path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    contexts.update(data)
        return contexts
    return {}


def load_questions_metadata(benchmark: str, variant: str = "32k") -> Dict[str, Dict]:
    """Load question metadata for mapping to shared contexts."""
    if benchmark == "personamem":
        path = DATA_DIR / "personamem" / f"questions_{variant}_{variant}.json"
        if not path.exists():
            return {}
        with open(path) as f:
            questions = json.load(f)
        return {f"personamem_{variant}_{i}": q for i, q in enumerate(questions)}
    return {}


def split_into_sessions(turns: List[Dict]) -> List[Dict]:
    """Split flat turns into separate sessions."""
    if not turns:
        return []

    sessions = []
    current_session = {"turns": [], "id": 0}

    for turn in turns:
        if turn.get("role") == "system" and current_session["turns"]:
            sessions.append(current_session)
            current_session = {"turns": [], "id": len(sessions)}
        current_session["turns"].append(turn)

    if current_session["turns"]:
        sessions.append(current_session)

    return sessions


def check_gold_in_session(session_turns: List[Dict], gold_answer: str) -> bool:
    """Check if gold answer appears in any session turn."""
    if not gold_answer:
        return False
    gold_lower = gold_answer.lower().strip()
    for turn in session_turns:
        content = turn.get("content", "").lower()
        if gold_lower in content:
            return True
    return False


def extract_pipeline_metrics(
    deep_log: Optional[Dict], stage_log: Dict
) -> Dict[str, Any]:
    """Extract pipeline timing and metrics from available logs."""
    metrics = {
        "ingestion": {"duration_ms": 0, "sessions": 0, "nodes": 0, "relationships": 0},
        "retrieval": {"duration_ms": 0, "nodes_retrieved": 0, "edges_retrieved": 0},
        "generation": {"duration_ms": 0, "model": "unknown"},
        "total_duration_ms": 0,
    }

    if stage_log.get("stage1_ingestion_complete"):
        ing = stage_log["stage1_ingestion_complete"]
        metrics["ingestion"]["sessions"] = ing.get("total_sessions", 0)
        metrics["ingestion"]["success"] = ing.get("success_count", 0)

    if stage_log.get("stage4_generation"):
        gen = stage_log["stage4_generation"]
        metrics["retrieval"]["duration_ms"] = gen.get("retrieval_ms", 0)
        metrics["retrieval"]["nodes_retrieved"] = gen.get("nodes_count", 0)
        metrics["retrieval"]["edges_retrieved"] = gen.get("edges_count", 0)
        metrics["generation"]["duration_ms"] = gen.get("generation_ms", 0)

    if deep_log:
        if "ingestion" in deep_log:
            ing = deep_log["ingestion"]
            metrics["ingestion"]["duration_ms"] = ing.get("duration_ms", 0)
            metrics["ingestion"]["nodes"] = ing.get("nodes_created", 0)
            metrics["ingestion"]["relationships"] = ing.get("relationships_created", 0)
            if ing.get("memories_created"):
                metrics["ingestion"]["memories"] = ing["memories_created"]

        if "retrieval" in deep_log:
            ret = deep_log["retrieval"]
            if not metrics["retrieval"]["duration_ms"]:
                metrics["retrieval"]["duration_ms"] = ret.get("duration_ms", 0)
            metrics["retrieval"]["context"] = ret.get("retrieved_context", "")

        if "generation" in deep_log:
            gen = deep_log["generation"]
            if not metrics["generation"]["duration_ms"]:
                metrics["generation"]["duration_ms"] = gen.get("duration_ms", 0)
            metrics["generation"]["model"] = gen.get("model", "unknown")

    metrics["total_duration_ms"] = (
        metrics["ingestion"]["duration_ms"]
        + metrics["retrieval"]["duration_ms"]
        + metrics["generation"]["duration_ms"]
    )

    return metrics


@app.route("/")
def index():
    db = get_db()

    runs = db.execute("""
        SELECT 
            res.run_id,
            res.system_name,
            COUNT(res.id) as total,
            SUM(res.correct) as correct,
            ROUND(AVG(res.correct) * 100, 1) as accuracy,
            MIN(res.created_at) as started_at,
            MAX(res.created_at) as ended_at,
            COALESCE(r.notes, '') as notes,
            ROUND(AVG(res.ingestion_duration_ms + res.retrieval_duration_ms), 0) as avg_duration_ms
        FROM results res
        LEFT JOIN runs r ON res.run_id = r.run_id
        GROUP BY res.run_id, res.system_name
        ORDER BY started_at DESC
    """).fetchall()

    return render_template("index.html", runs=runs)


@app.route("/run/<run_id>")
def run_overview(run_id: str):
    db = get_db()

    run_meta = db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()

    overview = db.execute(
        """
        SELECT 
            run_id,
            system_name,
            COUNT(*) as total,
            SUM(correct) as correct,
            ROUND(AVG(correct) * 100, 1) as accuracy,
            ROUND(AVG(ingestion_duration_ms), 0) as avg_ingestion_ms,
            ROUND(AVG(retrieval_duration_ms), 0) as avg_retrieval_ms,
            SUM(ingestion_duration_ms) as total_ingestion_ms,
            SUM(retrieval_duration_ms) as total_retrieval_ms
        FROM results
        WHERE run_id = ?
        GROUP BY run_id, system_name
    """,
        (run_id,),
    ).fetchone()

    by_type = db.execute(
        """
        SELECT 
            q.question_type,
            COUNT(*) as total,
            SUM(r.correct) as correct,
            ROUND(AVG(r.correct) * 100, 1) as accuracy,
            ROUND(AVG(r.retrieval_duration_ms), 0) as avg_retrieval_ms
        FROM results r
        JOIN questions q ON r.question_id = q.question_id
        WHERE r.run_id = ?
        GROUP BY q.question_type
        ORDER BY accuracy ASC
    """,
        (run_id,),
    ).fetchall()

    by_category = db.execute(
        """
        SELECT 
            failure_category,
            COUNT(*) as count
        FROM results
        WHERE run_id = ? AND correct = 0 AND failure_category IS NOT NULL
        GROUP BY failure_category
        ORDER BY count DESC
    """,
        (run_id,),
    ).fetchall()

    filter_type = request.args.get("type")
    filter_status = request.args.get("status")

    query = """
        SELECT 
            r.id,
            r.question_id,
            q.question_type,
            q.question_text,
            q.gold_answer,
            r.generated_answer,
            r.failure_category,
            r.retrieved_node_count,
            r.correct,
            r.retrieval_duration_ms,
            r.ingestion_duration_ms
        FROM results r
        JOIN questions q ON r.question_id = q.question_id
        WHERE r.run_id = ?
    """
    params = [run_id]

    if filter_type:
        query += " AND q.question_type = ?"
        params.append(filter_type)
    if filter_status == "failed":
        query += " AND r.correct = 0"
    elif filter_status == "passed":
        query += " AND r.correct = 1"

    query += " ORDER BY r.correct ASC, r.id LIMIT 200"

    questions = db.execute(query, params).fetchall()

    question_types = db.execute(
        "SELECT DISTINCT question_type FROM questions ORDER BY question_type"
    ).fetchall()

    return render_template(
        "run.html",
        run_id=run_id,
        run_meta=run_meta,
        overview=overview,
        by_type=by_type,
        by_category=by_category,
        questions=questions,
        question_types=question_types,
        filter_type=filter_type,
        filter_status=filter_status,
    )


@app.route("/question/<question_id>")
def question_detail(question_id: str):
    db = get_db()
    run_id = request.args.get("run_id")

    result = db.execute(
        """
        SELECT 
            r.*,
            q.question_type,
            q.question_text,
            q.gold_answer,
            q.benchmark
        FROM results r
        JOIN questions q ON r.question_id = q.question_id
        WHERE r.question_id = ?
        AND (? IS NULL OR r.run_id = ?)
        LIMIT 1
    """,
        (question_id, run_id, run_id),
    ).fetchone()

    if not result:
        return "Question not found", 404

    retrieved_nodes = db.execute(
        """
        SELECT * FROM retrieved_nodes
        WHERE result_id = ?
        ORDER BY retrieval_rank
    """,
        (result["id"],),
    ).fetchall()

    benchmark = result["benchmark"]

    stage_log = load_stage_log(question_id)
    deep_log = load_deep_log(question_id, run_id)

    pipeline = extract_pipeline_metrics(deep_log, stage_log)

    questions_meta = load_questions_metadata(benchmark)
    contexts = load_shared_contexts(benchmark)

    question_meta = questions_meta.get(question_id, {})
    shared_context_id = question_meta.get("shared_context_id", "")
    gold_answer = result["gold_answer"]

    sessions = []
    if shared_context_id and shared_context_id in contexts:
        turns = contexts[shared_context_id]
        raw_sessions = split_into_sessions(turns)
        for sess in raw_sessions:
            sess["contains_gold"] = check_gold_in_session(sess["turns"], gold_answer)
        sessions = raw_sessions

    return render_template(
        "question.html",
        result=result,
        retrieved_nodes=retrieved_nodes,
        sessions=sessions,
        deep_log=deep_log,
        stage_log=stage_log,
        pipeline=pipeline,
        question_meta=question_meta,
        run_id=run_id,
    )


@app.route("/api/annotate", methods=["POST"])
def annotate():
    data = request.json
    result_id = data.get("result_id")
    category = data.get("category")
    notes = data.get("notes")

    db = get_db()
    db.execute(
        """
        UPDATE results
        SET failure_category = ?, annotation_notes = ?, annotated = 1
        WHERE id = ?
    """,
        (category, notes, result_id),
    )
    db.commit()

    return jsonify({"success": True})


@app.route("/api/run/<run_id>/notes", methods=["POST"])
def update_run_notes(run_id: str):
    data = request.json
    notes = data.get("notes", "")

    db = get_db()
    db.execute(
        """INSERT INTO runs (run_id, system_name, notes, updated_at)
           VALUES (?, '', ?, CURRENT_TIMESTAMP)
           ON CONFLICT(run_id) DO UPDATE SET 
           notes = excluded.notes, updated_at = CURRENT_TIMESTAMP""",
        (run_id, notes),
    )
    db.commit()

    return jsonify({"success": True})


@app.route("/api/export/<run_id>")
def export_run(run_id: str):
    """Export run results as JSON."""
    db = get_db()

    results = db.execute(
        """
        SELECT 
            r.*,
            q.question_type,
            q.question_text,
            q.gold_answer,
            q.benchmark
        FROM results r
        JOIN questions q ON r.question_id = q.question_id
        WHERE r.run_id = ?
        ORDER BY r.id
    """,
        (run_id,),
    ).fetchall()

    export_data = {
        "run_id": run_id,
        "total": len(results),
        "correct": sum(1 for r in results if r["correct"]),
        "results": [dict(r) for r in results],
    }

    return jsonify(export_data)


@app.route("/benchmarks")
def benchmarks():
    """Show available benchmarks and download options."""
    benchmarks_info = [
        {
            "id": "personamem",
            "name": "PersonaMem",
            "description": "Personal memory evaluation with 589 questions across preference tracking, temporal reasoning, and cross-session understanding",
            "paper_url": "https://arxiv.org/abs/2410.12139",
            "download_url": "https://github.com/InnerNets/PersonaMem",
            "variants": ["32k", "128k"],
            "question_types": [
                "single_session_qa",
                "multi_session_qa",
                "preference_tracking",
                "temporal_reasoning",
                "adversarial_memory",
            ],
            "installed": (DATA_DIR / "personamem").exists(),
        },
        {
            "id": "longmemeval",
            "name": "LongMemEval",
            "description": "Long-term memory evaluation focusing on multi-session conversations",
            "paper_url": "https://arxiv.org/abs/2410.10813",
            "download_url": "https://github.com/xiaowu0162/LongMemEval",
            "variants": ["default"],
            "question_types": [
                "single-session-user",
                "temporal-reasoning",
                "knowledge-update",
            ],
            "installed": (DATA_DIR / "longmemeval").exists(),
        },
    ]

    return render_template("benchmarks.html", benchmarks=benchmarks_info)


if __name__ == "__main__":
    print(f"Starting Eval Explorer v2...")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Database: {get_db_path()}")
    print(f"\nOpen http://localhost:5001 in your browser\n")
    app.run(debug=True, port=5001)
