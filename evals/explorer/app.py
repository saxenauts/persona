"""
Eval Explorer - Flask app for browsing evaluation results.

Run with: python -m evals.explorer.app
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent / "data"


def get_db_path() -> Path:
    return RESULTS_DIR / "eval_analysis.db"


def get_db():
    db = sqlite3.connect(get_db_path())
    db.row_factory = sqlite3.Row
    return db


def load_shared_contexts(benchmark: str, variant: str = "32k") -> Dict[str, List[Dict]]:
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
    if benchmark == "personamem":
        path = DATA_DIR / "personamem" / f"questions_{variant}_{variant}.json"
        if not path.exists():
            return {}
        with open(path) as f:
            questions = json.load(f)
        return {f"personamem_{variant}_{i}": q for i, q in enumerate(questions)}
    return {}


@app.route("/")
def index():
    db = get_db()

    runs = db.execute("""
        SELECT 
            run_id,
            system_name,
            COUNT(*) as total,
            SUM(correct) as correct,
            ROUND(AVG(correct) * 100, 1) as accuracy,
            MIN(created_at) as started_at
        FROM results
        GROUP BY run_id, system_name
        ORDER BY started_at DESC
    """).fetchall()

    return render_template("index.html", runs=runs)


@app.route("/run/<run_id>")
def run_overview(run_id: str):
    db = get_db()

    overview = db.execute(
        """
        SELECT 
            run_id,
            system_name,
            COUNT(*) as total,
            SUM(correct) as correct,
            ROUND(AVG(correct) * 100, 1) as accuracy
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
            ROUND(AVG(r.correct) * 100, 1) as accuracy
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

    failures = db.execute(
        """
        SELECT 
            r.id,
            r.question_id,
            q.question_type,
            q.question_text,
            q.gold_answer,
            r.generated_answer,
            r.failure_category,
            r.retrieved_node_count
        FROM results r
        JOIN questions q ON r.question_id = q.question_id
        WHERE r.run_id = ? AND r.correct = 0
        ORDER BY r.failure_category, r.id
        LIMIT 100
    """,
        (run_id,),
    ).fetchall()

    return render_template(
        "run.html",
        run_id=run_id,
        overview=overview,
        by_type=by_type,
        by_category=by_category,
        failures=failures,
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
    questions_meta = load_questions_metadata(benchmark)
    contexts = load_shared_contexts(benchmark)

    question_meta = questions_meta.get(question_id, {})
    shared_context_id = question_meta.get("shared_context_id", "")

    sessions = []
    if shared_context_id and shared_context_id in contexts:
        turns = contexts[shared_context_id]
        current_session: List[Dict] = []
        for turn in turns:
            current_session.append(turn)
        if current_session:
            sessions.append(current_session)

    deep_log = None
    if run_id:
        for results_subdir in RESULTS_DIR.iterdir():
            if results_subdir.is_dir() and results_subdir.name.startswith("run_"):
                deep_logs_file = results_subdir / "deep_logs.jsonl"
                if deep_logs_file.exists():
                    with open(deep_logs_file) as f:
                        for line in f:
                            if line.strip():
                                log = json.loads(line)
                                if log.get("question_id") == question_id:
                                    deep_log = log
                                    break
                    if deep_log:
                        break

    return render_template(
        "question.html",
        result=result,
        retrieved_nodes=retrieved_nodes,
        sessions=sessions,
        deep_log=deep_log,
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


if __name__ == "__main__":
    print(f"Starting Eval Explorer...")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Database: {get_db_path()}")
    print(f"\nOpen http://localhost:5001 in your browser\n")
    app.run(debug=True, port=5001)
