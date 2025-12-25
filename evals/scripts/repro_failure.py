import sys
import os
import argparse
import asyncio

# Ensure we can import from local modules
sys.path.append(os.getcwd())

from evals.adapters.zep_adapter import GraphitiAdapter
from evals.runner import EvaluationRunner
from evals.loaders.longmemeval_loader import LongMemEvalLoader
from evals.config import EvalConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-id", default="00ca467f")
    args = parser.parse_args()

    print(f"Loading data for question {args.question_id}...")
    loader = LongMemEvalLoader()
    questions = loader.load()
    
    try:
        question = next(q for q in questions if q.question_id == args.question_id)
    except StopIteration:
        print(f"Question {args.question_id} not found!")
        return

    # Create dummy config with valid arguments
    config = EvalConfig(
        adapters=["graphiti"],
        output_dir="evals/results/repro_debug"
    )
    runner = EvaluationRunner(config)
    
    # Use runner helper to format sessions
    sessions = runner._prepare_longmemeval_sessions(question)
    
    print(f"Prepared {len(sessions)} sessions.")
    for i, s in enumerate(sessions):
        print(f"Session {i}: date={s['date']}, content_len={len(s['content'])}")
    
    adapter = GraphitiAdapter()
    
    # Ensure clean state for this repro
    print("Resetting adapter state for user...")
    adapter.reset(question.question_id)
    
    print("Starting ingestion...")
    # This will use the new logging we added to zep_adapter.py (printed to stdout)
    adapter.add_sessions(question.question_id, sessions)
    
    print("Ingestion complete attempt finished.")

if __name__ == "__main__":
    main()
