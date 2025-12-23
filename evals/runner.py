"""
Evaluation Runner

Core orchestrator for running benchmark evaluations against memory systems.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from .config import EvalConfig
from .loaders.unified_loader import UnifiedBenchmarkLoader
from .loaders.longmemeval_loader import LongMemEvalQuestion
from .loaders.personamem_loader import PersonaMemQuestion
from .logging.deep_logger import DeepLogger
from .logging.log_schema import (
    QuestionLog, IngestionLog, RetrievalLog, GenerationLog,
    EvaluationLog, VectorSearchLog, GraphTraversalLog,
    SeedNode, MemoryCreationStats
)
from .adapters.base import MemorySystem
from .adapters.persona_adapter import PersonaAdapter
from .longmemeval.evaluate_qa import get_anscheck_prompt, query_openai_with_retry, parse_judge_response


# Registry of available adapters
ADAPTERS = {
    "persona": PersonaAdapter,
}

# Lazy imports for optional adapters
def get_adapter(name: str) -> MemorySystem:
    """Get adapter instance by name."""
    if name == "persona":
        return PersonaAdapter()
    elif name == "mem0":
        from .adapters.mem0_adapter import Mem0Adapter
        return Mem0Adapter()
    elif name == "zep":
        from .adapters.zep_adapter import ZepAdapter
        return ZepAdapter()
    else:
        raise ValueError(f"Unknown adapter: {name}")


@dataclass
class EvaluationResult:
    """Result of evaluating a single question."""
    question_id: str
    question_type: str
    correct: bool
    generated_answer: str
    gold_answer: str
    ingestion_time_ms: float
    query_time_ms: float
    judge_response: str


class EvaluationRunner:
    """
    Main evaluation runner that orchestrates:
    1. Loading questions from benchmarks
    2. Running each question through memory adapters
    3. Evaluating answers with LLM judge
    4. Logging results
    """

    def __init__(self, config: EvalConfig, use_golden_set: bool = False):
        """
        Initialize evaluation runner.
        
        Args:
            config: Evaluation configuration
            use_golden_set: If True, use pre-generated golden sets instead of sampling
        """
        self.config = config
        self.use_golden_set = use_golden_set
        self.logger = DeepLogger(output_dir=config.output_dir)
        
        print(f"✓ Evaluation runner initialized")
        print(f"  Run ID: {self.logger.run_id}")
        print(f"  Output: {self.logger.run_dir}")

    def run(self) -> Dict[str, Any]:
        """
        Run evaluations for all configured benchmarks.
        
        Returns:
            Dictionary mapping benchmark names to results
        """
        results = {}
        
        # Run LongMemEval if configured
        if self.config.longmemeval:
            print("\n" + "="*60)
            print("Running LongMemEval Benchmark")
            print("="*60)
            results["longmemeval"] = self._run_benchmark(
                benchmark_name="longmemeval",
                config=self.config.longmemeval
            )
        
        # Run PersonaMem if configured
        if self.config.personamem:
            print("\n" + "="*60)
            print("Running PersonaMem Benchmark")
            print("="*60)
            results["personamem"] = self._run_benchmark(
                benchmark_name="personamem",
                config=self.config.personamem
            )
        
        # Save final summary
        self.logger.save_summary()
        
        return results

    def _run_benchmark(self, benchmark_name: str, config) -> Dict[str, Any]:
        """Run a single benchmark."""
        
        # Load questions
        if self.use_golden_set:
            questions = self._load_golden_set(benchmark_name)
        else:
            loader = UnifiedBenchmarkLoader(
                benchmark=benchmark_name,
                data_dir=config.source,
                variant=getattr(config, 'variant', None)
            )
            questions = loader.stratified_sample(
                sample_sizes=config.sample_sizes,
                random_seed=self.config.random_seed
            )
        
        print(f"\nLoaded {len(questions)} questions for {benchmark_name}")
        
        # Track results
        all_results: List[EvaluationResult] = []
        type_results: Dict[str, List[bool]] = {}
        
        # Run each adapter
        for adapter_name in self.config.adapters:
            print(f"\n--- Testing adapter: {adapter_name} ---")
            
            try:
                adapter = get_adapter(adapter_name)
            except Exception as e:
                print(f"Failed to load adapter {adapter_name}: {e}")
                continue
            
            # Evaluate each question
            for i, question in enumerate(questions):
                print(f"\n[{i+1}/{len(questions)}] {question.question_type}: {question.question[:50]}...")
                
                try:
                    result = self._evaluate_question(
                        adapter=adapter,
                        question=question,
                        benchmark_name=benchmark_name
                    )
                    all_results.append(result)
                    
                    # Track by type
                    qtype = result.question_type
                    if qtype not in type_results:
                        type_results[qtype] = []
                    type_results[qtype].append(result.correct)
                    
                    status = "✓" if result.correct else "✗"
                    print(f"  {status} Answer: {result.generated_answer[:80]}...")
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    continue
        
        # Calculate metrics
        total = len(all_results)
        correct = sum(1 for r in all_results if r.correct)
        
        type_accuracies = {}
        for qtype, results_list in type_results.items():
            type_accuracies[qtype] = {
                "accuracy": sum(results_list) / len(results_list) if results_list else 0,
                "correct": sum(results_list),
                "count": len(results_list)
            }
        
        return {
            "overall_accuracy": correct / total if total > 0 else 0,
            "total_questions": total,
            "correct": correct,
            "type_accuracies": type_accuracies
        }

    def _evaluate_question(
        self,
        adapter: MemorySystem,
        question: Union[LongMemEvalQuestion, PersonaMemQuestion],
        benchmark_name: str
    ) -> EvaluationResult:
        """Evaluate a single question."""
        
        # Generate unique user ID for isolation
        user_id = f"eval_{question.question_id}_{int(time.time())}"
        
        try:
            # Reset adapter state
            adapter.reset(user_id)
            
            # Prepare sessions for ingestion
            if benchmark_name == "longmemeval":
                sessions = self._prepare_longmemeval_sessions(question)
            else:
                sessions = self._prepare_personamem_sessions(question)
            
            # Calculate content size for progress
            total_chars = sum(len(s.get('content', '')) for s in sessions)
            print(f"    📥 Ingesting {len(sessions)} sessions (~{total_chars//1000}k chars)...", flush=True)
            
            # Ingest sessions
            start_ingest = time.time()
            adapter.add_sessions(user_id, sessions)
            ingest_time_ms = (time.time() - start_ingest) * 1000
            print(f"    ✓ Ingestion complete ({ingest_time_ms/1000:.1f}s)", flush=True)
            
            # Query
            print(f"    🔍 Retrieving context...", flush=True)
            start_query = time.time()
            generated_answer = adapter.query(user_id, question.question)
            query_time_ms = (time.time() - start_query) * 1000
            print(f"    ✓ Retrieval complete ({query_time_ms/1000:.1f}s)", flush=True)
            
            # Evaluate answer
            print(f"    ⚖️ Running judge...", flush=True)
            start_judge = time.time()
            if benchmark_name == "longmemeval":
                gold_answer = question.answer
                correct, judge_response = self._evaluate_longmemeval(
                    question, generated_answer
                )
            else:
                gold_answer = question.correct_answer
                correct, judge_response = self._evaluate_personamem(
                    question, generated_answer
                )
            judge_time_ms = (time.time() - start_judge) * 1000
            print(f"    ✓ Judge: {judge_response} ({judge_time_ms/1000:.1f}s)", flush=True)
            
            # Log result
            self._log_question(
                question=question,
                user_id=user_id,
                benchmark_name=benchmark_name,
                generated_answer=generated_answer,
                gold_answer=gold_answer,
                correct=correct,
                judge_response=judge_response,
                ingest_time_ms=ingest_time_ms,
                query_time_ms=query_time_ms, 
                sessions_count=len(sessions)
            )
            
            return EvaluationResult(
                question_id=question.question_id,
                question_type=question.question_type,
                correct=correct,
                generated_answer=generated_answer,
                gold_answer=gold_answer,
                ingestion_time_ms=ingest_time_ms,
                query_time_ms=query_time_ms,
                judge_response=judge_response
            )
            
        finally:
            # Cleanup
            try:
                adapter.reset(user_id)
            except:
                pass

    def _prepare_longmemeval_sessions(self, question: LongMemEvalQuestion) -> List[Dict]:
        """Convert LongMemEval haystack to session format."""
        sessions = []
        
        for date, session_turns in zip(question.haystack_dates, question.haystack_sessions):
            # Combine turns into conversation
            content_parts = []
            for turn in session_turns:
                role = turn.get("role", "user")
                text = turn.get("content", "")
                content_parts.append(f"{role.capitalize()}: {text}")
            
            content = "\n".join(content_parts)
            sessions.append({"date": date, "content": content})
        
        return sessions

    def _prepare_personamem_sessions(self, question: PersonaMemQuestion) -> List[Dict]:
        """Convert PersonaMem context to session format."""
        # PersonaMem has context as a single string
        return [{"date": "unknown", "content": question.context}]

    def _evaluate_longmemeval(
        self, question: LongMemEvalQuestion, generated_answer: str
    ) -> tuple[bool, str]:
        """Evaluate using LongMemEval GPT judge."""
        is_abstention = question.is_abstention
        
        prompt = get_anscheck_prompt(
            task=question.question_type,
            question=question.question,
            answer=question.answer,
            response=generated_answer,
            abstention=is_abstention
        )
        
        judge_response = query_openai_with_retry(prompt)
        correct = parse_judge_response(judge_response)
        
        return correct, judge_response

    def _evaluate_personamem(
        self, question: PersonaMemQuestion, generated_answer: str
    ) -> tuple[bool, str]:
        """Evaluate PersonaMem using exact match on option letter."""
        # Extract option letter from answer
        generated_lower = generated_answer.lower().strip()
        
        # Check for exact match of option letter
        for option in ['a', 'b', 'c', 'd']:
            if generated_lower.startswith(option) or f"({option})" in generated_lower:
                correct = (option == question.correct_answer.lower())
                return correct, f"Extracted: {option}, Expected: {question.correct_answer}"
        
        # If no clear option, check if answer contains the correct option text
        correct_text = question.options.get(question.correct_answer.lower(), "")
        if correct_text and correct_text.lower() in generated_lower:
            return True, f"Contains correct answer text"
        
        return False, f"Could not extract option from: {generated_answer[:50]}"

    def _log_question(
        self,
        question,
        user_id: str,
        benchmark_name: str,
        generated_answer: str,
        gold_answer: str,
        correct: bool,
        judge_response: str,
        ingest_time_ms: float,
        query_time_ms: float,
        sessions_count: int
    ):
        """Log question evaluation to deep logger."""
        question_log = QuestionLog(
            question_id=question.question_id,
            user_id=user_id,
            benchmark=benchmark_name,
            question_type=question.question_type,
            question=question.question,
            ingestion=IngestionLog(
                duration_ms=ingest_time_ms,
                sessions_count=sessions_count,
                memories_created=MemoryCreationStats(episodes=0, psyche=0, goals=0),
                nodes_created=0,
                relationships_created=0,
                embeddings_generated=0
            ),
            retrieval=RetrievalLog(
                query=question.question,
                duration_ms=query_time_ms,
                vector_search=VectorSearchLog(top_k=5, seeds=[], duration_ms=0),
                graph_traversal=GraphTraversalLog(
                    max_hops=2,
                    nodes_visited=0,
                    relationships_traversed=0,
                    final_ranked_nodes=[],
                    duration_ms=0
                ),
                context_size_tokens=0
            ),
            generation=GenerationLog(
                duration_ms=0,
                model="unknown",
                temperature=0,
                prompt_tokens=0,
                completion_tokens=0,
                answer=generated_answer
            ),
            evaluation=EvaluationLog(
                gold_answer=gold_answer,
                correct=correct,
                judge_response=judge_response,
                judge_model="gpt-4o",
                score_type="binary" if benchmark_name == "longmemeval" else "exact_match"
            )
        )
        
        self.logger.log_question(question_log)

    def _load_golden_set(self, benchmark_name: str) -> List:
        """Load pre-generated golden set."""
        golden_path = Path(f"evals/data/golden_sets/{benchmark_name}_golden_set.json")
        
        if not golden_path.exists():
            raise FileNotFoundError(
                f"Golden set not found: {golden_path}. "
                f"Run 'python evals/scripts/generate_golden_sets.py' first."
            )
        
        with open(golden_path) as f:
            data = json.load(f)
        
        # Convert to question objects
        if benchmark_name == "longmemeval":
            from .loaders.longmemeval_loader import LongMemEvalQuestion
            return [
                LongMemEvalQuestion(
                    question_id=q["question_id"],
                    question_type=q["question_type"],
                    question=q["question"],
                    answer=q["answer"],
                    question_date=q.get("question_date", ""),
                    haystack_dates=q.get("haystack_dates", []),
                    haystack_session_ids=q.get("haystack_session_ids", []),
                    haystack_sessions=q.get("haystack_sessions", []),
                    is_abstention=q.get("is_abstention", False),
                    metadata=q.get("metadata", {})
                )
                for q in data
            ]
        else:
            from .loaders.personamem_loader import PersonaMemQuestion
            return [
                PersonaMemQuestion(
                    question_id=q["question_id"],
                    question_type=q["question_type"],
                    question=q["question"],
                    options=q.get("options", {}),
                    correct_answer=q.get("correct_answer", ""),
                    context=q.get("context", ""),
                    metadata=q.get("metadata", {})
                )
                for q in data
            ]
