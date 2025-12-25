"""
Evaluation Runner

Core orchestrator for running benchmark evaluations against memory systems.
"""

# =============================================================================
# CRITICAL: Apply graphiti_core reasoning.effort bugfix BEFORE any imports
# =============================================================================
# graphiti_core 0.24.3 incorrectly treats gpt-5.x as reasoning models and sends
# the 'reasoning.effort' parameter, which Azure OpenAI rejects with 400 error.
# This patch MUST be applied before any graphiti_core imports.
# =============================================================================
try:
    from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
    from graphiti_core.llm_client import OpenAIClient

    @staticmethod
    def _patched_supports_reasoning(model: str) -> bool:
        """Only enable reasoning.effort for actual reasoning models (o1, o3)."""
        return model.startswith(("o1-", "o3-"))

    AzureOpenAILLMClient._supports_reasoning_features = _patched_supports_reasoning
    OpenAIClient._supports_reasoning_features = _patched_supports_reasoning
    print(
        "[EvalRunner] Applied graphiti_core reasoning.effort bugfix for gpt-5.x models"
    )
except ImportError:
    pass  # graphiti_core not installed, skip patch
# =============================================================================

import json
import time
import threading
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    QuestionLog,
    IngestionLog,
    RetrievalLog,
    GenerationLog,
    EvaluationLog,
    VectorSearchLog,
    GraphTraversalLog,
    SeedNode,
    MemoryCreationStats,
)
from .adapters.base import MemorySystem
from .adapters.persona_adapter import PersonaAdapter
from .longmemeval.evaluate_qa import (
    get_anscheck_prompt,
    query_openai_with_retry,
    parse_judge_response,
)


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
    elif name in {"zep", "graphiti"}:
        from .adapters.zep_adapter import GraphitiAdapter

        return GraphitiAdapter()
    else:
        raise ValueError(f"Unknown adapter: {name}")


@dataclass
class EvaluationResult:
    """Result of evaluating a single question."""

    question_id: str
    question_type: str
    correct: Optional[bool]
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
        self._log_lock = threading.Lock()
        self._print_lock = threading.Lock()

        # Checkpointing support (prototype from supermemory research)
        self.checkpoint_enabled = os.getenv(
            "EVAL_CHECKPOINT_ENABLED", "false"
        ).lower() in {"1", "true", "yes"}
        self._checkpoint_path = Path(self.logger.run_dir) / "checkpoint.json"
        self._completed_questions: set = set()
        if self.checkpoint_enabled:
            self._load_checkpoint()

        self._print(f"✓ Evaluation runner initialized")
        self._print(f"  Run ID: {self.logger.run_id}")
        self._print(f"  Output: {self.logger.run_dir}")
        if self.checkpoint_enabled:
            self._print(
                f"  Checkpointing: ENABLED ({len(self._completed_questions)} completed)"
            )

    def _print(self, message: str, flush: bool = False) -> None:
        with self._print_lock:
            print(message, flush=flush)

    def _load_checkpoint(self) -> None:
        """Load checkpoint from disk if it exists."""
        if self._checkpoint_path.exists():
            try:
                with open(self._checkpoint_path) as f:
                    data = json.load(f)
                self._completed_questions = set(data.get("completed_questions", []))
                self._print(
                    f"  📂 Loaded checkpoint: {len(self._completed_questions)} questions completed"
                )
            except Exception as e:
                self._print(f"  ⚠️ Failed to load checkpoint: {e}")
                self._completed_questions = set()

    def _save_checkpoint(self, question_id: str) -> None:
        """Save checkpoint after completing a question (atomic write)."""
        if not self.checkpoint_enabled:
            return
        with self._log_lock:
            self._completed_questions.add(question_id)
            checkpoint_data = {
                "run_id": self.logger.run_id,
                "last_updated": datetime.now().isoformat(),
                "completed_questions": list(self._completed_questions),
                "total_completed": len(self._completed_questions),
            }
            try:
                import tempfile

                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=self._checkpoint_path.parent, suffix=".tmp"
                )
                with os.fdopen(tmp_fd, "w") as f:
                    json.dump(checkpoint_data, f, indent=2)
                os.replace(tmp_path, self._checkpoint_path)
            except Exception as e:
                self._print(f"  ⚠️ Failed to save checkpoint: {e}")

    def run(self) -> Dict[str, Any]:
        """
        Run evaluations for all configured benchmarks.

        Returns:
            Dictionary mapping benchmark names to results
        """
        results = {}

        # Run LongMemEval if configured
        if self.config.longmemeval:
            self._print("\n" + "=" * 60)
            self._print("Running LongMemEval Benchmark")
            self._print("=" * 60)
            results["longmemeval"] = self._run_benchmark(
                benchmark_name="longmemeval", config=self.config.longmemeval
            )

        # Run PersonaMem if configured
        if self.config.personamem:
            self._print("\n" + "=" * 60)
            self._print("Running PersonaMem Benchmark")
            self._print("=" * 60)
            results["personamem"] = self._run_benchmark(
                benchmark_name="personamem", config=self.config.personamem
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
                variant=getattr(config, "variant", None),
            )
            if getattr(config, "full_dataset", False) or not config.sample_sizes:
                questions = loader.load()
            else:
                questions = loader.stratified_sample(
                    sample_sizes=config.sample_sizes,
                    random_seed=self.config.random_seed,
                )

        self._print(f"\nLoaded {len(questions)} questions for {benchmark_name}")

        # Track results
        all_results: List[EvaluationResult] = []
        type_results: Dict[str, List[bool]] = {}

        # Run each adapter
        total_questions = len(questions)
        for adapter_name in self.config.adapters:
            self._print(f"\n--- Testing adapter: {adapter_name} ---")

            try:
                get_adapter(adapter_name)
            except Exception as e:
                self._print(f"Failed to load adapter {adapter_name}: {e}")
                continue

            if self.config.parallel_workers > 1:

                def evaluate_one(index: int, q):
                    adapter = get_adapter(adapter_name)
                    result = self._evaluate_question(
                        adapter=adapter,
                        question=q,
                        benchmark_name=benchmark_name,
                        verbose=False,
                    )
                    return index, q, result

                with ThreadPoolExecutor(
                    max_workers=self.config.parallel_workers
                ) as executor:
                    futures = [
                        executor.submit(evaluate_one, i, question)
                        for i, question in enumerate(questions)
                    ]

                    for future in as_completed(futures):
                        try:
                            idx, question, result = future.result()
                        except Exception as e:
                            self._print(f"  ✗ Error: {e}")
                            try:
                                adapter = get_adapter(adapter_name)
                                adapter.reset(f"cleanup_{int(time.time())}")
                            except Exception:
                                pass
                            continue

                        all_results.append(result)

                        qtype = result.question_type
                        if qtype not in type_results:
                            type_results[qtype] = []
                        if result.correct is not None:
                            type_results[qtype].append(result.correct)

                        if result.correct is None:
                            status = "·"
                        else:
                            status = "✓" if result.correct else "✗"
                        self._print(
                            f"[{idx + 1}/{total_questions}] {qtype}: "
                            f"{status} Answer: {result.generated_answer[:80]}..."
                        )
            else:
                adapter = get_adapter(adapter_name)
                for i, question in enumerate(questions):
                    # Checkpoint: skip already completed questions
                    if (
                        self.checkpoint_enabled
                        and question.question_id in self._completed_questions
                    ):
                        self._print(
                            f"\n[{i + 1}/{total_questions}] {question.question_type}: SKIPPED (checkpointed)"
                        )
                        continue

                    self._print(
                        f"\n[{i + 1}/{total_questions}] {question.question_type}: "
                        f"{question.question[:50]}..."
                    )

                    try:
                        result = self._evaluate_question(
                            adapter=adapter,
                            question=question,
                            benchmark_name=benchmark_name,
                            verbose=True,
                        )
                        all_results.append(result)

                        # Save checkpoint after each successful evaluation
                        self._save_checkpoint(question.question_id)

                        qtype = result.question_type
                        if qtype not in type_results:
                            type_results[qtype] = []
                        if result.correct is not None:
                            type_results[qtype].append(result.correct)

                        if result.correct is None:
                            status = "·"
                        else:
                            status = "✓" if result.correct else "✗"
                        self._print(
                            f"  {status} Answer: {result.generated_answer[:80]}..."
                        )

                    except Exception as e:
                        self._print(f"  ✗ Error: {e}")
                        continue

        # Calculate metrics
        total = len(all_results)
        judged_results = [r for r in all_results if r.correct is not None]
        judged_total = len(judged_results)
        correct = sum(1 for r in judged_results if r.correct)
        skipped = total - judged_total

        type_accuracies = {}
        for qtype, results_list in type_results.items():
            type_accuracies[qtype] = {
                "accuracy": sum(results_list) / len(results_list)
                if results_list
                else 0,
                "correct": sum(results_list),
                "count": len(results_list),
            }

        return {
            "overall_accuracy": correct / judged_total if judged_total > 0 else 0,
            "total_questions": total,
            "judged_questions": judged_total,
            "skipped_questions": skipped,
            "correct": correct,
            "type_accuracies": type_accuracies,
        }

    def _evaluate_question(
        self,
        adapter: MemorySystem,
        question: Union[LongMemEvalQuestion, PersonaMemQuestion],
        benchmark_name: str,
        verbose: bool = True,
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
            total_chars = sum(len(s.get("content", "")) for s in sessions)
            if verbose:
                self._print(
                    f"    📥 Ingesting {len(sessions)} sessions (~{total_chars // 1000}k chars)...",
                    flush=True,
                )

            # Ingest sessions
            start_ingest = time.time()
            adapter.add_sessions(user_id, sessions)
            ingest_time_ms = (time.time() - start_ingest) * 1000
            if verbose:
                self._print(
                    f"    ✓ Ingestion complete ({ingest_time_ms / 1000:.1f}s)",
                    flush=True,
                )

            # Query
            if verbose:
                self._print(f"    🔍 Retrieving context...", flush=True)
            start_query = time.time()
            query_text = question.question
            if benchmark_name == "longmemeval":
                include_date = os.getenv(
                    "LONGMEMEVAL_INCLUDE_DATE", "true"
                ).lower() in {
                    "1",
                    "true",
                    "yes",
                }
                if include_date and question.question_date:
                    query_text = f"(date: {question.question_date}) {query_text}"
            elif benchmark_name == "personamem":
                query_text = self._format_personamem_query(question)
            generated_answer = adapter.query(user_id, query_text)
            query_time_ms = (time.time() - start_query) * 1000
            if verbose:
                self._print(
                    f"    ✓ Retrieval complete ({query_time_ms / 1000:.1f}s)",
                    flush=True,
                )

            # Evaluate answer
            skip_judge = self.config.skip_judge and benchmark_name == "longmemeval"
            if skip_judge:
                gold_answer = question.answer
                correct = None
                judge_response = "skipped"
                judge_time_ms = 0.0
                if verbose:
                    self._print("    ⚖️ Judge skipped (deferred)", flush=True)
            else:
                if verbose:
                    self._print(f"    ⚖️ Running judge...", flush=True)
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
                if verbose:
                    self._print(
                        f"    ✓ Judge: {judge_response} ({judge_time_ms / 1000:.1f}s)",
                        flush=True,
                    )

            # Log result
            query_stats = getattr(adapter, "last_query_stats", None)
            ingest_stats = getattr(adapter, "last_ingest_stats", None)
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
                sessions_count=len(sessions),
                query_text=query_text,
                ingest_stats=ingest_stats,
                query_stats=query_stats,
            )

            return EvaluationResult(
                question_id=question.question_id,
                question_type=question.question_type,
                correct=correct,
                generated_answer=generated_answer,
                gold_answer=gold_answer,
                ingestion_time_ms=ingest_time_ms,
                query_time_ms=query_time_ms,
                judge_response=judge_response,
            )

        finally:
            # Cleanup
            try:
                adapter.reset(user_id)
            except:
                pass

    def _prepare_longmemeval_sessions(
        self, question: LongMemEvalQuestion
    ) -> List[Dict]:
        """Convert LongMemEval haystack to session format."""
        sessions = []

        for date, session_turns in zip(
            question.haystack_dates, question.haystack_sessions
        ):
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

    def _format_personamem_query(self, question: PersonaMemQuestion) -> str:
        """Format PersonaMem multiple-choice prompt, optionally truncated by env."""
        letters = ["a", "b", "c", "d"]
        option_texts = {
            letter: question.options.get(letter, "").strip() for letter in letters
        }
        question_text = question.question.strip()
        max_chars_env = os.getenv("PERSONAMEM_PROMPT_MAX_CHARS")
        max_chars = int(max_chars_env) if max_chars_env else None

        def truncate(text: str, max_len: int) -> str:
            if len(text) <= max_len:
                return text
            return text[: max_len - 3].rstrip() + "..."

        def build(q_text: str, opts: Dict[str, str]) -> str:
            options_str = " ".join(
                f"({letter}) {opts[letter]}" for letter in letters if opts.get(letter)
            )
            return (
                f"Question: {q_text}\n"
                f"Options: {options_str}\n"
                "Answer with only the letter (a/b/c/d)."
            )

        prompt = build(question_text, option_texts)
        if max_chars is None or max_chars <= 0:
            return prompt
        if len(prompt) <= max_chars:
            return prompt

        for max_opt_len in [200, 160, 120, 100, 80, 60]:
            truncated_opts = {
                letter: truncate(option_texts[letter], max_opt_len)
                for letter in letters
                if option_texts.get(letter)
            }
            prompt = build(question_text, truncated_opts)
            if len(prompt) <= max_chars:
                return prompt

        truncated_question = truncate(question_text, 200)
        for max_opt_len in [80, 60, 40]:
            truncated_opts = {
                letter: truncate(option_texts[letter], max_opt_len)
                for letter in letters
                if option_texts.get(letter)
            }
            prompt = build(truncated_question, truncated_opts)
            if len(prompt) <= max_chars:
                return prompt

        return prompt[:max_chars]

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
            abstention=is_abstention,
        )

        judge_response = query_openai_with_retry(prompt)
        correct = parse_judge_response(judge_response)

        return correct, judge_response

    def _judge_retrieval_quality(
        self, question: str, gold_answer: str, retrieved_context: str
    ) -> dict:
        """
        Evaluate retrieval quality using LLM-as-Judge (Supermemory pattern).

        Returns:
            dict with 'relevant' (bool), 'score' (0-1), and 'reasoning' (str)
        """
        if not retrieved_context or not retrieved_context.strip():
            return {
                "relevant": False,
                "score": 0.0,
                "reasoning": "No context retrieved",
            }

        prompt = f"""I will give you a question, the correct answer, and retrieved context from a memory system.
Your task is to determine if the retrieved context contains information that could help answer the question correctly.

Question: {question}

Correct Answer: {gold_answer}

Retrieved Context:
{retrieved_context[:2000]}  # Truncate for token efficiency

Does the retrieved context contain information relevant to answering the question correctly?
Answer with ONLY one of: YES or NO"""

        try:
            judge_response = query_openai_with_retry(prompt)
            relevant = parse_judge_response(judge_response)
            return {
                "relevant": relevant,
                "score": 1.0 if relevant else 0.0,
                "reasoning": judge_response,
            }
        except Exception as e:
            return {"relevant": False, "score": 0.0, "reasoning": f"Error: {e}"}

    def _evaluate_personamem(
        self, question: PersonaMemQuestion, generated_answer: str
    ) -> tuple[bool, str]:
        """Evaluate PersonaMem using exact match on option letter."""
        generated_lower = generated_answer.lower().strip()
        cleaned = re.sub(
            r"^\s*(answer|option)\s*[:\-]*\s*", "", generated_lower
        ).strip()

        tokens = cleaned.split()
        if tokens:
            candidate = tokens[0].strip("().")
            if candidate in ["a", "b", "c", "d"]:
                if len(tokens) == 1 or tokens[0].endswith((")", ".", ":", "-")):
                    correct = candidate == question.correct_answer.lower()
                    return (
                        correct,
                        f"Extracted: {candidate}, Expected: {question.correct_answer}",
                    )

        for option in ["a", "b", "c", "d"]:
            if f"({option})" in cleaned:
                correct = option == question.correct_answer.lower()
                return (
                    correct,
                    f"Extracted: {option}, Expected: {question.correct_answer}",
                )

        correct_text = question.options.get(question.correct_answer.lower(), "")
        if correct_text and correct_text.lower() in generated_lower:
            return True, "Contains correct answer text"

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
        sessions_count: int,
        query_text: str,
        ingest_stats: Optional[dict] = None,
        query_stats: Optional[dict] = None,
    ):
        """Log question evaluation to deep logger."""
        safe_gold_answer = "" if gold_answer is None else str(gold_answer)
        safe_generated_answer = (
            "" if generated_answer is None else str(generated_answer)
        )

        type_counts = {}
        if ingest_stats and isinstance(ingest_stats, dict):
            type_counts = ingest_stats.get("memories_created_by_type", {}) or {}
        total_memories = 0
        if ingest_stats and isinstance(ingest_stats, dict):
            total_memories = ingest_stats.get("memories_created", 0) or 0
        links_created = 0
        if ingest_stats and isinstance(ingest_stats, dict):
            links_created = ingest_stats.get("links_created", 0) or 0

        retrieval_stats = {}
        if query_stats and isinstance(query_stats, dict):
            retrieval_stats = query_stats.get("retrieval", {}) or {}
        vector_stats = retrieval_stats.get("vector_search", {}) or {}
        graph_stats = retrieval_stats.get("graph_traversal", {}) or {}

        prompt_tokens = 0
        completion_tokens = 0
        model_name = "unknown"
        temperature = 0
        retrieval_duration_ms = query_time_ms
        generation_duration_ms = 0
        if query_stats and isinstance(query_stats, dict):
            prompt_tokens = query_stats.get("prompt_tokens") or 0
            completion_tokens = query_stats.get("completion_tokens") or 0
            model_name = query_stats.get("model") or "unknown"
            temperature = query_stats.get("temperature") or 0
            retrieval_duration_ms = query_stats.get("retrieval_ms") or query_time_ms
            generation_duration_ms = query_stats.get("generation_ms") or 0

        ingest_timings = {}
        if ingest_stats and isinstance(ingest_stats, dict):
            ingest_timings = ingest_stats.get("timings_ms", {}) or {}

        question_log = QuestionLog(
            question_id=question.question_id,
            user_id=user_id,
            benchmark=benchmark_name,
            question_type=question.question_type,
            question=question.question,
            ingestion=IngestionLog(
                duration_ms=ingest_time_ms,
                sessions_count=sessions_count,
                memories_created=MemoryCreationStats(
                    episodes=type_counts.get("episode", 0),
                    psyche=type_counts.get("psyche", 0),
                    goals=type_counts.get("goal", 0),
                    events=type_counts.get("event", 0),
                ),
                nodes_created=total_memories,
                relationships_created=links_created,
                embeddings_generated=total_memories,
                extract_ms=ingest_timings.get("extract"),
                embed_ms=ingest_timings.get("embed"),
                persist_ms=ingest_timings.get("persist"),
                total_ms=ingest_timings.get("total"),
            ),
            retrieval=RetrievalLog(
                query=query_text,
                duration_ms=retrieval_duration_ms,
                vector_search=VectorSearchLog(
                    top_k=vector_stats.get("top_k", 5),
                    seeds=vector_stats.get("seeds", []),
                    duration_ms=vector_stats.get("duration_ms", 0),
                ),
                graph_traversal=GraphTraversalLog(
                    max_hops=graph_stats.get("max_hops", 2),
                    nodes_visited=graph_stats.get("nodes_visited", 0),
                    relationships_traversed=graph_stats.get(
                        "relationships_traversed", 0
                    ),
                    final_ranked_nodes=graph_stats.get("final_ranked_nodes", []),
                    duration_ms=graph_stats.get("duration_ms", 0),
                ),
                context_size_tokens=prompt_tokens,
                retrieved_context=retrieval_stats.get("context_preview"),
            ),
            generation=GenerationLog(
                duration_ms=generation_duration_ms,
                model=model_name,
                temperature=temperature,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                answer=safe_generated_answer,
            ),
            evaluation=EvaluationLog(
                gold_answer=safe_gold_answer,
                correct=correct,
                judge_response=judge_response,
                judge_model=(
                    os.getenv("EVAL_JUDGE_MODEL", "gpt-4o")
                    if benchmark_name == "longmemeval" and correct is not None
                    else None
                ),
                score_type=(
                    "deferred"
                    if benchmark_name == "longmemeval" and correct is None
                    else (
                        "binary" if benchmark_name == "longmemeval" else "exact_match"
                    )
                ),
            ),
        )

        with self._log_lock:
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
                    metadata=q.get("metadata", {}),
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
                    metadata=q.get("metadata", {}),
                )
                for q in data
            ]
