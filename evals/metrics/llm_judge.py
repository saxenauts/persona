"""
LLM-as-Judge metrics for semantic evaluation.

Design Principles (from Hamel/Eugene research):
- Binary YES/NO preferred over 1-5 scales
- Task-specific prompts for different question types
- Strict parsing with ambiguity logging
- Async by default for throughput
"""

from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from ..core.models import TestCase, QueryResult, MetricResult, MetricKind, ScoreType
from ..core.interfaces import AdapterCapabilities
from .base import BaseMetric

logger = logging.getLogger(__name__)


# === Prompt Templates (from LongMemEval) ===

PROMPT_TEMPLATES = {
    "single-session-user": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Question: {question}

Correct Answer: {reference}

Model Response: {response}

Is the model response correct? Answer yes or no only.""",
    "single-session-assistant": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Question: {question}

Correct Answer: {reference}

Model Response: {response}

Is the model response correct? Answer yes or no only.""",
    "multi-session": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Question: {question}

Correct Answer: {reference}

Model Response: {response}

Is the model response correct? Answer yes or no only.""",
    "temporal-reasoning": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.

Question: {question}

Correct Answer: {reference}

Model Response: {response}

Is the model response correct? Answer yes or no only.""",
    "knowledge-update": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Question: {question}

Correct Answer: {reference}

Model Response: {response}

Is the model response correct? Answer yes or no only.""",
    "single-session-preference": """I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.

Question: {question}

Rubric: {reference}

Model Response: {response}

Is the model response correct? Answer yes or no only.""",
    # Abstention prompt for unanswerable questions
    "abstention": """I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.

Question: {question}

Explanation: {reference}

Model Response: {response}

Does the model correctly identify the question as unanswerable? Answer yes or no only.""",
    # Generic fallback
    "default": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer or is semantically equivalent. Otherwise, answer no.

Question: {question}

Correct Answer: {reference}

Model Response: {response}

Is the model response correct? Answer yes or no only.""",
}


def parse_yes_no(response: str) -> tuple[bool, str]:
    """
    Parse LLM response to binary YES/NO.

    Returns:
        (passed, reason) tuple
    """
    if not response:
        return False, "Empty judge response"

    cleaned = response.strip().upper()

    if cleaned in ("YES", "YES."):
        return True, "Judge said YES"
    elif cleaned in ("NO", "NO."):
        return False, "Judge said NO"
    else:
        # Check for yes/no anywhere in response (lenient fallback)
        lower = response.lower()
        if "yes" in lower and "no" not in lower:
            return True, f"Judge leaned YES: {response[:50]}"
        elif "no" in lower and "yes" not in lower:
            return False, f"Judge leaned NO: {response[:50]}"
        else:
            logger.warning(
                f"Ambiguous judge response: '{response}' - defaulting to False"
            )
            return False, f"Ambiguous response: {response[:50]}"


class LLMBinaryJudge(BaseMetric):
    """
    LLM-as-judge metric using binary YES/NO evaluation.

    Features:
    - Task-specific prompts (LongMemEval methodology)
    - Abstention detection for unanswerable questions
    - Strict YES/NO parsing
    - Configurable model and temperature

    Usage:
        judge = LLMBinaryJudge()
        result = await judge.evaluate(test_case, query_result, resources={"llm": client})

    Resources:
        - llm: Async LLM client with chat(messages, temperature, max_tokens) method
        - llm_model (optional): Model name override
    """

    name = "llm_binary_judge"
    kind: MetricKind = "generation"
    score_type: ScoreType = "binary"

    def __init__(
        self,
        *,
        default_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 10,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.default_model = default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

    def required_capabilities(self) -> AdapterCapabilities:
        """No special adapter capabilities needed."""
        return AdapterCapabilities()

    def _get_prompt(self, test_case: TestCase, response: str) -> str:
        """Get the appropriate prompt template for this test case."""
        # Check if this is an abstention question
        is_abstention = test_case.metadata.get("is_abstention", False)
        if is_abstention or "abstention" in test_case.tags:
            template_key = "abstention"
        else:
            template_key = test_case.question_type or "default"

        template = PROMPT_TEMPLATES.get(template_key, PROMPT_TEMPLATES["default"])

        return template.format(
            question=test_case.query,
            reference=test_case.reference_answer or "",
            response=response,
        )

    async def evaluate(
        self,
        test_case: TestCase,
        query_result: QueryResult,
        *,
        resources: Mapping[str, Any],
    ) -> MetricResult:
        """
        Evaluate using LLM-as-judge.

        Resources required:
            - llm: Async callable(prompt: str) -> str, or client with chat() method
            - llm_model (optional): Model name to use
        """
        llm = resources.get("llm")
        if llm is None:
            return MetricResult(
                metric=self.name,
                kind=self.kind,
                score_type=self.score_type,
                score=0.0,
                passed=False,
                reason="No LLM client provided in resources",
            )

        model = resources.get("llm_model", self.default_model)
        prompt = self._get_prompt(test_case, query_result.answer)

        # Call LLM with retries
        judge_response = ""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                judge_response = await self._call_llm(llm, prompt, model)
                break
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(min(2**attempt, 30))

        if not judge_response and last_error:
            return MetricResult(
                metric=self.name,
                kind=self.kind,
                score_type=self.score_type,
                score=0.0,
                passed=False,
                reason=f"LLM call failed after {self.max_retries} attempts: {last_error}",
                artifacts={"error": str(last_error)},
            )

        # Parse response
        passed, reason = parse_yes_no(judge_response)

        return MetricResult(
            metric=self.name,
            kind=self.kind,
            score_type=self.score_type,
            score=1.0 if passed else 0.0,
            passed=passed,
            reason=reason,
            artifacts={
                "prompt": prompt,
                "judge_response": judge_response,
                "model": model,
                "question_type": test_case.question_type,
                "is_abstention": test_case.metadata.get("is_abstention", False),
            },
        )

    async def _call_llm(self, llm: Any, prompt: str, model: str) -> str:
        """Call the LLM client. Supports multiple client interfaces."""
        # If llm is a simple callable
        if callable(llm) and not hasattr(llm, "chat"):
            result = await asyncio.wait_for(
                llm(prompt),
                timeout=self.timeout,
            )
            return str(result).strip()

        # If llm has a chat() method (our standard interface)
        if hasattr(llm, "chat"):
            messages = [{"role": "user", "content": prompt}]
            response = await asyncio.wait_for(
                llm.chat(
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                ),
                timeout=self.timeout,
            )
            # Handle different response formats
            if hasattr(response, "content"):
                return response.content.strip()
            elif isinstance(response, str):
                return response.strip()
            elif hasattr(response, "choices"):
                return response.choices[0].message.content.strip()
            return str(response).strip()

        raise ValueError(f"Unsupported LLM client type: {type(llm)}")


class AbstentionAccuracy(BaseMetric):
    """
    Specialized metric for abstention questions.

    Checks if the model correctly identifies unanswerable questions.
    Uses keywords to detect abstention responses (no LLM judge needed).

    Design: Fast, deterministic check for abstention detection.
    LLMBinaryJudge handles semantic correctness separately.
    """

    name = "abstention_accuracy"
    kind: MetricKind = "generation"
    score_type: ScoreType = "binary"

    # Keywords indicating the model is abstaining/declining
    ABSTENTION_KEYWORDS = [
        "i don't have",
        "i do not have",
        "cannot find",
        "no information",
        "not mentioned",
        "not specified",
        "unable to find",
        "don't know",
        "do not know",
        "not sure",
        "unclear",
        "no record",
        "not available",
        "insufficient information",
        "cannot determine",
        "can't determine",
        "unknown",
        "not enough information",
        "i'm not aware",
        "i am not aware",
    ]

    def required_capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities()

    async def evaluate(
        self,
        test_case: TestCase,
        query_result: QueryResult,
        *,
        resources: Mapping[str, Any],
    ) -> MetricResult:
        """
        Check if model correctly abstains on unanswerable questions.

        For abstention questions: pass if model abstains
        For answerable questions: pass if model doesn't abstain
        """
        is_abstention = test_case.metadata.get("is_abstention", False)
        if not is_abstention and "abstention" not in test_case.tags:
            # Not an abstention question - skip this metric
            return MetricResult(
                metric=self.name,
                kind=self.kind,
                score_type=self.score_type,
                score=1.0,
                passed=True,
                reason="Not an abstention question - skipped",
                artifacts={"skipped": True},
            )

        # Check if response contains abstention keywords
        response_lower = query_result.answer.lower()
        found_keywords = [kw for kw in self.ABSTENTION_KEYWORDS if kw in response_lower]

        model_abstained = len(found_keywords) > 0

        # For abstention questions, model should abstain
        passed = model_abstained == is_abstention

        if passed:
            reason = (
                f"Model correctly abstained: found [{', '.join(found_keywords[:3])}]"
                if model_abstained
                else "Model correctly answered (not abstention)"
            )
        else:
            reason = (
                "Model failed to abstain on unanswerable question"
                if is_abstention
                else f"Model incorrectly abstained: [{', '.join(found_keywords[:3])}]"
            )

        return MetricResult(
            metric=self.name,
            kind=self.kind,
            score_type=self.score_type,
            score=1.0 if passed else 0.0,
            passed=passed,
            reason=reason,
            artifacts={
                "is_abstention_question": is_abstention,
                "model_abstained": model_abstained,
                "found_keywords": found_keywords,
            },
        )


class SemanticSimilarity(BaseMetric):
    """
    Semantic similarity using embeddings.

    Computes cosine similarity between reference and response embeddings.
    Useful for soft matching when exact match is too strict.

    Returns continuous score (0-1), use ThresholdGate to convert to binary.
    """

    name = "semantic_similarity"
    kind: MetricKind = "generation"
    score_type: ScoreType = "continuous"

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def required_capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities()

    async def evaluate(
        self,
        test_case: TestCase,
        query_result: QueryResult,
        *,
        resources: Mapping[str, Any],
    ) -> MetricResult:
        """
        Compute semantic similarity.

        Resources required:
            - embedder: Async callable(text: str) -> list[float]
        """
        embedder = resources.get("embedder")
        if embedder is None:
            return MetricResult(
                metric=self.name,
                kind=self.kind,
                score_type=self.score_type,
                score=0.0,
                passed=False,
                reason="No embedder provided in resources",
            )

        reference = test_case.reference_answer or ""
        response = query_result.answer

        if not reference or not response:
            return MetricResult(
                metric=self.name,
                kind=self.kind,
                score_type=self.score_type,
                score=0.0,
                passed=False,
                reason="Empty reference or response",
            )

        try:
            # Get embeddings
            ref_embedding, resp_embedding = await asyncio.gather(
                embedder(reference),
                embedder(response),
            )

            # Compute cosine similarity
            score = self._cosine_similarity(ref_embedding, resp_embedding)
            passed = score >= self.threshold

            return MetricResult(
                metric=self.name,
                kind=self.kind,
                score_type=self.score_type,
                score=score,
                passed=passed,
                threshold=self.threshold,
                reason=f"Similarity {score:.3f} {'≥' if passed else '<'} {self.threshold}",
            )

        except Exception as e:
            return MetricResult(
                metric=self.name,
                kind=self.kind,
                score_type=self.score_type,
                score=0.0,
                passed=False,
                reason=f"Embedding error: {e}",
                artifacts={"error": str(e)},
            )

    def _cosine_similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)
