"""
Cost Tracking for Eval Runs

Tracks token usage and calculates costs based on model pricing tables.
Supports Azure Foundry, OpenAI, and embedding models.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime
import json


# Pricing per 1M tokens (USD) - December 2025
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # Azure Foundry Models
    "foundry/gpt-5.2": {"input": 2.50, "output": 10.00},
    "foundry/gpt-4.1": {"input": 2.00, "output": 8.00},
    "foundry/gpt-4o": {"input": 2.50, "output": 10.00},
    "foundry/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # OpenAI Direct
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "openai/gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Embedding Models
    "openai/text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "openai/text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "openai/text-embedding-ada-002": {"input": 0.10, "output": 0.0},
    # Anthropic
    "anthropic/claude-3-opus": {"input": 15.00, "output": 75.00},
    "anthropic/claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
    # Google
    "gemini/gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini/gemini-1.5-flash": {"input": 0.075, "output": 0.30},
}

# Quota limits per model (requests per minute / tokens per minute)
MODEL_QUOTAS: Dict[str, Dict[str, int]] = {
    "foundry/gpt-5.2": {"rpm": 60, "tpm": 150000},
    "foundry/gpt-4o": {"rpm": 60, "tpm": 150000},
    "foundry/gpt-4o-mini": {"rpm": 200, "tpm": 500000},
    "openai/gpt-4o": {"rpm": 500, "tpm": 200000},
    "openai/gpt-4o-mini": {"rpm": 5000, "tpm": 2000000},
    "openai/text-embedding-3-small": {"rpm": 5000, "tpm": 5000000},
}


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.embedding_tokens


@dataclass
class CostBreakdown:
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost


@dataclass
class EvalCostTracker:
    """Tracks costs across an entire eval run."""

    run_id: str
    start_time: datetime = field(default_factory=datetime.now)

    # Per-model token tracking
    usage_by_model: Dict[str, TokenUsage] = field(default_factory=dict)

    # Call counts
    llm_calls: int = 0
    embedding_calls: int = 0

    # Stage breakdown
    ingestion_tokens: TokenUsage = field(default_factory=TokenUsage)
    retrieval_tokens: TokenUsage = field(default_factory=TokenUsage)
    generation_tokens: TokenUsage = field(default_factory=TokenUsage)
    judge_tokens: TokenUsage = field(default_factory=TokenUsage)

    def record_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        stage: str = "generation",
    ):
        """Record a single LLM call."""
        self.llm_calls += 1

        if model not in self.usage_by_model:
            self.usage_by_model[model] = TokenUsage()

        self.usage_by_model[model].prompt_tokens += prompt_tokens
        self.usage_by_model[model].completion_tokens += completion_tokens

        stage_usage = getattr(self, f"{stage}_tokens", self.generation_tokens)
        stage_usage.prompt_tokens += prompt_tokens
        stage_usage.completion_tokens += completion_tokens

    def record_embedding_call(self, model: str, tokens: int):
        """Record an embedding call."""
        self.embedding_calls += 1

        if model not in self.usage_by_model:
            self.usage_by_model[model] = TokenUsage()

        self.usage_by_model[model].embedding_tokens += tokens

    def calculate_costs(self) -> Dict[str, CostBreakdown]:
        """Calculate costs for all models used."""
        costs = {}

        for model, usage in self.usage_by_model.items():
            pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})

            input_cost = (
                (usage.prompt_tokens + usage.embedding_tokens)
                / 1_000_000
                * pricing["input"]
            )
            output_cost = usage.completion_tokens / 1_000_000 * pricing["output"]

            costs[model] = CostBreakdown(
                model=model,
                input_tokens=usage.prompt_tokens + usage.embedding_tokens,
                output_tokens=usage.completion_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
            )

        return costs

    @property
    def total_cost(self) -> float:
        """Total cost across all models."""
        return sum(c.total_cost for c in self.calculate_costs().values())

    @property
    def total_tokens(self) -> int:
        """Total tokens across all models."""
        return sum(u.total_tokens for u in self.usage_by_model.values())

    def get_summary(self) -> Dict:
        """Generate a summary dict for logging."""
        costs = self.calculate_costs()
        duration = (datetime.now() - self.start_time).total_seconds()

        return {
            "run_id": self.run_id,
            "duration_seconds": duration,
            "total_cost_usd": round(self.total_cost, 4),
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "embedding_calls": self.embedding_calls,
            "by_model": {
                model: {
                    "input_tokens": cb.input_tokens,
                    "output_tokens": cb.output_tokens,
                    "cost_usd": round(cb.total_cost, 4),
                }
                for model, cb in costs.items()
            },
            "by_stage": {
                "ingestion": self.ingestion_tokens.total_tokens,
                "retrieval": self.retrieval_tokens.total_tokens,
                "generation": self.generation_tokens.total_tokens,
                "judge": self.judge_tokens.total_tokens,
            },
        }

    def print_summary(self):
        """Print a formatted cost summary."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("COST SUMMARY")
        print("=" * 60)
        print(f"Run ID: {summary['run_id']}")
        print(f"Duration: {summary['duration_seconds']:.1f}s")
        print(f"\nTotal Cost: ${summary['total_cost_usd']:.4f}")
        print(f"Total Tokens: {summary['total_tokens']:,}")
        print(f"LLM Calls: {summary['llm_calls']}")
        print(f"Embedding Calls: {summary['embedding_calls']}")

        print("\nBy Model:")
        for model, data in summary["by_model"].items():
            print(f"  {model}:")
            print(f"    Input: {data['input_tokens']:,} tokens")
            print(f"    Output: {data['output_tokens']:,} tokens")
            print(f"    Cost: ${data['cost_usd']:.4f}")

        print("\nBy Stage (tokens):")
        for stage, tokens in summary["by_stage"].items():
            print(f"  {stage}: {tokens:,}")

    def save(self, path: str):
        """Save cost summary to JSON file."""
        with open(path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)


def format_pricing_table() -> str:
    """Format pricing table for documentation."""
    lines = [
        "| Model | Input ($/1M) | Output ($/1M) | RPM | TPM |",
        "|-------|--------------|---------------|-----|-----|",
    ]

    for model, pricing in sorted(MODEL_PRICING.items()):
        quota = MODEL_QUOTAS.get(model, {"rpm": "N/A", "tpm": "N/A"})
        lines.append(
            f"| {model} | ${pricing['input']:.2f} | ${pricing['output']:.2f} | "
            f"{quota.get('rpm', 'N/A')} | {quota.get('tpm', 'N/A'):,} |"
            if isinstance(quota.get("tpm"), int)
            else f"| {model} | ${pricing['input']:.2f} | ${pricing['output']:.2f} | "
            f"{quota.get('rpm', 'N/A')} | {quota.get('tpm', 'N/A')} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    print("Model Pricing and Quotas (December 2025)\n")
    print(format_pricing_table())
