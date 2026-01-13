#!/usr/bin/env python3
"""
Micro-experiments: Test different prompts against the same retrieved context.

This bypasses the full agent loop - just tests LLM response to different prompt framings
with the SAME context from the eval failures.

Run: python micro_experiments_direct.py
"""

import json
import asyncio
import os
from pathlib import Path
import openai
from dotenv import load_dotenv

load_dotenv()

# Load cases
CASES_FILE = Path("/tmp/micro_experiment_cases.json")
with open(CASES_FILE) as f:
    CASES = json.load(f)


def make_prompt(hypothesis: str, context: str, question: str, options: str) -> str:
    """Build full prompt with hypothesis-specific framing."""

    if hypothesis == "BASELINE":
        return f"""You are a personal AI assistant with access to the user's memories.

Here are the user's memories:
{context}

User: {question}
{options}

Answer with only the letter (a/b/c/d)."""

    elif hypothesis == "H1_RECENCY":
        return f"""You are a personal AI assistant with access to the user's memories.

## Critical: Temporal Reasoning
When multiple memories exist about the same topic:
- LATER experiences supersede EARLIER ones
- The MOST RECENT experience reflects the user's CURRENT state
- Look for temporal clues (dates, "recently", "later", "afterwards")

Here are the user's memories (may span different time periods):
{context}

User: {question}
{options}

Think about which memory is most recent, then answer with only the letter (a/b/c/d)."""

    elif hypothesis == "H2_EVIDENCE":
        return f"""You are a personal AI assistant with access to the user's memories.

## Critical: Evidence Hierarchy
1. **Episodes** (what actually happened) = PRIMARY evidence
2. **Entities** (specific things/events with facts) = PRIMARY evidence
3. **Psyche** (general preferences/traits) = generalizations with exceptions

If a general preference says "avoids X" but a specific Episode/Entity shows "did X",
the user DID X. Concrete actions override general tendencies.

Here are the user's memories:
{context}

User: {question}
{options}

Weight concrete actions over general preferences. Answer with only the letter (a/b/c/d)."""

    elif hypothesis == "H3_CONTRADICTIONS":
        return f"""You are a personal AI assistant with access to the user's memories.

## Critical: Handling Contradictions
Users often have MIXED experiences with the same topic:
- "enjoyed X" in one context, "found X overwhelming" in another
- Both can be true at different times or in different circumstances

When memories contradict:
1. Don't default to the first/highest-scored memory
2. Look for which memory matches the SPECIFIC situation being asked about
3. Match the question's context to the right memory

Here are the user's memories (may contain contradictions):
{context}

User: {question}
{options}

Read all memories carefully, find the one matching this context. Answer with only the letter (a/b/c/d)."""

    elif hypothesis == "H4_SCORING_CAVEAT":
        return f"""You are a personal AI assistant with access to the user's memories.

## Important: Understanding Memory Scores
Memories below are ranked by SIMILARITY to the question, NOT by correctness:
- score=0.85 means "semantically similar to your query"
- A lower-scored memory may actually be the correct answer
- Don't assume the first result is the right one

Here are the user's memories (ranked by similarity, not correctness):
{context}

User: {question}
{options}

Consider all memories, not just the highest-scored. Answer with only the letter (a/b/c/d)."""

    elif hypothesis == "H5_SENTIMENT_MATCH":
        return f"""You are a personal AI assistant with access to the user's memories.

## Handling Contradictory Memories

You may retrieve multiple memories about the same topic with DIFFERENT sentiments:
- Positive experience: "felt satisfying, engaging, therapeutic"
- Negative experience: "felt overwhelming, tedious, frustrating"

When this happens:
1. **Read ALL memories**, not just the highest-scored one
2. **Match the question's tone** to the memory's sentiment:
   - Question implies positive experience → find positive memory
   - Question implies negative experience → find negative memory
3. **Actions override preferences**: If memories show they DID something (e.g., "joined a forum"), that outweighs general preferences (e.g., "prefers not to join forums")
4. **Don't average contradictions**: Pick the memory that matches the question's framing

Example: Question "I realized how interconnected subjects can be" (positive tone)
- Memory A: "mind maps felt overwhelming" (negative) ← score 0.86
- Memory B: "concept maps felt satisfying" (positive) ← score 0.79
→ Pick B despite lower score — it matches the question's positive framing.

Here are the user's memories:
{context}

User: {question}
{options}

Match question sentiment to memory sentiment. Answer with only the letter (a/b/c/d)."""

    elif hypothesis == "H6_TRUE_CONFLICTS":
        return f"""You are a personal AI assistant with access to the user's memories.

## Handling Contradictory Memories

You may retrieve multiple memories about the same topic with DIFFERENT sentiments.

When this happens:
1. **Read ALL memories**, not just the highest-scored one
2. **Match the question's tone** to the memory's sentiment:
   - Question implies positive experience → find positive memory
   - Question implies negative experience → find negative memory
   - Question is neutral (just states a fact) → see rule 5
3. **Actions override preferences**: If memories show they DID something (e.g., "joined a forum"), that outweighs general preferences (e.g., "prefers not to join forums")
4. **Don't average contradictions**: Pick the memory that matches the question's framing
5. **True conflicts → stay neutral**: If you find genuinely conflicting experiences (e.g., one positive film club, one negative), AND the question doesn't imply which experience they're referring to, give a neutral/vague response that acknowledges the memory without committing to a specific sentiment.

Here are the user's memories:
{context}

User: {question}
{options}

Apply these rules carefully. Answer with only the letter (a/b/c/d)."""

    elif hypothesis == "H7_COMBINED":
        return f"""You are a personal AI assistant with access to the user's memories.

## Critical: Handling Multiple Memories

When you retrieve multiple memories about the same topic:

### 1. Recency First
If memories span different time periods:
- LATER experiences supersede EARLIER ones
- The MOST RECENT experience reflects the user's CURRENT state
- Look for temporal clues (dates, "recently", "later", "afterwards")

### 2. Sentiment Matching
If question has a clear sentiment (positive/negative):
- Match the question's tone to the memory's sentiment
- Don't just pick the highest-scored result

### 3. True Conflicts → Stay Neutral
If you find genuinely conflicting experiences (e.g., one positive, one negative) with similar recency, AND the question doesn't imply which experience they're referring to:
- Give a neutral/vague response that acknowledges the memory
- Don't commit to a specific sentiment when uncertain

### 4. Actions Over Preferences
If memories show they DID something (action), that outweighs general preferences.

Here are the user's memories:
{context}

User: {question}
{options}

Apply these rules in order. Answer with only the letter (a/b/c/d)."""

    return f"Unknown hypothesis: {hypothesis}"


async def run_llm(prompt: str) -> str:
    """Call LLM and return response."""
    api_base = os.environ.get("AZURE_API_BASE", "").rstrip("/")
    if not api_base.endswith("/openai/v1"):
        api_base = f"{api_base}/openai/v1/"

    client = openai.AsyncOpenAI(
        api_key=os.environ.get("AZURE_API_KEY"),
        base_url=api_base,
    )

    try:
        response = await client.chat.completions.create(
            model=os.environ.get("AZURE_CHAT_DEPLOYMENT", "gpt-5.2"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=10,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        return f"ERROR: {e}"


def extract_answer(response: str) -> str:
    """Extract a/b/c/d from response."""
    response = response.strip().lower()
    for letter in "abcd":
        if response.startswith(letter) or response.startswith(f"({letter})"):
            return letter
    return "unknown"


async def main():
    print("=" * 80)
    print("MICRO-EXPERIMENTS: Direct LLM Testing with Retrieved Context")
    print("=" * 80)

    hypotheses = [
        "BASELINE",
        "H1_RECENCY",
        "H6_TRUE_CONFLICTS",
        "H7_COMBINED",
    ]

    results = []

    for case_id, case in CASES.items():
        print(f"\n{'=' * 80}")
        print(f"CASE: {case_id}")
        print(f"Gold: {case['gold']} | Baseline eval: {case['got']}")
        print(f"Question: {case['question'][:60]}...")
        print("=" * 80)

        # Extract options from the query
        query_parts = case["query"].split("Options:")
        question = case["question"]
        options = f"Options:{query_parts[1]}" if len(query_parts) > 1 else ""
        context = case["retrieved_context"] or "(no context retrieved)"

        for hyp in hypotheses:
            prompt = make_prompt(hyp, context, question, options)
            response = await run_llm(prompt)
            answer = extract_answer(response)
            improved = answer == case["gold"]

            results.append(
                {
                    "case_id": case_id,
                    "hypothesis": hyp,
                    "gold": case["gold"],
                    "baseline": case["got"],
                    "answer": answer,
                    "improved": improved,
                    "response": response,
                }
            )

            status = "✅" if improved else "❌"
            match_baseline = " (same as baseline)" if answer == case["got"] else ""
            print(f"{hyp}: {status} {answer}{match_baseline}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Improvement Rate by Hypothesis")
    print("=" * 80)

    for hyp in hypotheses:
        hyp_results = [r for r in results if r["hypothesis"] == hyp]
        improved = sum(1 for r in hyp_results if r["improved"])
        total = len(hyp_results)
        pct = improved / total * 100 if total > 0 else 0
        bar = "█" * improved + "░" * (total - improved)
        print(f"{hyp:20} [{bar}] {improved}/{total} ({pct:.0f}%)")

    # Save results
    with open("/tmp/micro_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to /tmp/micro_experiment_results.json")


if __name__ == "__main__":
    asyncio.run(main())
