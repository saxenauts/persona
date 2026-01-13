#!/usr/bin/env python3
"""
Micro-experiments to validate root cause hypotheses for PersonaMem failures.

This directly uses the PersonaService to test different prompt variations
on the exact failure cases from our eval run.

Run from persona repo root:
  cd /Users/saxenauts/Documents/InnerNets AI Inc/persona/persona
  python -m micro_experiments
"""

import asyncio
import json
import sys
from pathlib import Path

# Add persona to path
sys.path.insert(0, str(Path(__file__).parent))

from persona.services.persona_service import PersonaService
from persona.core.graph_ops import create_graph_ops


# Load the 3 failure cases
CASES_FILE = Path("/tmp/micro_experiment_cases.json")
if CASES_FILE.exists():
    with open(CASES_FILE) as f:
        CASES = json.load(f)
else:
    # Fallback - define inline
    CASES = {}
    print("Warning: No cases file found. Run the extraction first.")


# =============================================================================
# HYPOTHESIS PROMPTS
# =============================================================================

BASELINE_PROMPT = """You are a personal AI assistant with memory.

{world_model}

## Your Tools

**recall(query)** - Search memories.
**browse()** - List recent memories.

## How to Be Useful

Use tools to search before answering questions about the user.
"""

# Hypothesis 1: Emphasize recency/date in decision making
HYPOTHESIS_1_RECENCY = """You are a personal AI assistant with memory.

{world_model}

## Your Tools

**recall(query)** - Search memories. Results include dates when available.
**browse()** - List memories in time order.

## Critical: Temporal Reasoning

When multiple memories exist about the same topic:
1. LATER experiences often supersede EARLIER ones
2. If user had negative experience THEN positive, the positive is current state
3. If user had positive experience THEN negative, the negative is current state
4. Look at dates/order to determine which is most relevant NOW

## How to Answer

Search memories, note the temporal order, then answer based on the user's CURRENT state.
"""

# Hypothesis 2: Episodes/Entities as evidence over Psyche generalizations
HYPOTHESIS_2_EVIDENCE = """You are a personal AI assistant with memory.

{world_model}

## Your Tools

**recall(query)** - Search memories.
**browse()** - List recent memories.

## Critical: Evidence Hierarchy

Memory types have different evidential weight:

1. **Episodes** (what happened) = PRIMARY EVIDENCE of actions/participation
2. **Entities** (things/events with facts) = PRIMARY EVIDENCE of specific experiences  
3. **Psyche** (traits/preferences) = GENERALIZATIONS that may have exceptions

When Psyche says "user avoids X" but an Episode/Entity shows "user did X":
→ The user DID X, even if they generally prefer to avoid it.

Concrete participation evidence OVERRIDES general preference statements.

## How to Answer

Search for evidence, weight Episodes/Entities over Psyche, then answer.
"""

# Hypothesis 3: Explicit contradiction handling
HYPOTHESIS_3_CONTRADICTIONS = """You are a personal AI assistant with memory.

{world_model}

## Your Tools

**recall(query)** - Search memories. May return CONTRADICTORY information.
**browse()** - List recent memories.

## Critical: Handling Contradictions

The same topic may have MULTIPLE memories with OPPOSITE sentiments:
- "enjoyed dance classes" AND "found dance classes overwhelming"  
- "likes online forums" AND "prefers avoiding online forums"
- "great experience at club" AND "disappointing experience at club"

This is NORMAL - people have varied experiences over time.

When you see contradictions:
1. Don't just pick the highest-scored result
2. Read ALL relevant memories
3. Match the specific question context to the right memory
4. If question asks about a SPECIFIC experience, find THAT experience

## How to Answer

Search, identify which memory matches the question's context, answer accordingly.
"""

# Hypothesis 4: XML-structured context (modify how we present memories)
HYPOTHESIS_4_XML_FORMAT = """You are a personal AI assistant with memory.

{world_model}

## Your Tools

**recall(query)** - Search memories. Returns structured results.
**browse()** - List recent memories.

## Reading Memory Results

Memory results are structured as:
<memory type="episode|entity|psyche" score="0.XX" date="YYYY-MM-DD">
  <title>What this is about</title>
  <content>Details and facts</content>
  <links>Related memories</links>
</memory>

- **score**: Similarity to your query (higher = more similar, NOT more correct)
- **type**: episode (events), entity (things), psyche (preferences)
- **date**: When this happened (use for recency reasoning)

## How to Answer

Search, read all results carefully, pick the one that matches the question context.
"""


async def run_single_test(
    user_id: str, query: str, system_prompt: str, hypothesis_name: str
) -> dict:
    """Run a single test with a specific prompt."""
    graph_ops = await create_graph_ops()

    try:
        service = PersonaService(graph_ops)

        # Run agent with custom prompt
        result = await service.run_agent(
            user_id=user_id,
            message=query,
            system_prompt_template=system_prompt,
        )

        # Extract answer letter from response
        response = result.get("response", "")
        answer = extract_answer(response)

        return {
            "hypothesis": hypothesis_name,
            "response": response[:200],
            "answer": answer,
            "tool_calls": result.get("stats", {}).get("tool_calls_made", 0),
        }
    except Exception as e:
        return {
            "hypothesis": hypothesis_name,
            "response": f"ERROR: {e}",
            "answer": "error",
            "tool_calls": 0,
        }
    finally:
        await graph_ops.close()


def extract_answer(response: str) -> str:
    """Extract a/b/c/d from response."""
    response = response.strip().lower()

    # Direct letter at start
    if response and response[0] in "abcd":
        return response[0]

    # (a), (b), etc.
    for letter in "abcd":
        if f"({letter})" in response:
            return letter

    # "answer is X" patterns
    for letter in "abcd":
        if f"answer: {letter}" in response or f"answer is {letter}" in response:
            return letter

    return "unknown"


async def main():
    if not CASES:
        print("No test cases loaded. First run the extraction command.")
        return

    hypotheses = [
        ("BASELINE", BASELINE_PROMPT),
        ("H1: Recency emphasis", HYPOTHESIS_1_RECENCY),
        ("H2: Evidence hierarchy", HYPOTHESIS_2_EVIDENCE),
        ("H3: Contradiction handling", HYPOTHESIS_3_CONTRADICTIONS),
    ]

    print("=" * 80)
    print("MICRO-EXPERIMENTS: Root Cause Validation")
    print("=" * 80)

    all_results = []

    for case_id, case in CASES.items():
        print(f"\n{'=' * 80}")
        print(f"CASE: {case_id}")
        print(f"Question: {case['question'][:60]}...")
        print(f"Gold: {case['gold']} | Baseline got: {case['got']}")
        print("=" * 80)

        for hyp_name, prompt in hypotheses:
            result = await run_single_test(
                user_id=case["user_id"],
                query=case["query"],
                system_prompt=prompt,
                hypothesis_name=hyp_name,
            )

            result["case_id"] = case_id
            result["gold"] = case["gold"]
            result["baseline"] = case["got"]
            result["improved"] = result["answer"] == case["gold"]

            all_results.append(result)

            status = "✅" if result["improved"] else "❌"
            print(f"\n{hyp_name}")
            print(
                f"  {status} Answer: {result['answer']} (gold: {case['gold']}, baseline: {case['got']})"
            )
            print(f"  Tool calls: {result['tool_calls']}")
            if result["answer"] != case["gold"]:
                print(f"  Response: {result['response'][:100]}...")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Improvements by Hypothesis")
    print("=" * 80)

    for hyp_name, _ in hypotheses:
        hyp_results = [r for r in all_results if r["hypothesis"] == hyp_name]
        improved = sum(1 for r in hyp_results if r["improved"])
        total = len(hyp_results)
        pct = (improved / total * 100) if total > 0 else 0
        print(f"{hyp_name}: {improved}/{total} ({pct:.0f}%)")

    # Save results
    results_file = Path("/tmp/micro_experiment_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
