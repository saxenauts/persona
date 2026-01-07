#!/usr/bin/env python3
"""
Micro-Experiment Framework for Persona Evals

Design principles:
1. Ingest ONCE, query MANY - amortize expensive ingestion
2. Focus on retrieval quality - does the right context come back?
3. Target specific hypotheses - not broad accuracy, but specific capabilities

Usage:
    # Pre-ingest a context (expensive, do once)
    python -m evals.scripts.micro_experiment ingest --context-id <id>

    # Run queries against pre-ingested data (cheap, do many times)
    python -m evals.scripts.micro_experiment query --user-id <id> --question-type recall_user_shared_facts

    # Compare retrieval with/without Entity attributes
    python -m evals.scripts.micro_experiment compare-retrieval --user-id <id>
"""

import asyncio
import json
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import requests

from evals.benchmarks.personamem import PersonaMemBenchmark


@dataclass
class MicroExperimentConfig:
    api_base: str = "http://localhost:8000/api/v1"
    variant: str = "32k"
    timeout: int = 600


def get_context_with_questions(
    context_id: Optional[str] = None, min_questions: int = 5
):
    """Find a context with enough questions for meaningful testing."""
    bench = PersonaMemBenchmark()
    cases = bench.load(variant="32k")

    from collections import Counter

    users = Counter(tc.user_id for tc in cases)

    if context_id:
        user_id = f"user_{context_id}"
        if user_id not in users:
            print(f"Context {context_id} not found")
            return None, []
    else:
        # Find context with most questions
        user_id = [u for u, c in users.most_common() if c >= min_questions][0]

    user_cases = [tc for tc in cases if tc.user_id == user_id]
    return user_id, user_cases


def ingest_context(user_id: str, cases: list, config: MicroExperimentConfig):
    """Ingest a context's sessions (expensive - do once)."""
    if not cases or not cases[0].sessions:
        print("No sessions to ingest")
        return False

    session = cases[0].sessions[0]
    content = session.content

    print(f"=== INGEST CONTEXT ===")
    print(f"User: {user_id}")
    print(f"Content: {len(content):,} chars")
    print(f"Questions available: {len(cases)}")

    # Create user
    requests.post(f"{config.api_base}/users/{user_id}")

    # Check if already ingested
    resp = requests.get(
        f"{config.api_base}/users/{user_id}/memories?limit=1", timeout=30
    )
    if resp.status_code == 200:
        data = resp.json()
        if data.get("memories"):
            print(f"Context already ingested ({len(data['memories'])} memories found)")
            return True

    print("Ingesting (this takes 1-2 minutes)...")
    t0 = time.time()
    resp = requests.post(
        f"{config.api_base}/users/{user_id}/ingest/batch",
        json={"items": [{"content": content, "source_type": "conversation"}]},
        timeout=config.timeout,
    )
    elapsed = time.time() - t0

    if resp.status_code in [200, 201]:
        data = resp.json()
        print(f"Ingested in {elapsed:.1f}s")
        print(f"  Memories: {data.get('memories_created', 0)}")
        print(f"  Types: {data.get('memories_created_by_type', {})}")
        return True
    else:
        print(f"Error: {resp.status_code} - {resp.text[:200]}")
        return False


def run_query(user_id: str, query: str, config: MicroExperimentConfig):
    """Run a single query and return retrieval + answer."""
    resp = requests.post(
        f"{config.api_base}/users/{user_id}/chat",
        json={
            "messages": [{"role": "user", "content": query}],
            "include_stats": True,
        },
        timeout=120,
    )

    if resp.status_code != 200:
        return {"error": f"{resp.status_code}: {resp.text[:100]}"}

    data = resp.json()
    return {
        "answer": data.get("response", ""),
        "retrieval": data.get("stats", {}).get("retrieval", {}),
        "latency_ms": data.get("stats", {}).get("retrieval", {}).get("duration_ms", 0),
    }


def test_retrieval_quality(user_id: str, cases: list, config: MicroExperimentConfig):
    """Test retrieval quality on a set of questions."""
    print(f"\n=== RETRIEVAL QUALITY TEST ===")
    print(f"User: {user_id}")
    print(f"Questions: {len(cases)}")

    results = []
    for i, tc in enumerate(cases[:10]):  # Limit to 10 for quick test
        print(f"\n[{i + 1}/{min(10, len(cases))}] {tc.question_type}")
        print(f"  Q: {tc.query[:60]}...")

        result = run_query(user_id, tc.query, config)

        if "error" in result:
            print(f"  Error: {result['error']}")
            continue

        retrieval = result.get("retrieval", {})
        seeds = retrieval.get("vector_search", {}).get("seeds", [])

        print(f"  Retrieval: {len(seeds)} seeds, {result['latency_ms']:.0f}ms")

        # Check if answer looks correct (basic heuristic)
        answer = result["answer"].lower()
        correct_option = tc.correct_option

        # For multiple choice, check if correct option letter is mentioned
        if correct_option:
            correct_text = tc.options.get(correct_option, "")[:30].lower()
            has_correct = correct_text in answer or f"({correct_option})" in answer
            print(f"  Answer contains correct option ({correct_option}): {has_correct}")

        # Check retrieval content for Entity attributes
        entity_seeds = [s for s in seeds if s.get("node_type") == "entity"]
        if entity_seeds:
            print(f"  Entity seeds: {len(entity_seeds)}")
            for es in entity_seeds[:2]:
                content = es.get("content", "")[:100]
                has_facts = "Facts:" in content or "|" in content
                print(
                    f"    - {es.get('canonical_name', 'unknown')}: has_facts={has_facts}"
                )

        results.append(
            {
                "question_type": tc.question_type,
                "query": tc.query,
                "answer": result["answer"],
                "correct_option": correct_option,
                "seeds_count": len(seeds),
                "entity_seeds": len(entity_seeds),
                "latency_ms": result["latency_ms"],
            }
        )

    return results


def compare_entity_retrieval(user_id: str, config: MicroExperimentConfig):
    """Specifically test if Entity attributes are being retrieved."""
    print(f"\n=== ENTITY ATTRIBUTE TEST ===")
    print(f"User: {user_id}")

    # Get all entities for this user
    resp = requests.get(
        f"{config.api_base}/users/{user_id}/memories?memory_types=entity&limit=20",
        timeout=30,
    )

    if resp.status_code != 200:
        print(f"Error fetching entities: {resp.status_code}")
        return

    entities = resp.json().get("memories", [])
    print(f"Found {len(entities)} entities")

    for e in entities[:5]:
        print(f"\n  Entity: {e.get('canonical_name', 'unknown')}")
        print(f"    Type: {e.get('entity_type', 'unknown')}")
        content = e.get("content", "")
        print(f"    Content preview: {content[:150]}...")

        # Check if attributes are in content
        has_facts = "Facts:" in content
        has_structure = "|" in content
        print(f"    Has Facts section: {has_facts}")
        print(f"    Has structured format: {has_structure}")

        # Test retrieval for this entity
        query = f"What do you know about {e.get('canonical_name', '')}?"
        result = run_query(user_id, query, config)
        if "error" not in result:
            print(f"    Retrieval test: {result['latency_ms']:.0f}ms")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Micro-experiment framework")
    parser.add_argument(
        "command", choices=["ingest", "query", "test-retrieval", "test-entity"]
    )
    parser.add_argument("--context-id", help="Specific context ID to use")
    parser.add_argument("--user-id", help="User ID for queries")
    parser.add_argument("--question-type", help="Filter by question type")
    args = parser.parse_args()

    config = MicroExperimentConfig()

    if args.command == "ingest":
        user_id, cases = get_context_with_questions(args.context_id)
        if user_id:
            ingest_context(user_id, cases, config)

    elif args.command == "test-retrieval":
        if args.user_id:
            user_id = args.user_id
            bench = PersonaMemBenchmark()
            cases = [tc for tc in bench.load(variant="32k") if tc.user_id == user_id]
        else:
            user_id, cases = get_context_with_questions(
                args.context_id, min_questions=10
            )

        if user_id and cases:
            if args.question_type:
                cases = [tc for tc in cases if tc.question_type == args.question_type]
            test_retrieval_quality(user_id, cases, config)

    elif args.command == "test-entity":
        if args.user_id:
            compare_entity_retrieval(args.user_id, config)
        else:
            user_id, _ = get_context_with_questions(args.context_id)
            if user_id:
                compare_entity_retrieval(user_id, config)


if __name__ == "__main__":
    main()
