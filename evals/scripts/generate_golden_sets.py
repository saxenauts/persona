"""
Generate Golden Sets for Evaluation

This script generates golden sets for both LongMemEval and PersonaMem benchmarks
using stratified sampling.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.loaders.unified_loader import UnifiedBenchmarkLoader, SampleConfig


def generate_longmemeval_golden_set():
    """Generate LongMemEval golden set (250 questions)"""
    print("=== LongMemEval Golden Set ===\n")

    config = SampleConfig(
        sample_sizes={
            "single-session-user": 35,
            "multi-session": 60,
            "temporal-reasoning": 60,
            "knowledge-update": 40,
            "single-session-preference": 25,
            # Note: abstention questions will be sampled from within these types
            # The design calls for 30 abstention, but they overlap with other types
        },
        random_seed=42
    )

    loader = UnifiedBenchmarkLoader(benchmark="longmemeval")
    manifest = loader.create_golden_set(config)

    return manifest


def generate_personamem_golden_set():
    """Generate PersonaMem golden set (120 questions)"""
    print("\n" + "="*80 + "\n")
    print("=== PersonaMem Golden Set ===\n")

    config = SampleConfig(
        sample_sizes={
            "recall_user_shared_facts": 30,
            "track_full_preference_evolution": 30,
            # "acknowledge_latest_user_preferences": 20,  # Not in dataset
            "generalizing_to_new_scenarios": 20,
            "provide_preference_aligned_recommendations": 20,
            "recalling_the_reasons_behind_previous_updates": 20,  # Added to reach 120
        },
        random_seed=42
    )

    loader = UnifiedBenchmarkLoader(benchmark="personamem", variant="32k")
    manifest = loader.create_golden_set(config)

    return manifest


if __name__ == "__main__":
    # Generate both golden sets
    longmemeval_manifest = generate_longmemeval_golden_set()
    personamem_manifest = generate_personamem_golden_set()

    # Print final summary
    print("\n" + "="*80)
    print("GOLDEN SETS GENERATION COMPLETE")
    print("="*80)
    print(f"\nLongMemEval: {longmemeval_manifest['total_questions']} questions")
    print(f"PersonaMem: {personamem_manifest['total_questions']} questions")
    print(f"Total: {longmemeval_manifest['total_questions'] + personamem_manifest['total_questions']} questions")
    print("\n✓ Golden sets saved to: evals/data/golden_sets/")
