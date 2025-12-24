"""
Generate Golden Sets for Evaluation

This script generates golden sets for both LongMemEval and PersonaMem benchmarks
using stratified sampling from a single combined config file.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.loaders.unified_loader import UnifiedBenchmarkLoader, SampleConfig


DEFAULT_LONGMEMEVAL_SAMPLES = {
    "single-session-user": 60,
    "single-session-assistant": 56,
    "multi-session": 60,
    "temporal-reasoning": 60,
    "knowledge-update": 60,
    "single-session-preference": 30,
}

DEFAULT_PERSONAMEM_SAMPLES = {
    "recall_user_shared_facts": 60,
    "track_full_preference_evolution": 60,
    "generalizing_to_new_scenarios": 57,
    "provide_preference_aligned_recommendations": 55,
    "recalling_the_reasons_behind_previous_updates": 60,
    "suggest_new_ideas": 60,
    "recalling_facts_mentioned_by_the_user": 17,
}


def load_combined_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {
            "random_seed": 42,
            "output_dir": "evals/data/golden_sets",
            "benchmarks": {
                "longmemeval": {
                    "sample_sizes": DEFAULT_LONGMEMEVAL_SAMPLES,
                },
                "personamem": {
                    "variant": "32k",
                    "sample_sizes": DEFAULT_PERSONAMEM_SAMPLES,
                },
            },
        }

    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}

    benchmarks = data.get("benchmarks", {})
    longmemeval = benchmarks.get("longmemeval", {})
    personamem = benchmarks.get("personamem", {})

    return {
        "random_seed": int(data.get("random_seed", 42)),
        "output_dir": data.get("output_dir", "evals/data/golden_sets"),
        "benchmarks": {
            "longmemeval": {
                "sample_sizes": longmemeval.get("sample_sizes", DEFAULT_LONGMEMEVAL_SAMPLES),
            },
            "personamem": {
                "variant": personamem.get("variant", "32k"),
                "sample_sizes": personamem.get("sample_sizes", DEFAULT_PERSONAMEM_SAMPLES),
            },
        },
    }


def validate_sample_sizes(
    loader: UnifiedBenchmarkLoader,
    sample_sizes: Dict[str, int],
    benchmark_label: str
) -> None:
    distribution = loader.get_type_distribution()
    available_types = set(distribution.keys())
    configured_types = set(sample_sizes.keys())

    missing = sorted(available_types - configured_types)
    extra = sorted(configured_types - available_types)

    if missing or extra:
        message_parts = []
        if missing:
            message_parts.append(f"missing types: {', '.join(missing)}")
        if extra:
            message_parts.append(f"unknown types: {', '.join(extra)}")
        raise ValueError(
            f"{benchmark_label} sample_sizes invalid ({'; '.join(message_parts)}). "
            "Update evals/configs/golden_set.yaml to match dataset types."
        )


def generate_longmemeval_golden_set(sample_sizes: Dict[str, int], random_seed: int, output_dir: str):
    """Generate LongMemEval golden set."""
    print("=== LongMemEval Golden Set ===\n")

    config = SampleConfig(
        sample_sizes=sample_sizes,
        random_seed=random_seed
    )

    loader = UnifiedBenchmarkLoader(benchmark="longmemeval")
    validate_sample_sizes(loader, sample_sizes, "LongMemEval")
    manifest = loader.create_golden_set(config, output_dir=output_dir)

    return manifest


def generate_personamem_golden_set(
    sample_sizes: Dict[str, int],
    random_seed: int,
    variant: str,
    output_dir: str
):
    """Generate PersonaMem golden set."""
    print("\n" + "="*80 + "\n")
    print("=== PersonaMem Golden Set ===\n")

    config = SampleConfig(
        sample_sizes=sample_sizes,
        random_seed=random_seed
    )

    loader = UnifiedBenchmarkLoader(benchmark="personamem", variant=variant)
    validate_sample_sizes(loader, sample_sizes, "PersonaMem")
    manifest = loader.create_golden_set(config, output_dir=output_dir)

    return manifest


def build_combined_manifest(output_dir: Path) -> Dict[str, Any]:
    long_path = output_dir / "longmemeval_golden_set.json"
    persona_path = output_dir / "personamem_golden_set.json"

    combined_entries = []

    if long_path.exists():
        with open(long_path, "r") as f:
            long_items = json.load(f)
        for item in long_items:
            combined_entries.append({
                "benchmark": "longmemeval",
                "question_id": item.get("question_id"),
                "question_type": item.get("question_type"),
            })

    if persona_path.exists():
        with open(persona_path, "r") as f:
            persona_items = json.load(f)
        for item in persona_items:
            combined_entries.append({
                "benchmark": "personamem",
                "question_id": item.get("question_id"),
                "question_type": item.get("question_type"),
            })

    combined_manifest = {
        "total_questions": len(combined_entries),
        "questions": combined_entries,
    }

    combined_path = output_dir / "combined_golden_set_manifest.json"
    with open(combined_path, "w") as f:
        json.dump(combined_manifest, f, indent=2)

    return combined_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate combined golden sets.")
    parser.add_argument(
        "--config",
        default="evals/configs/golden_set.yaml",
        help="Path to combined golden set config."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory for golden sets."
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_combined_config(config_path)
    output_dir = args.output_dir or config["output_dir"]

    long_cfg = config["benchmarks"]["longmemeval"]
    persona_cfg = config["benchmarks"]["personamem"]

    longmemeval_manifest = generate_longmemeval_golden_set(
        sample_sizes=long_cfg["sample_sizes"],
        random_seed=config["random_seed"],
        output_dir=output_dir
    )
    personamem_manifest = generate_personamem_golden_set(
        sample_sizes=persona_cfg["sample_sizes"],
        random_seed=config["random_seed"],
        variant=persona_cfg["variant"],
        output_dir=output_dir
    )

    combined_manifest = build_combined_manifest(Path(output_dir))

    print("\n" + "="*80)
    print("GOLDEN SETS GENERATION COMPLETE")
    print("="*80)
    print(f"\nLongMemEval: {longmemeval_manifest['total_questions']} questions")
    print(f"PersonaMem: {personamem_manifest['total_questions']} questions")
    print(f"Total: {combined_manifest['total_questions']} questions")
    print(f"\n✓ Golden sets saved to: {output_dir}")
    print("✓ Combined manifest saved to: combined_golden_set_manifest.json")
