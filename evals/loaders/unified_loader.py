"""
Unified Benchmark Loader

Provides a unified interface for loading and sampling from multiple benchmark datasets.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass

from .personamem_loader import PersonaMemLoader, PersonaMemQuestion
from .longmemeval_loader import LongMemEvalLoader, LongMemEvalQuestion


@dataclass
class SampleConfig:
    """Configuration for stratified sampling"""

    sample_sizes: Dict[str, int]  # {question_type: count}
    random_seed: int = 42
    save_manifest: bool = True


class UnifiedBenchmarkLoader:
    """Unified loader for multiple benchmark datasets"""

    def __init__(
        self,
        benchmark: str,
        data_dir: Optional[str] = None,
        variant: Optional[str] = None
    ):
        """
        Initialize unified benchmark loader

        Args:
            benchmark: Benchmark name ('longmemeval' or 'personamem')
            data_dir: Optional data directory path
            variant: Optional variant name (for PersonaMem: '32k', '128k', '1M')
        """
        self.benchmark = benchmark.lower()

        if self.benchmark == "longmemeval":
            data_path = data_dir or "evals/data/longmemeval_oracle.json"
            self.loader = LongMemEvalLoader(data_path=data_path)
        elif self.benchmark == "personamem":
            data_dir = data_dir or "evals/data/personamem"
            variant = variant or "32k"
            self.loader = PersonaMemLoader(data_dir=data_dir, variant=variant)
        else:
            raise ValueError(
                f"Unknown benchmark: {benchmark}. "
                f"Supported: 'longmemeval', 'personamem'"
            )

    def load(self) -> List[Union[LongMemEvalQuestion, PersonaMemQuestion]]:
        """
        Load all questions from the benchmark

        Returns:
            List of question objects
        """
        return self.loader.load()

    def load_by_type(self, question_type: str) -> List[Union[LongMemEvalQuestion, PersonaMemQuestion]]:
        """
        Load questions filtered by type

        Args:
            question_type: Question type to filter

        Returns:
            List of filtered questions
        """
        return self.loader.load_by_type(question_type)

    def get_type_distribution(self) -> Dict[str, int]:
        """
        Get distribution of questions by type

        Returns:
            Dictionary mapping question type to count
        """
        return self.loader.get_type_distribution()

    def stratified_sample(
        self,
        sample_sizes: Dict[str, int],
        random_seed: int = 42
    ) -> List[Union[LongMemEvalQuestion, PersonaMemQuestion]]:
        """
        Perform stratified random sampling

        Args:
            sample_sizes: Dictionary mapping question type to sample size
            random_seed: Random seed for reproducibility

        Returns:
            List of sampled questions
        """
        np.random.seed(random_seed)

        # Load all questions
        all_questions = self.load()

        # Group by type
        questions_by_type: Dict[str, List] = {}
        for q in all_questions:
            qtype = q.question_type
            if qtype not in questions_by_type:
                questions_by_type[qtype] = []
            questions_by_type[qtype].append(q)

        # Sample from each type
        sampled_questions = []
        for qtype, n_samples in sample_sizes.items():
            if qtype not in questions_by_type:
                print(f"Warning: Question type '{qtype}' not found in dataset")
                continue

            questions_of_type = questions_by_type[qtype]
            available = len(questions_of_type)

            if n_samples > available:
                print(
                    f"Warning: Requested {n_samples} samples of type '{qtype}' "
                    f"but only {available} available. Using all {available}."
                )
                n_samples = available

            # Random sample without replacement
            indices = np.random.choice(
                len(questions_of_type),
                size=n_samples,
                replace=False
            )

            sampled = [questions_of_type[i] for i in indices]
            sampled_questions.extend(sampled)

            print(f"Sampled {len(sampled)}/{available} questions of type '{qtype}'")

        print(f"\nTotal sampled: {len(sampled_questions)} questions")
        return sampled_questions

    def create_golden_set(
        self,
        config: SampleConfig,
        output_dir: str = "evals/data/golden_sets"
    ) -> Dict[str, Any]:
        """
        Create a golden set with stratified sampling

        Args:
            config: Sample configuration
            output_dir: Directory to save golden set

        Returns:
            Dictionary with golden set metadata
        """
        print(f"\n{'='*60}")
        print(f"Creating Golden Set: {self.benchmark}")
        print(f"{'='*60}\n")

        # Perform stratified sampling
        sampled_questions = self.stratified_sample(
            sample_sizes=config.sample_sizes,
            random_seed=config.random_seed
        )

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save questions
        questions_file = output_path / f"{self.benchmark}_golden_set.json"
        questions_data = [q.to_dict() for q in sampled_questions]

        with open(questions_file, 'w') as f:
            json.dump(questions_data, f, indent=2)

        print(f"\n✓ Saved golden set to: {questions_file}")

        # Create manifest
        manifest = {
            "benchmark": self.benchmark,
            "total_questions": len(sampled_questions),
            "random_seed": config.random_seed,
            "sample_sizes": config.sample_sizes,
            "question_ids": [q.question_id for q in sampled_questions],
            "type_distribution": {}
        }

        # Calculate actual distribution
        for q in sampled_questions:
            qtype = q.question_type
            manifest["type_distribution"][qtype] = manifest["type_distribution"].get(qtype, 0) + 1

        # Save manifest
        if config.save_manifest:
            manifest_file = output_path / f"{self.benchmark}_golden_set_manifest.json"
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)

            print(f"✓ Saved manifest to: {manifest_file}")

        print(f"\n{'='*60}")
        print("Golden Set Summary:")
        print(f"{'='*60}")
        print(f"Total questions: {manifest['total_questions']}")
        print("\nType Distribution:")
        for qtype, count in sorted(manifest["type_distribution"].items()):
            print(f"  {qtype}: {count}")
        print(f"{'='*60}\n")

        return manifest


# Example usage
if __name__ == "__main__":
    print("=== LongMemEval Golden Set ===\n")

    # Create LongMemEval golden set
    longmemeval_config = SampleConfig(
        sample_sizes={
            "single-session-user": 35,
            "multi-session": 60,
            "temporal-reasoning": 60,
            "knowledge-update": 40,
            "single-session-preference": 25,
        },
        random_seed=42
    )

    longmemeval_loader = UnifiedBenchmarkLoader(benchmark="longmemeval")
    longmemeval_manifest = longmemeval_loader.create_golden_set(longmemeval_config)

    print("\n" + "="*80 + "\n")
    print("=== PersonaMem Golden Set ===\n")

    # Create PersonaMem golden set
    personamem_config = SampleConfig(
        sample_sizes={
            "recall_user_shared_facts": 30,
            "track_full_preference_evolution": 30,
            "acknowledge_latest_user_preferences": 20,
            "generalizing_to_new_scenarios": 20,
            "provide_preference_aligned_recommendations": 20,
        },
        random_seed=42
    )

    personamem_loader = UnifiedBenchmarkLoader(
        benchmark="personamem",
        variant="32k"
    )
    personamem_manifest = personamem_loader.create_golden_set(personamem_config)
