"""
Configuration Parser for Evaluation Framework

Handles loading and parsing of evaluation configurations from YAML files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark"""
    source: str
    sample_sizes: Optional[Dict[str, int]] = None
    variant: Optional[str] = None
    full_dataset: bool = False


@dataclass
class EvalConfig:
    """Complete evaluation configuration"""
    longmemeval: Optional[BenchmarkConfig] = None
    personamem: Optional[BenchmarkConfig] = None

    # Global settings
    random_seed: int = 42
    adapters: List[str] = field(default_factory=lambda: ["persona"])
    parallel_workers: int = 5
    checkpoint_dir: str = "evals/results"
    deep_logging: bool = True
    skip_judge: bool = False

    # Output settings
    output_dir: str = "evals/results"
    save_retrieval_logs: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "EvalConfig":
        """
        Load configuration from YAML file

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            EvalConfig object
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Parse global settings
        global_config = data.get('global', {})

        # Parse LongMemEval config
        longmemeval = None
        if 'longmemeval' in data:
            lme_data = data['longmemeval']
            lme_full = bool(lme_data.get('full_dataset', False))
            longmemeval = BenchmarkConfig(
                source=lme_data.get('source', 'evals/data/longmemeval_oracle.json'),
                sample_sizes=None if lme_full else lme_data.get('sample_sizes', {}),
                variant=lme_data.get('variant'),
                full_dataset=lme_full
            )

        # Parse PersonaMem config
        personamem = None
        if 'personamem' in data:
            pm_data = data['personamem']
            pm_full = bool(pm_data.get('full_dataset', False))
            personamem = BenchmarkConfig(
                source=pm_data.get('source', 'evals/data/personamem'),
                sample_sizes=None if pm_full else pm_data.get('sample_sizes', {}),
                variant=pm_data.get('variant', '32k'),
                full_dataset=pm_full
            )

        return cls(
            longmemeval=longmemeval,
            personamem=personamem,
            random_seed=global_config.get('random_seed', 42),
            adapters=global_config.get('adapters', ['persona']),
            parallel_workers=global_config.get('parallel_workers', 5),
            checkpoint_dir=global_config.get('checkpoint_dir', 'evals/results'),
            deep_logging=global_config.get('deep_logging', True),
            skip_judge=global_config.get('skip_judge', False),
            output_dir=global_config.get('output_dir', 'evals/results'),
            save_retrieval_logs=global_config.get('save_retrieval_logs', True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        result = {
            'global': {
                'random_seed': self.random_seed,
                'adapters': self.adapters,
                'parallel_workers': self.parallel_workers,
                'checkpoint_dir': self.checkpoint_dir,
                'deep_logging': self.deep_logging,
                'skip_judge': self.skip_judge,
                'output_dir': self.output_dir,
                'save_retrieval_logs': self.save_retrieval_logs,
            }
        }

        if self.longmemeval:
            result['longmemeval'] = {
                'source': self.longmemeval.source,
                'full_dataset': self.longmemeval.full_dataset,
            }
            if self.longmemeval.sample_sizes is not None:
                result['longmemeval']['sample_sizes'] = self.longmemeval.sample_sizes
            if self.longmemeval.variant:
                result['longmemeval']['variant'] = self.longmemeval.variant

        if self.personamem:
            result['personamem'] = {
                'source': self.personamem.source,
                'full_dataset': self.personamem.full_dataset,
                'variant': self.personamem.variant,
            }
            if self.personamem.sample_sizes is not None:
                result['personamem']['sample_sizes'] = self.personamem.sample_sizes

        return result

    def save(self, output_path: str):
        """
        Save configuration to YAML file

        Args:
            output_path: Path to save to
        """
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def create_default_configs():
    """Create default configuration files"""

    configs_dir = Path("evals/configs")
    configs_dir.mkdir(parents=True, exist_ok=True)

    # Full evaluation config
    full_eval = {
        'longmemeval': {
            'source': 'evals/data/longmemeval_oracle.json',
            'sample_sizes': {
                'single-session-user': 60,
                'single-session-assistant': 56,
                'multi-session': 60,
                'temporal-reasoning': 60,
                'knowledge-update': 60,
                'single-session-preference': 30,
            }
        },
        'personamem': {
            'source': 'evals/data/personamem',
            'variant': '32k',
            'sample_sizes': {
                'recall_user_shared_facts': 60,
                'track_full_preference_evolution': 60,
                'generalizing_to_new_scenarios': 57,
                'provide_preference_aligned_recommendations': 55,
                'recalling_the_reasons_behind_previous_updates': 60,
                'suggest_new_ideas': 60,
                'recalling_facts_mentioned_by_the_user': 17,
            }
        },
        'global': {
            'random_seed': 42,
            'adapters': ['persona'],
            'parallel_workers': 5,
            'checkpoint_dir': 'evals/results',
            'deep_logging': True,
        }
    }

    with open(configs_dir / "full_eval.yaml", 'w') as f:
        yaml.dump(full_eval, f, default_flow_style=False, sort_keys=False)

    # Quick test config (smaller sample)
    quick_test = {
        'longmemeval': {
            'source': 'evals/data/longmemeval_oracle.json',
            'sample_sizes': {
                'single-session-user': 2,
                'single-session-assistant': 2,
                'multi-session': 2,
                'temporal-reasoning': 2,
                'knowledge-update': 2,
                'single-session-preference': 2,
            }
        },
        'personamem': {
            'source': 'evals/data/personamem',
            'variant': '32k',
            'sample_sizes': {
                'recall_user_shared_facts': 2,
                'track_full_preference_evolution': 2,
                'generalizing_to_new_scenarios': 2,
                'provide_preference_aligned_recommendations': 2,
                'recalling_the_reasons_behind_previous_updates': 2,
                'suggest_new_ideas': 2,
                'recalling_facts_mentioned_by_the_user': 2,
            }
        },
        'global': {
            'random_seed': 42,
            'adapters': ['persona'],
            'parallel_workers': 2,
            'deep_logging': True,
        }
    }

    with open(configs_dir / "quick_test.yaml", 'w') as f:
        yaml.dump(quick_test, f, default_flow_style=False, sort_keys=False)

    # LongMemEval only config
    longmemeval_only = {
        'longmemeval': {
            'source': 'evals/data/longmemeval_oracle.json',
            'sample_sizes': {
                'single-session-user': 60,
                'single-session-assistant': 56,
                'multi-session': 60,
                'temporal-reasoning': 60,
                'knowledge-update': 60,
                'single-session-preference': 30,
            }
        },
        'global': {
            'random_seed': 42,
            'adapters': ['persona'],
            'parallel_workers': 5,
            'deep_logging': True,
        }
    }

    with open(configs_dir / "longmemeval_only.yaml", 'w') as f:
        yaml.dump(longmemeval_only, f, default_flow_style=False, sort_keys=False)

    # Full dataset config
    full_dataset = {
        'longmemeval': {
            'source': 'evals/data/longmemeval_oracle.json',
            'full_dataset': True,
        },
        'personamem': {
            'source': 'evals/data/personamem',
            'variant': '32k',
            'full_dataset': True,
        },
        'global': {
            'random_seed': 42,
            'adapters': ['persona'],
            'parallel_workers': 5,
            'checkpoint_dir': 'evals/results',
            'deep_logging': True,
        }
    }

    with open(configs_dir / "full_dataset.yaml", 'w') as f:
        yaml.dump(full_dataset, f, default_flow_style=False, sort_keys=False)

    print("✓ Created default config files:")
    print("  - evals/configs/full_eval.yaml")
    print("  - evals/configs/quick_test.yaml")
    print("  - evals/configs/longmemeval_only.yaml")
    print("  - evals/configs/full_dataset.yaml")


# Example usage
if __name__ == "__main__":
    # Create default configs
    create_default_configs()

    # Test loading
    print("\n" + "="*60)
    print("Testing config loading...")
    print("="*60)

    config = EvalConfig.from_yaml("evals/configs/full_eval.yaml")

    print(f"\nRandom seed: {config.random_seed}")
    print(f"Adapters: {config.adapters}")
    print(f"Parallel workers: {config.parallel_workers}")

    if config.longmemeval:
        print(f"\nLongMemEval samples:")
        if config.longmemeval.sample_sizes:
            for qtype, count in config.longmemeval.sample_sizes.items():
                print(f"  {qtype}: {count}")
        else:
            print("  full_dataset: true")

    if config.personamem:
        print(f"\nPersonaMem samples ({config.personamem.variant}):")
        if config.personamem.sample_sizes:
            for qtype, count in config.personamem.sample_sizes.items():
                print(f"  {qtype}: {count}")
        else:
            print("  full_dataset: true")
