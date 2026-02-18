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
    sample_sizes: Dict[str, int]
    variant: Optional[str] = None


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
            longmemeval = BenchmarkConfig(
                source=lme_data.get('source', 'evals/data/longmemeval_oracle.json'),
                sample_sizes=lme_data.get('sample_sizes', {}),
                variant=lme_data.get('variant')
            )

        # Parse PersonaMem config
        personamem = None
        if 'personamem' in data:
            pm_data = data['personamem']
            personamem = BenchmarkConfig(
                source=pm_data.get('source', 'evals/data/personamem'),
                sample_sizes=pm_data.get('sample_sizes', {}),
                variant=pm_data.get('variant', '32k')
            )

        return cls(
            longmemeval=longmemeval,
            personamem=personamem,
            random_seed=global_config.get('random_seed', 42),
            adapters=global_config.get('adapters', ['persona']),
            parallel_workers=global_config.get('parallel_workers', 5),
            checkpoint_dir=global_config.get('checkpoint_dir', 'evals/results'),
            deep_logging=global_config.get('deep_logging', True),
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
                'output_dir': self.output_dir,
                'save_retrieval_logs': self.save_retrieval_logs,
            }
        }

        if self.longmemeval:
            result['longmemeval'] = {
                'source': self.longmemeval.source,
                'sample_sizes': self.longmemeval.sample_sizes,
            }
            if self.longmemeval.variant:
                result['longmemeval']['variant'] = self.longmemeval.variant

        if self.personamem:
            result['personamem'] = {
                'source': self.personamem.source,
                'sample_sizes': self.personamem.sample_sizes,
                'variant': self.personamem.variant,
            }

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
                'single-session-user': 35,
                'multi-session': 60,
                'temporal-reasoning': 60,
                'knowledge-update': 40,
                'single-session-preference': 25,
            }
        },
        'personamem': {
            'source': 'evals/data/personamem',
            'variant': '32k',
            'sample_sizes': {
                'recall_user_shared_facts': 30,
                'track_full_preference_evolution': 30,
                'generalizing_to_new_scenarios': 20,
                'provide_preference_aligned_recommendations': 20,
                'recalling_the_reasons_behind_previous_updates': 20,
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
                'single-session-user': 5,
                'multi-session': 5,
                'temporal-reasoning': 5,
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
                'single-session-user': 35,
                'multi-session': 60,
                'temporal-reasoning': 60,
                'knowledge-update': 40,
                'single-session-preference': 25,
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

    print("✓ Created default config files:")
    print("  - evals/configs/full_eval.yaml")
    print("  - evals/configs/quick_test.yaml")
    print("  - evals/configs/longmemeval_only.yaml")


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
        for qtype, count in config.longmemeval.sample_sizes.items():
            print(f"  {qtype}: {count}")

    if config.personamem:
        print(f"\nPersonaMem samples ({config.personamem.variant}):")
        for qtype, count in config.personamem.sample_sizes.items():
            print(f"  {qtype}: {count}")
