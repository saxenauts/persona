"""
Benchmark registry and plugin system.

Design Principles (from OpenBench research):
- One official implementation per benchmark
- Benchmarks produce TestCase lists, nothing more
- Registry pattern for discoverability
"""

from .registry import BenchmarkRegistry, get_registry, register_benchmark
from .personamem import PersonaMemBenchmark
from .longmemeval import LongMemEvalBenchmark

__all__ = [
    # Registry
    "BenchmarkRegistry",
    "get_registry",
    "register_benchmark",
    # Benchmarks
    "PersonaMemBenchmark",
    "LongMemEvalBenchmark",
]
