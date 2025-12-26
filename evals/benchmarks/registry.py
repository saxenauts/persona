"""
Benchmark registry for discovering and loading benchmarks.

Design: Simple registry pattern. Benchmarks register themselves,
engine discovers them by name.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.interfaces import Benchmark


class BenchmarkRegistry:
    """
    Registry for benchmark implementations.

    Usage:
        registry = BenchmarkRegistry()
        registry.register("personamem", PersonaMemBenchmark)

        benchmark = registry.get("personamem")
        test_cases = benchmark.load(variant="32k")
    """

    def __init__(self):
        self._benchmarks: Dict[str, Type["Benchmark"]] = {}

    def register(
        self,
        name: str,
        benchmark_class: Type["Benchmark"],
    ) -> None:
        """Register a benchmark implementation."""
        self._benchmarks[name] = benchmark_class

    def get(self, name: str) -> Optional["Benchmark"]:
        """Get a benchmark by name (instantiated)."""
        cls = self._benchmarks.get(name)
        if cls:
            return cls()
        return None

    def list_benchmarks(self) -> list[str]:
        """List all registered benchmark names."""
        return list(self._benchmarks.keys())

    def has(self, name: str) -> bool:
        """Check if a benchmark is registered."""
        return name in self._benchmarks


# Global registry singleton
_global_registry: Optional[BenchmarkRegistry] = None


def get_registry() -> BenchmarkRegistry:
    """Get the global benchmark registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = BenchmarkRegistry()
        _register_builtin_benchmarks(_global_registry)
    return _global_registry


def register_benchmark(name: str):
    """
    Decorator to register a benchmark class.

    Usage:
        @register_benchmark("my_benchmark")
        class MyBenchmark:
            name = "my_benchmark"
            version = "1.0"

            def load(self, *, variant=None):
                return [...]
    """

    def decorator(cls: Type["Benchmark"]) -> Type["Benchmark"]:
        get_registry().register(name, cls)
        return cls

    return decorator


def _register_builtin_benchmarks(registry: BenchmarkRegistry) -> None:
    """Register built-in benchmarks."""
    # Import here to avoid circular imports
    try:
        from .personamem import PersonaMemBenchmark

        registry.register("personamem", PersonaMemBenchmark)
    except ImportError:
        pass  # PersonaMem loader not available

    try:
        from .longmemeval import LongMemEvalBenchmark

        registry.register("longmemeval", LongMemEvalBenchmark)
    except ImportError:
        pass  # LongMemEval loader not available
