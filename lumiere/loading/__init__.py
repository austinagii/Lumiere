"""
Centralized loading infrastructure for Lumière.

This module provides base classes and utilities for creating loaders throughout
the codebase. It eliminates code duplication across the 9+ loader implementations
and provides a consistent API for registration, discovery, and instantiation.

Main components:
- Registry: Generic registry with decorator-based registration and auto-discovery
- HierarchicalRegistry: Registry for hierarchical keys (e.g., "attention.multihead")
- ConfigLoader: Generic config-based loader with dependency injection
- resolve_value: Dependency injection helper for resolving @variable syntax

Loaders (import directly from submodules):
- from lumiere.loading.tokenizer import load  (or load_tokenizer)
- from lumiere.loading.optimizer import load  (or load_optimizer)
- from lumiere.loading.scheduler import load  (or load_scheduler)
- from lumiere.loading.dataset import load    (or load_dataset)
- from lumiere.loading.pipeline import load   (or load_pipeline)
"""

from lumiere.loading.loader import ConfigLoader
from lumiere.loading.registry import HierarchicalRegistry, Registry
from lumiere.loading.resolver import resolve_value

# Loaders are imported lazily to avoid torch dependency issues
# Import them directly from submodules:
#   from lumiere.loading.tokenizer import load as load_tokenizer
#   from lumiere.loading.optimizer import load as load_optimizer
#   from lumiere.loading.scheduler import load as load_scheduler
#   from lumiere.loading.dataset import load as load_dataset
#   from lumiere.loading.pipeline import load as load_pipeline

__all__ = [
    "Registry",
    "HierarchicalRegistry",
    "ConfigLoader",
    "resolve_value",
]


# Provide lazy imports for convenience
def __getattr__(name):
    """Lazy import loaders to avoid requiring torch at import time."""
    if name == "load_tokenizer":
        from lumiere.loading.tokenizer import load
        return load
    elif name == "load_optimizer":
        from lumiere.loading.optimizer import load
        return load
    elif name == "load_scheduler":
        from lumiere.loading.scheduler import load
        return load
    elif name == "load_dataset":
        from lumiere.loading.dataset import load
        return load
    elif name == "load_pipeline":
        from lumiere.loading.pipeline import load
        return load
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
