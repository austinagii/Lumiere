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
"""

from lumiere.loading.loader import ConfigLoader
from lumiere.loading.registry import HierarchicalRegistry, Registry
from lumiere.loading.resolver import resolve_value

__all__ = [
    "Registry",
    "HierarchicalRegistry",
    "ConfigLoader",
    "resolve_value",
]
