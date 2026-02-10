"""Internal infrastructure for component loading and registry."""

from lumiere.internal.di import DependencyContainer, resolve_value
from lumiere.internal.loader import (
    load as load_component,
    load_dataset,
    load_optimizer,
    load_pipeline,
    load_scheduler,
    load_tokenizer,
)
from lumiere.internal.registry import discover, get, get_component, register


__all__ = [
    # DI
    "DependencyContainer",
    "resolve_value",
    # Loader
    "load_component",
    "load_dataset",
    "load_optimizer",
    "load_pipeline",
    "load_scheduler",
    "load_tokenizer",
    # Registry
    "discover",
    "get",
    "get_component",
    "register",
]
