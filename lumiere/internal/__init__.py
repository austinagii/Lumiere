"""Internal infrastructure for component loading and registry."""

from lumiere.internal.di import (
    DependencyContainer,
    clear_global_dependencies,
    get_global_container,
    register_dependency,
    resolve_value,
)
from lumiere.internal.loader import (
    Loader,
    load as load_component,
    load_data,
    load_model,
    load_optimizer,
    load_pipeline,
    load_scheduler,
    load_tokenizer,
)
from lumiere.internal.registry import discover, get, get_component, register


__all__ = [
    # DI
    "DependencyContainer",
    "get_global_container",
    "register_dependency",
    "clear_global_dependencies",
    "resolve_value",
    # Loader (class and individual functions)
    "Loader",
    "load_component",
    "load_data",
    "load_model",
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
