"""Lumiere - A deep learning framework."""

from .internal.di import (
    DependencyContainer,
    clear_global_dependencies,
    get_global_container,
    register_dependency,
)
from .internal.loader import (
    Loader,
    load_data,
    load_model,
    load_optimizer,
    load_pipeline,
    load_scheduler,
    load_tokenizer,
)


__all__ = [
    # Dependency Injection
    "DependencyContainer",
    "get_global_container",
    "register_dependency",
    "clear_global_dependencies",
    # Component Loader (class interface)
    "Loader",
    # Component Loader (functional interface - backwards compatibility)
    "load_data",
    "load_model",
    "load_optimizer",
    "load_pipeline",
    "load_scheduler",
    "load_tokenizer",
]
