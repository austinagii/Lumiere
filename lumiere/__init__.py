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
from .lumiere import load_template, resume, start
from .utils.signals import register_signal_handlers


__all__ = [
    "DependencyContainer",
    "get_global_container",
    "register_dependency",
    "clear_global_dependencies",
    "Loader",
    "load_data",
    "load_model",
    "load_optimizer",
    "load_pipeline",
    "load_scheduler",
    "load_tokenizer",
    "start",
    "resume",
    "load_template",
]

register_signal_handlers()
