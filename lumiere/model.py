"""Model registry system for dynamically loading models."""

import contextlib
import importlib
from pathlib import Path
from typing import Protocol

import torch.nn as nn


class Model(Protocol):
    """Protocol defining the interface for model implementations."""

    def forward(self, *args, **kwargs):
        """Forward pass of the model."""
        ...


# A registry of models indexed by custom names.
_model_registry: dict[str, type[nn.Module]] = {}


def model(model_name: str):
    """Decorator to register a model class in the global registry.

    Registered models can be retrieved by name using get_model().

    Args:
        model_name: Unique identifier for the model in the registry.

    """

    def decorator(cls):
        register_model(model_name, cls)
        return cls

    return decorator


def register_model(name: str, cls: type[nn.Module]) -> None:
    """Register a model class in the registry.

    Args:
        name: The name to register the model under.
        cls: The model class to register.
    """
    _model_registry[name] = cls


def get_model(model_name: str) -> type[nn.Module] | None:
    """Retrieve a model class from the registry by name.

    Args:
        model_name: Registered identifier of the model to retrieve.

    Returns:
        Model class if found in the registry, None otherwise.
    """
    if not _model_registry:  # Refresh the imports.
        models_dir = Path(__file__).parent / "models"
        if not models_dir.exists():
            return None

        module_files = models_dir.glob("*.py")
        module_files = [f for f in module_files if not f.stem.startswith("_")]

        # Import each module to trigger @model decorator registration
        for module_file in module_files:
            module_name = f"lumiere.models.{module_file.stem}"
            with contextlib.suppress(ImportError):
                importlib.import_module(module_name)

    return _model_registry.get(model_name)
