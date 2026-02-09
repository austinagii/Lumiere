"""
Generic discovery decorator for registering components across the codebase.

This provides a unified @discover(Type, name) decorator that replaces all
the specific decorators like @tokenizer, @dataset, @pipeline, etc.

All registries are created and managed centrally in this module.

The module also provides type-safe register() and get() functions:
    - register(Tokenizer, "bpe", BPETokenizer)
    - get(Tokenizer, "bpe")  # Returns type[Tokenizer] | None
"""

from typing import Any, TypeVar, overload

T = TypeVar("T")

# Registry mapping: type class -> registry
_REGISTRIES: dict[type, Any] = {}


def _create_registries() -> None:
    """Create all registries on first access.

    This is called lazily to avoid circular imports.
    """
    if _REGISTRIES:
        return  # Already initialized

    from lumiere.loading import HierarchicalRegistry, Registry

    # Tokenizer registry
    from lumiere.tokenizer import Tokenizer
    _REGISTRIES[Tokenizer] = Registry[type[Tokenizer]](
        name="tokenizer",
        base_module="lumiere.tokenizers",
        discovery_paths=["."],
    )

    # Dataset registry
    from lumiere.data.dataset import Dataset
    _REGISTRIES[Dataset] = Registry[type[Dataset]](
        name="dataset",
        base_module="lumiere.data.datasets",
        discovery_paths=["."],
    )

    # Pipeline registry
    from lumiere.data.pipeline import Pipeline
    _REGISTRIES[Pipeline] = Registry[type[Pipeline]](
        name="pipeline",
        base_module="lumiere.data.pipelines",
        discovery_paths=["."],
    )

    # Preprocessor registry
    from lumiere.data.preprocessor import Preprocessor
    _REGISTRIES[Preprocessor] = Registry[type[Preprocessor]](
        name="preprocessor",
        base_module="lumiere.data.preprocessors",
        discovery_paths=["."],
    )

    # Optimizer registry
    import torch
    _REGISTRIES[torch.optim.Optimizer] = Registry[type[torch.optim.Optimizer]](
        name="optimizer",
        base_module="lumiere.training.optimizers",
        discovery_paths=["."],
    )

    # Scheduler registry
    _REGISTRIES[torch.optim.lr_scheduler.LRScheduler] = Registry[type](
        name="scheduler",
        base_module="lumiere.training.lr_schedulers",
        discovery_paths=["."],
    )

    # Component registry (hierarchical)
    from torch import nn
    _REGISTRIES[nn.Module] = HierarchicalRegistry[type[nn.Module]](
        name="component",
        base_module="lumiere.nn",
        discovery_paths=["attention", "feedforward", "embedding", "blocks"],
    )


def get_registry(type_cls: type) -> Any:
    """Get the registry for a given type.

    Args:
        type_cls: The type/protocol class (e.g., Tokenizer, Dataset, Pipeline)

    Returns:
        The registry object for the given type.

    Raises:
        ValueError: If no registry exists for the given type.
    """
    _create_registries()
    registry = _REGISTRIES.get(type_cls)
    if registry is None:
        raise ValueError(f"No registry found for type: {type_cls}")
    return registry


def register(type_cls: type[T], name: str, cls: type[T]) -> None:
    """Register a class in the registry for the given type.

    This is a type-safe alternative to calling type-specific register functions.

    For hierarchical registries (like nn.Module), use dot notation in the name
    (e.g., "attention.multihead").

    Args:
        type_cls: The type/protocol class (e.g., Tokenizer, Dataset, Pipeline)
        name: The name/id for this specific implementation (use dot notation for
            hierarchical registries, e.g., "attention.multihead")
        cls: The class to register

    Example:
        >>> from lumiere.tokenizer import Tokenizer
        >>> from lumiere.tokenizers.bpe import BPETokenizer
        >>> register(Tokenizer, "bpe", BPETokenizer)
        >>>
        >>> from torch import nn
        >>> register(nn.Module, "normalization.layer", LayerNorm)
    """
    _create_registries()
    registry = _REGISTRIES.get(type_cls)
    if registry is None:
        raise ValueError(f"No registry found for type: {type_cls}")

    # Handle hierarchical registries (like nn.Module component registry)
    if "." in name and hasattr(registry, "get_by_parts"):
        # Hierarchical registry - split name into parts
        parts = name.split(".", 1)
        if len(parts) == 2:
            type_part, name_part = parts
            registry.register(type_part, name_part, cls)
        else:
            # Fallback
            registry.register(name, cls)
    else:
        # Simple registry
        registry.register(name, cls)


def get(type_cls: type[T], name: str) -> type[T] | None:
    """Get a registered class for the given type and name.

    This is a type-safe alternative to calling type-specific get functions.

    Args:
        type_cls: The type/protocol class (e.g., Tokenizer, Dataset, Pipeline)
        name: The name/id of the implementation to retrieve

    Returns:
        The registered class if found, None otherwise.

    Example:
        >>> from lumiere.tokenizer import Tokenizer
        >>> tokenizer_cls = get(Tokenizer, "bpe")
        >>> # tokenizer_cls is typed as type[Tokenizer] | None
    """
    _create_registries()
    registry = _REGISTRIES.get(type_cls)
    if registry is None:
        raise ValueError(f"No registry found for type: {type_cls}")
    return registry.get(name)


def get_component(component_type: str, component_name: str):
    """Get a hierarchical component (for nn.Module components).

    This is a convenience function for hierarchical registries that use
    dot notation (e.g., "attention.multihead").

    Args:
        component_type: The type/category of component (e.g., "attention")
        component_name: The specific name of the component (e.g., "multihead")

    Returns:
        The registered component class if found, None otherwise.

    Example:
        >>> attention_cls = get_component("attention", "multihead")
    """
    from torch import nn
    _create_registries()
    registry = _REGISTRIES.get(nn.Module)
    if registry is None:
        raise ValueError("No registry found for nn.Module")
    return registry.get_by_parts(component_type, component_name)


def discover(type_cls: type, name: str):
    """Generic decorator for registering components with their registries.

    This replaces all the specific decorators like @tokenizer, @dataset, etc.
    with a single unified decorator using actual type classes.

    Args:
        type_cls: The type/protocol class (e.g., Tokenizer, Dataset, Pipeline)
        name: The name/id for this specific implementation

    Example:
        >>> from lumiere.tokenizer import Tokenizer
        >>> from lumiere.data.dataset import Dataset
        >>> from torch import nn
        >>>
        >>> @discover(Tokenizer, "bpe")
        >>> class BPETokenizer:
        ...     pass
        >>>
        >>> @discover(Dataset, "wikitext")
        >>> class WikiText2Dataset:
        ...     pass
        >>>
        >>> @discover(nn.Module, "attention.multihead")
        >>> class MultiheadAttention(nn.Module):
        ...     pass

    For hierarchical components (like nn.Module), use dot notation:
        @discover(nn.Module, "attention.multihead")
    """
    def decorator(cls: T) -> T:
        _create_registries()
        registry = _REGISTRIES.get(type_cls)
        if registry is None:
            # Registry not found - this shouldn't happen after initialization
            raise ValueError(f"No registry found for type: {type_cls}")

        # Handle hierarchical registries (like nn.Module component registry)
        if "." in name and hasattr(registry, "register"):
            # Hierarchical registry - split name into parts
            parts = name.split(".", 1)
            if len(parts) == 2:
                type_part, name_part = parts
                registry.register(type_part, name_part, cls)
            else:
                # Fallback for registries with manual registration
                registry.register(name, cls)
        elif hasattr(registry, "register"):
            # Simple registry with register method
            registry.register(name, cls)
        else:
            # Should not happen
            raise AttributeError(
                f"Registry for '{type_cls.__name__}' does not have a register method"
            )

        return cls

    return decorator
