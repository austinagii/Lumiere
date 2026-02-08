"""Component registry system for neural network modules."""

from typing import Any

from torch import nn
from torch.nn import LayerNorm, RMSNorm

from lumiere.loading import HierarchicalRegistry

# Registry for all neural network components
_registry = HierarchicalRegistry[type[nn.Module]](
    name="component",
    base_module="lumiere.nn",
    discovery_paths=["attention", "feedforward", "embedding", "blocks"],
)


def component(component_type: str, component_name: str):
    """Decorator to register a component in the global registry.

    Args:
        component_type: The type/category of component (e.g., 'attention', 'feedforward').
        component_name: The specific name of this component (e.g., 'multihead', 'linear').

    Example:
        >>> @component('attention', 'multihead')
        >>> class MultiHeadAttention(nn.Module):
        ...     pass
    """
    return _registry.decorator(component_type, component_name)


def register_component(key: str, cls: type[nn.Module]) -> None:
    """Register a component class in the registry.

    Args:
        key: The full key in format 'type.name' (e.g., 'attention.multihead').
        cls: The component class to register.
    """
    if "." not in key:
        raise ValueError(f"Component key must be in format 'type.name', got: {key}")
    component_type, component_name = key.split(".", 1)
    _registry.register(component_type, component_name, cls)


def get_component(component_type: str, component_name: str) -> type[nn.Module] | None:
    """Retrieve a component class from the registry.

    Args:
        component_type: The type/category of component.
        component_name: The specific name of the component.

    Returns:
        Component class if found, None otherwise.
    """
    return _registry.get_by_parts(component_type, component_name)


def create_factory(config: dict[str, Any], container: Any = None):
    """Create a factory function from a component configuration.

    This function creates a factory (callable) that can be used by the
    TransformerBuilder. The factory, when called, will instantiate the
    component with the configured parameters.

    Args:
        config: Configuration dict with 'type' and 'name' fields plus parameters.
        container: Optional DependencyContainer for resolving dependencies.

    Returns:
        A callable factory function that creates the component.

    Example:
        >>> config = {
        ...     "type": "attention",
        ...     "name": "multihead",
        ...     "num_heads": 8,
        ...     "d_key": 64
        ... }
        >>> factory = create_factory(config)
        >>> attention_module = factory()
    """
    component_type = config.get("type")
    component_name = config.get("name")

    if component_type is None or component_name is None:
        raise ValueError(
            "Component config must contain 'type' and 'name' fields. "
            f"Got: {config}"
        )

    component_cls = get_component(component_type, component_name)
    if component_cls is None:
        raise ValueError(
            f"Component '{component_type}.{component_name}' not found in registry."
        )

    # Extract parameters (everything except 'type' and 'name')
    params = {k: v for k, v in config.items() if k not in ("type", "name")}

    # Resolve dependencies if container is provided
    if container is not None:
        params = {
            k: _resolve_value(v, container, f"{component_type}.{component_name}", k)
            for k, v in params.items()
        }

    # Return a factory function
    def factory(**override_params):
        # Merge override params with configured params
        final_params = {**params, **override_params}
        return component_cls(**final_params)

    return factory


def _resolve_value(value: Any, container: Any, context: str, key: str) -> Any:
    """Resolve a config value, handling dependency injection references."""
    from lumiere.loading import resolve_value

    return resolve_value(value, container)


# Register PyTorch normalization layers at module import time
register_component("normalization.layer", LayerNorm)
register_component("normalization.rms", RMSNorm)
