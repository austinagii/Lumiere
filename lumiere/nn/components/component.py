"""Component registry system for neural network modules."""

from typing import Any

from torch import nn
from torch.nn import LayerNorm, RMSNorm


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

    from lumiere.discover import get_component
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
    from lumiere.loader import resolve_value

    return resolve_value(value, container)


# Register PyTorch normalization layers at module import time
from lumiere.discover import register
register(nn.Module, "normalization.layer", LayerNorm)
register(nn.Module, "normalization.rms", RMSNorm)
