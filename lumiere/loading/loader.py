"""
Generic config-based loader implementation for Lumière.

Provides a base class for creating loaders that instantiate components from
configuration dictionaries with dependency injection support.
"""

from typing import Any, Callable, Generic, TypeVar

from lumiere.loading.registry import Registry
from lumiere.loading.resolver import resolve_value

T = TypeVar("T")


class ConfigLoader(Generic[T]):
    """
    Generic config-based loader with dependency injection.

    Handles the common pattern of loading components from configuration dictionaries:
    1. Validate config has required fields (name/type)
    2. Look up component class in registry
    3. Resolve dependencies using @variable syntax
    4. Instantiate component with resolved config
    5. Handle errors with helpful messages

    Type Parameters:
        T: The type of objects this loader creates

    Args:
        registry: Registry to look up component classes
        required_params: List of parameter names that must be passed to load()
            (e.g., ["params"] for optimizer loader, ["optimizer"] for scheduler loader)
        nested_loaders: Dict mapping config keys to loader functions for nested components
            (e.g., {"datasets": load_dataset} for dataset loader that loads multiple datasets)
        name_field: Config field containing the component name (default: "name")

    Example:
        >>> # Simple loader
        >>> registry = Registry[type[Tokenizer]](...)
        >>> loader = ConfigLoader(registry)
        >>> tokenizer = loader.load({"name": "bpe", "vocab_size": 50000})
        >>>
        >>> # Loader with required params
        >>> opt_registry = Registry[type[Optimizer]](...)
        >>> opt_loader = ConfigLoader(opt_registry, required_params=["params"])
        >>> optimizer = opt_loader.load({"name": "adamw", "lr": 0.001}, model.parameters())
        >>>
        >>> # Loader with nested components
        >>> dataset_loader = ConfigLoader(
        ...     registry,
        ...     nested_loaders={"datasets": load_single_dataset}
        ... )
        >>> combined = dataset_loader.load({"datasets": [{"name": "wikitext"}, {"name": "bookcorpus"}]})
    """

    def __init__(
        self,
        registry: Registry[type[T]],
        required_params: list[str] | None = None,
        nested_loaders: dict[str, Callable] | None = None,
        name_field: str = "name",
    ):
        self.registry = registry
        self.required_params = required_params or []
        self.nested_loaders = nested_loaders or {}
        self.name_field = name_field

    def load(self, config: dict[str, Any], *args, container: Any = None, **kwargs) -> T:
        """
        Load a component from a configuration dictionary.

        Args:
            config: Configuration dictionary containing at minimum the name field
            *args: Required positional arguments (e.g., model parameters for optimizer)
            container: Optional DependencyContainer for resolving @variable references
            **kwargs: Additional keyword arguments to pass to component constructor

        Returns:
            Instantiated component

        Raises:
            ValueError: If config is invalid or component not found
            TypeError: If required parameters are missing

        Example:
            >>> loader.load({"name": "adamw", "lr": 0.001}, model.parameters())
            AdamW(...)
            >>> loader.load({"name": "bpe", "vocab_size": "@vocab_size"}, container=deps)
            BPETokenizer(vocab_size=50000)
        """
        # Validate config
        if not isinstance(config, dict):
            raise ValueError(
                f"Config must be a dictionary, got {type(config).__name__}"
            )

        if self.name_field not in config:
            raise ValueError(
                f"Config must contain '{self.name_field}' field, got keys: {list(config.keys())}"
            )

        component_name = config[self.name_field]

        # Handle nested loaders (e.g., dataset loader with multiple datasets)
        for nested_key, nested_loader in self.nested_loaders.items():
            if nested_key in config:
                nested_configs = config[nested_key]
                if not isinstance(nested_configs, list):
                    nested_configs = [nested_configs]

                # Load each nested component
                nested_components = []
                for nested_config in nested_configs:
                    nested_component = nested_loader(nested_config, container=container)
                    nested_components.append(nested_component)

                # Update config with loaded components
                config = config.copy()
                config[nested_key] = nested_components

        # Look up component class in registry
        component_cls = self.registry.get(component_name)
        if component_cls is None:
            available = self.registry.list_keys()
            raise ValueError(
                f"{self.registry.name.capitalize()} '{component_name}' not found. "
                f"Available: {available}"
            )

        # Resolve dependencies in config
        resolved_config = {}
        for key, value in config.items():
            if key != self.name_field:  # Skip the name field
                resolved_config[key] = resolve_value(value, container)

        # Merge with kwargs
        resolved_config.update(kwargs)

        # Validate required parameters
        for param in self.required_params:
            if not args and param not in resolved_config:
                raise TypeError(
                    f"Loading {self.registry.name} '{component_name}' requires '{param}' parameter"
                )

        # Instantiate component
        try:
            if args:
                return component_cls(*args, **resolved_config)
            else:
                return component_cls(**resolved_config)
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate {self.registry.name} '{component_name}': {e}"
            ) from e
