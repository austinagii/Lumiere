"""
Generic registry implementation for Lumière loaders.

Provides base classes for managing registered components with decorator-based
registration and automatic module discovery.
"""

import importlib
import pkgutil
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """
    Generic registry for managing registered components.

    Provides decorator-based registration, manual registration, and automatic
    discovery of components in specified modules.

    Type Parameters:
        T: The type of objects stored in this registry

    Args:
        name: Human-readable name for this registry (e.g., "tokenizer", "optimizer")
        base_module: Base module path for auto-discovery (e.g., "lumiere.tokenizers")
        discovery_paths: List of submodule paths to search during discovery (e.g., ["."])

    Example:
        >>> registry = Registry[type[Tokenizer]](
        ...     name="tokenizer",
        ...     base_module="lumiere.tokenizers",
        ...     discovery_paths=["."],
        ... )
        >>>
        >>> # Use as decorator
        >>> @registry.decorator("bpe")
        ... class BPETokenizer:
        ...     pass
        >>>
        >>> # Manual registration
        >>> registry.register("custom", CustomTokenizer)
        >>>
        >>> # Lookup (triggers auto-discovery on first call)
        >>> tokenizer_cls = registry.get("bpe")
    """

    def __init__(
        self,
        name: str,
        base_module: str,
        discovery_paths: list[str],
    ):
        self.name = name
        self.base_module = base_module
        self.discovery_paths = discovery_paths
        self._registry: dict[str, T] = {}
        self._discovered = False

    def register(self, key: str, value: T) -> None:
        """
        Manually register a component.

        Args:
            key: Registration key (e.g., "bpe", "adamw")
            value: Component to register
        """
        self._registry[key] = value

    def decorator(self, key: str) -> Callable[[T], T]:
        """
        Create a decorator for registering components.

        Args:
            key: Registration key for the decorated component

        Returns:
            Decorator function that registers the component and returns it unchanged

        Example:
            >>> @registry.decorator("bpe")
            ... class BPETokenizer:
            ...     pass
        """
        def _decorator(cls: T) -> T:
            self.register(key, cls)
            return cls
        return _decorator

    def get(self, key: str) -> T | None:
        """
        Get a registered component by key.

        Triggers auto-discovery on first call to ensure all components are registered.

        Args:
            key: Registration key to lookup

        Returns:
            The registered component, or None if not found
        """
        if not self._discovered:
            self._discover()
        return self._registry.get(key)

    def list_keys(self) -> list[str]:
        """
        List all registered keys.

        Triggers auto-discovery on first call.

        Returns:
            List of all registration keys
        """
        if not self._discovered:
            self._discover()
        return list(self._registry.keys())

    def _discover(self) -> None:
        """
        Automatically discover and import modules to trigger registrations.

        Searches for modules in the configured discovery paths and imports them,
        which triggers any @decorator registrations they contain.
        """
        if self._discovered:
            return

        self._discovered = True

        try:
            base = importlib.import_module(self.base_module)
        except ImportError:
            return

        for discovery_path in self.discovery_paths:
            if discovery_path == ".":
                # Import all modules in the base package
                if hasattr(base, "__path__"):
                    for _, module_name, _ in pkgutil.iter_modules(base.__path__):
                        try:
                            importlib.import_module(f"{self.base_module}.{module_name}")
                        except ImportError:
                            continue
            else:
                # Import modules in a specific subpackage
                try:
                    subpackage = importlib.import_module(f"{self.base_module}.{discovery_path}")
                    if hasattr(subpackage, "__path__"):
                        for _, module_name, _ in pkgutil.iter_modules(subpackage.__path__):
                            try:
                                importlib.import_module(
                                    f"{self.base_module}.{discovery_path}.{module_name}"
                                )
                            except ImportError:
                                continue
                except ImportError:
                    continue


class HierarchicalRegistry(Generic[T]):
    """
    Registry for components with hierarchical keys.

    Supports two-part keys like "attention.multihead" or ("attention", "multihead").
    Used for component registries where components are organized by type.

    Type Parameters:
        T: The type of objects stored in this registry

    Args:
        name: Human-readable name for this registry (e.g., "component")
        base_module: Base module path for auto-discovery (e.g., "lumiere.nn")
        discovery_paths: List of submodule paths to search (e.g., ["attention", "feedforward"])

    Example:
        >>> registry = HierarchicalRegistry[type[nn.Module]](
        ...     name="component",
        ...     base_module="lumiere.nn",
        ...     discovery_paths=["attention", "feedforward"],
        ... )
        >>>
        >>> # Use as decorator with two-part key
        >>> @registry.decorator("attention", "multihead")
        ... class MultiheadAttention(nn.Module):
        ...     pass
        >>>
        >>> # Lookup by parts
        >>> attention_cls = registry.get_by_parts("attention", "multihead")
        >>>
        >>> # Or by combined key
        >>> attention_cls = registry.get("attention.multihead")
    """

    def __init__(
        self,
        name: str,
        base_module: str,
        discovery_paths: list[str],
    ):
        self.name = name
        self.base_module = base_module
        self.discovery_paths = discovery_paths
        self._registry: dict[str, dict[str, T]] = {}
        self._discovered = False

    def register(self, type_key: str, name_key: str, value: T) -> None:
        """
        Manually register a component.

        Args:
            type_key: First part of the key (e.g., "attention")
            name_key: Second part of the key (e.g., "multihead")
            value: Component to register
        """
        if type_key not in self._registry:
            self._registry[type_key] = {}
        self._registry[type_key][name_key] = value

    def decorator(self, type_key: str, name_key: str) -> Callable[[T], T]:
        """
        Create a decorator for registering components.

        Args:
            type_key: First part of the key (e.g., "attention")
            name_key: Second part of the key (e.g., "multihead")

        Returns:
            Decorator function that registers the component and returns it unchanged

        Example:
            >>> @registry.decorator("attention", "multihead")
            ... class MultiheadAttention(nn.Module):
            ...     pass
        """
        def _decorator(cls: T) -> T:
            self.register(type_key, name_key, cls)
            return cls
        return _decorator

    def get_by_parts(self, type_key: str, name_key: str) -> T | None:
        """
        Get a registered component by two-part key.

        Triggers auto-discovery on first call.

        Args:
            type_key: First part of the key (e.g., "attention")
            name_key: Second part of the key (e.g., "multihead")

        Returns:
            The registered component, or None if not found
        """
        if not self._discovered:
            self._discover()
        return self._registry.get(type_key, {}).get(name_key)

    def get(self, combined_key: str) -> T | None:
        """
        Get a registered component by combined key.

        Args:
            combined_key: Combined key like "attention.multihead"

        Returns:
            The registered component, or None if not found
        """
        if "." not in combined_key:
            return None
        type_key, name_key = combined_key.split(".", 1)
        return self.get_by_parts(type_key, name_key)

    def list_keys(self) -> list[tuple[str, str]]:
        """
        List all registered keys as (type, name) tuples.

        Triggers auto-discovery on first call.

        Returns:
            List of all registration keys as tuples
        """
        if not self._discovered:
            self._discover()
        keys = []
        for type_key, names in self._registry.items():
            for name_key in names:
                keys.append((type_key, name_key))
        return keys

    def _discover(self) -> None:
        """
        Automatically discover and import modules to trigger registrations.

        Searches for modules in the configured discovery paths and imports them.
        """
        if self._discovered:
            return

        self._discovered = True

        try:
            base = importlib.import_module(self.base_module)
        except ImportError:
            return

        for discovery_path in self.discovery_paths:
            try:
                subpackage = importlib.import_module(f"{self.base_module}.{discovery_path}")
                if hasattr(subpackage, "__path__"):
                    for _, module_name, _ in pkgutil.iter_modules(subpackage.__path__):
                        try:
                            importlib.import_module(
                                f"{self.base_module}.{discovery_path}.{module_name}"
                            )
                        except ImportError:
                            continue
            except ImportError:
                continue
