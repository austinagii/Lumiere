"""Dependency injection system for managing and resolving dependencies.

This module provides a simple dependency injection container that allows
registering objects by name and type, and resolving them when needed.
"""

from typing import Any, TypeVar


T = TypeVar("T")


class DependencyContainer:
    """Container for managing dependency injection.

    Dependencies can be registered by name or by type, and retrieved
    when needed for object construction.

    Example:
        >>> container = DependencyContainer()
        >>> tokenizer = BPETokenizer()
        >>> container.register("tokenizer", tokenizer)
        >>> container.register_type(Tokenizer, tokenizer)
        >>>
        >>> # Retrieve by name
        >>> tok = container.get("tokenizer")
        >>>
        >>> # Retrieve by type
        >>> tok = container.get_type(Tokenizer)
    """

    def __init__(self):
        """Initialize an empty dependency container."""
        self._dependencies: dict[str, Any] = {}
        self._type_dependencies: dict[type, Any] = {}

    def register(self, name: str, instance: Any) -> None:
        """Register a dependency by name.

        Args:
            name: The name to register the dependency under.
            instance: The dependency instance to register.
        """
        self._dependencies[name] = instance

    def register_type(self, type_: type[T], instance: T) -> None:
        """Register a dependency by type.

        Args:
            type_: The type to register the dependency under.
            instance: The dependency instance to register.
        """
        self._type_dependencies[type_] = instance

    def get(self, name: str, default: Any = None) -> Any:
        """Retrieve a dependency by name.

        Args:
            name: The name of the dependency to retrieve.
            default: Default value to return if dependency is not found.

        Returns:
            The registered dependency, or default if not found.
        """
        return self._dependencies.get(name, default)

    def get_type(self, type_: type[T], default: T | None = None) -> T | None:
        """Retrieve a dependency by type.

        Args:
            type_: The type of the dependency to retrieve.
            default: Default value to return if dependency is not found.

        Returns:
            The registered dependency, or default if not found.
        """
        return self._type_dependencies.get(type_, default)

    def has(self, name: str) -> bool:
        """Check if a dependency is registered by name.

        Args:
            name: The name to check.

        Returns:
            True if the dependency is registered, False otherwise.
        """
        return name in self._dependencies

    def has_type(self, type_: type) -> bool:
        """Check if a dependency is registered by type.

        Args:
            type_: The type to check.

        Returns:
            True if the dependency is registered, False otherwise.
        """
        return type_ in self._type_dependencies

    def clear(self) -> None:
        """Clear all registered dependencies."""
        self._dependencies.clear()
        self._type_dependencies.clear()

    def copy(self) -> "DependencyContainer":
        """Create a shallow copy of this container.

        Returns:
            A new DependencyContainer with the same dependencies.
        """
        new_container = DependencyContainer()
        new_container._dependencies = self._dependencies.copy()
        new_container._type_dependencies = self._type_dependencies.copy()
        return new_container
