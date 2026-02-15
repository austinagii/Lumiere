"""Simple discovery system for registering components.

Components are registered using @discover(Type, "name") and stored in a simple
nested dictionary. Autodiscovery happens on import by reading lumiere.yaml.
"""

import importlib
import pkgutil
from pathlib import Path
from typing import TypeVar

import yaml


T = TypeVar("T")

_REGISTRY: dict[type, dict[str, type]] = {}


def discover(type_cls: type, name: str):
    """Register a class with the discovery system.

    Args:
        type_cls: The type/protocol class (e.g., Tokenizer, Dataset)
        name: The name/id for this implementation

    Example:
        ```python
        @discover(Tokenizer, "bpe")
        class BPETokenizer:
            pass
        ```
    """

    def decorator(cls: T) -> T:
        if type_cls not in _REGISTRY:
            _REGISTRY[type_cls] = {}
        _REGISTRY[type_cls][name] = cls
        return cls

    return decorator


def register(type_cls: type[T], name: str, cls: type[T]) -> None:
    """Manually register a class.

    Args:
        type_cls: The type/protocol class
        name: The name/id for this implementation
        cls: The class to register
    """
    if type_cls not in _REGISTRY:
        _REGISTRY[type_cls] = {}
    _REGISTRY[type_cls][name] = cls


def get(type_cls: type[T], name: str) -> type[T] | None:
    """Get a registered class by type and name.

    Args:
        type_cls: The type/protocol class
        name: The name/id of the implementation

    Returns:
        The registered class if found, None otherwise
    """
    return _REGISTRY.get(type_cls, {}).get(name)


def get_component(component_type: str, component_name: str):
    """Get a component by type and name (for nn.Module).

    Args:
        component_type: Component category (e.g., "attention")
        component_name: Specific component (e.g., "multihead")

    Returns:
        The registered component class if found
    """
    from torch import nn

    full_name = f"{component_type}.{component_name}"
    return get(nn.Module, full_name)


def _autodiscover():
    """Autodiscover modules from lumiere.yaml configuration."""
    # Find lumiere.yaml
    current_path = Path.cwd()
    config_path = None

    for _ in range(5):
        candidate = current_path / "lumiere.yaml"
        if candidate.exists():
            config_path = candidate
            break
        if current_path.parent == current_path:
            break
        current_path = current_path.parent

    if not config_path:
        return

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception:
        return

    # Get discovery config
    discovery_config = config.get("discovery", {})
    registries_config = discovery_config.get("registries", {})

    # Import modules for each registry
    for registry_info in registries_config.values():
        base_module = registry_info.get("base_module")
        discovery_paths = registry_info.get("discovery_paths", [])

        if not base_module:
            continue

        try:
            # Import base module
            base = importlib.import_module(base_module)
        except ImportError:
            continue

        # Import submodules
        for discovery_path in discovery_paths:
            if discovery_path == ".":
                # Import all modules in base package
                if hasattr(base, "__path__"):
                    for _, module_name, _ in pkgutil.iter_modules(base.__path__):
                        try:
                            importlib.import_module(f"{base_module}.{module_name}")
                        except ImportError:
                            pass
            else:
                # Import modules in specific subpackage
                try:
                    subpackage = importlib.import_module(
                        f"{base_module}.{discovery_path}"
                    )
                    if hasattr(subpackage, "__path__"):
                        for _, module_name, _ in pkgutil.iter_modules(
                            subpackage.__path__
                        ):
                            try:
                                importlib.import_module(
                                    f"{base_module}.{discovery_path}.{module_name}"
                                )
                            except ImportError:
                                pass
                except ImportError:
                    pass


# Run autodiscovery on import
_autodiscover()
