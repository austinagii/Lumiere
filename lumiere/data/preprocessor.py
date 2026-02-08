import contextlib
import importlib
from pathlib import Path
from typing import Any


class Preprocessor:
    def __call__(self, *args, **kwargs) -> Any: ...


# A registry of preprocessors indexed by custom names.
_preprocessor_registry: dict[str, type[Preprocessor]] = {}


def preprocessor(preprocessor_name: str):
    """Decorator to register a preprocessor class in the global registry.

    Registered preprocessors can be retrieved by name using get_preprocessor().

    Args:
        preprocessor_name: Unique identifier for the preprocessor in the registry.

    """

    def decorator(cls):
        register_preprocessor(preprocessor_name, cls)
        return cls

    return decorator


def register_preprocessor(name: str, cls: type[Preprocessor]) -> None:
    _preprocessor_registry[name] = cls


def get_preprocessor(preprocessor_name: str) -> type[Preprocessor] | None:
    """Retrieve a preprocessor class from the registry by name.

    Args:
        preprocessor_name: Registered identifier of the preprocessor to retrieve.

    Returns:
        Preprocessor class if found in the registry, None otherwise.
    """
    if not _preprocessor_registry:  # Refresh the imports.
        preprocessors_dir = Path(__file__).parent / "preprocessors"
        module_files = preprocessors_dir.glob("*.py")
        module_files = [f for f in module_files if not f.stem.startswith("_")]

        # Import each module to trigger @preprocessor decorator registration
        for module_file in module_files:
            module_name = f"lumiere.data.preprocessors.{module_file.stem}"
            with contextlib.suppress(ImportError):
                importlib.import_module(module_name)

    return _preprocessor_registry.get(preprocessor_name)
