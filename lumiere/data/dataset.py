import contextlib
import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, TypeVar


T = TypeVar("T")


class Dataset(Protocol):
    """Protocol defining the interface for dataset implementations.

    All dataset classes must implement the __getitem__ method to provide
    access to different data splits (e.g., train, validation, test).

    """

    def __getitem__(self, split_name: str) -> Iterator[T]:
        """Return an iterator over samples from the specified split.

        Args:
            split_name: Name of the split to access (e.g., "train", "valid", "test").

        Returns:
            An Iterator over samples from the specified split.

        Raises:
            KeyError: If the specified split is not present in the dataset.
        """
        ...


# A registry of datasets indexed by custom names.
_dataset_registry: dict[str, type[Dataset]] = {}


def dataset(dataset_name: str):
    """Decorator to register a dataset class in the global registry.

    Registered datasets can be retrieved by name using get_dataset().

    Args:
        dataset_name: Unique identifier for the dataset in the registry.

    """

    def decorator(cls):
        register_dataset(dataset_name, cls)
        return cls

    return decorator


def register_dataset(name: str, cls: type[Dataset]) -> None:
    _dataset_registry[name] = cls


def get_dataset(dataset_name: str) -> type[Dataset] | None:
    """Retrieve a dataset class from the registry by name.

    Args:
        dataset_name: Registered identifier of the dataset to retrieve.

    Returns:
        Dataset class if found in the registry, None otherwise.
    """
    if not _dataset_registry:  # Refresh the imports.
        datasets_dir = Path(__file__).parent / "datasets"
        module_files = datasets_dir.glob("*.py")
        module_files = [f for f in module_files if not f.stem.startswith("_")]

        # Import each module to trigger @dataset decorator registration
        for module_file in module_files:
            module_name = f"lumiere.data.datasets.{module_file.stem}"
            with contextlib.suppress(ImportError):
                importlib.import_module(module_name)

    return _dataset_registry.get(dataset_name)
