import contextlib
import importlib
from pathlib import Path
from typing import Protocol


class Pipeline(Protocol):
    def batches(self, data): ...


# A registry of pipelines indexed by custom names.
_pipeline_registry: dict[str, type[Pipeline]] = {}


def pipeline(pipeline_name: str):
    """Decorator to register a pipeline class in the global registry.

    Registered pipelines can be retrieved by name using get_pipeline().

    Args:
        pipeline_name: Unique identifier for the pipeline in the registry.

    """

    def decorator(cls):
        register_pipeline(pipeline_name, cls)
        return cls

    return decorator


def register_pipeline(name: str, cls: type[Pipeline]) -> None:
    _pipeline_registry[name] = cls


def get_pipeline(pipeline_name: str) -> type[Pipeline] | None:
    """Retrieve a pipeline class from the registry by name.

    Args:
        pipeline_name: Registered identifier of the pipeline to retrieve.

    Returns:
        Pipeline class if found in the registry, None otherwise.
    """
    if not _pipeline_registry:  # Refresh the imports.
        pipelines_dir = Path(__file__).parent / "pipelines"
        module_files = pipelines_dir.glob("*.py")
        module_files = [f for f in module_files if not f.stem.startswith("_")]

        # Import each module to trigger @pipeline decorator registration
        for module_file in module_files:
            module_name = f"lumiere.data.pipelines.{module_file.stem}"
            with contextlib.suppress(ImportError):
                importlib.import_module(module_name)

    return _pipeline_registry.get(pipeline_name)
