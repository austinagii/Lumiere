from typing import Any

from lumiere.loading import Registry


class Preprocessor:
    def __call__(self, *args, **kwargs) -> Any: ...


# A registry of preprocessors indexed by custom names.
_registry = Registry[type[Preprocessor]](
    name="preprocessor",
    base_module="lumiere.data.preprocessors",
    discovery_paths=["."],
)

# Expose existing API for backward compatibility
preprocessor = _registry.decorator
register_preprocessor = _registry.register
get_preprocessor = _registry.get
