from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from lumiere.research.src.components.feedforward import (
    LinearFeedForward,
    SwigluFeedForward,
)

from .transformer import Transformer


SUBKEY_SEPARATOR = "."


class TransformerSpec:
    """A specification for dynamically constructing and configuring a transformer model"""

    def __init__(self, spec: dict[str, Any]):
        self.spec = spec

    @classmethod
    def from_yaml(cls, path: str | Path):
        """Create a Config instance from a file."""
        if isinstance(path, str):
            try:
                path = Path(path)
            except Exception:
                raise ValueError(f"'{path}' is not a valid path")

        if not path.exists():
            raise FileNotFoundError(f"Spec file not found: {path}")

        with open(path, "r") as f:
            spec = yaml.safe_load(f)

        return cls(spec)

    @classmethod
    def from_bytes(cls, data: bytes):
        """Create a Config instance from a file."""
        return cls(yaml.safe_load(data))

    def get(self, key: str) -> Any:
        value = None
        try:
            value = self.__getitem__(key)
        except KeyError:
            pass

        return value

    def __getitem__(self, key: str):
        if not isinstance(key, str) or len(key) == 0:
            raise TypeError("Key must be a non-empty string.")

        key_components = key.split(SUBKEY_SEPARATOR)
        parents = key_components[:-1]
        target = key_components[-1]

        def _get(obj, components, target):
            if len(components) == 0:
                value = obj.get(target)
            else:
                value = _get(obj[components[0]], components[1:], target)
                if value is None:
                    value = obj.get(target)
            return value

        value = _get(self.spec, parents, target)

        if value is None:
            raise KeyError(f"Field '{key}' not found in config")

        return value

    def __delitem__(self, key: str):
        if not isinstance(key, str) or len(key) == 0:
            raise TypeError("Key must be a non-empty string.")

        key_components = key.split(SUBKEY_SEPARATOR)

        obj = self.spec
        for component in key_components[:-1]:
            obj = obj.get(component)

            if obj is None:
                raise KeyError(f"Field '{key}' not found in config")

        del obj[key_components[len(key_components) - 1]]


module_by_type = {
    "linear": LinearFeedForward,
    "swiglu": SwigluFeedForward,
}


class TransformerBuilder:
    """Constructs transformer models from a specification."""

    @staticmethod
    def build(spec: TransformerSpec) -> Transformer:
        """Create a transformer model according to the provided specification.

        Args:
            spec: The specification of the desired transformer.

        Returns:
            A transformer built according to the provided specification.

        Raises:
            ValueError: If no specification is provided.
        """

        # there's a bug here, where block.type, would apply to feedforward if no type
        # is on feedforward.

        # Have a spider crawl the spec and construct the necessary function
        # calls. e.g. traverse spec tree and find attention, check type, see that type is
        # x, create a builder for x, using the various attention subfields, assign builder to
        # keyword attention in parent call.

        transformer_args = deepcopy(spec.spec)

        def _resolve_nested_factories(spec):
            for key, value in spec.items():
                if isinstance(value, dict):
                    assert (module_type := value.get("type")) is not None
                    del value["type"]  # the type of module is no longer needed.

                    # Could counsider prepending module name to disambiguate.
                    # type "linear" under the "feedforward" section would become
                    # "feedforward.linear".
                    module = module_by_type.get(module_type)
                    if module is None:
                        raise ValueError(f"Unrecognized module type: {module_type}")

                    module_args = _resolve_nested_factories(value)
                    module_factory = _wrap_factory(module, **module_args)
                    spec[key] = module_factory

            return spec

        _resolve_nested_factories(transformer_args)

        return Transformer(**transformer_args)


def _wrap_factory(cls, *args, **kwargs) -> Callable[[Any, ...], Any]:
    def factory():
        return cls(*args, **kwargs)

    return factory
