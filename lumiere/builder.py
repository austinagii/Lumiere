"""Classes for dynamically building a Transformer model from a specification."""

import inspect
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from torch import nn
from torch.nn import LayerNorm, RMSNorm

from lumiere.nn.architectures.transformer import Transformer
from lumiere.nn.components.component import create_factory


class ModelSpec:
    """A specification for dynamically constructing a model.

    This class represents a hierarchical configuration that maps directly to the
    initialization parameters of a model and its nested components. The specification
    structure should mirror the model's component hierarchy, with each nested dictionary
    defining both the desired sub-module type and its initialization parameters.

    For example, if a transformer model requires 'context_size', 'embedding_size', and
    'block' (where 'block' is a callable producing transformer block instances),
    and a 'standard' block type requires 'hidden_size' and 'dropout', the
    specification would then include both sets of parameters in a nested structure:

        context_size: 10
        embedding_size: 16
        block:
            type: standard
            hidden_size: 24
            dropout: 0.1

    The 'type' field specifies which module to instantiate, while other nested
    parameters are passed to that module's initializer. Top-level parameters are
    passed to the transformer's initializer.

    In cases where a sub-module's initializer shares parameters in common with
    an ancestor, then those parameters can be omitted from the sub-module's spec
    and the corresponding values will be inherited from the nearest ancestor that
    defines them.

        context_size: 10
        embedding_size: 16
        block:
            type: swiglu
            # embedding_size: 16  <-- This value will be inherited if omitted.
            hidden_size: 24
            dropout: 0.1

    """

    def __init__(self, args: dict[str, Any]):
        """Initialize a model specification.

        Args:
            args: The arguments for the model and its components.

        """
        if args is None:
            raise ValueError("A dict of spec arguments is required.")

        self.args = args

    @classmethod
    def from_yaml(cls, path: str | Path):
        """Create a model spec instance from a YAML file.

        Args:
            path: The path to the desired YAML file.

        Returns:
            A spec matching the contents of the YAML file.

        Raises:
            ValueError: If the path is not a valid path.
            FileNotFoundError: If the YAML file does not exist.

        """
        if isinstance(path, str):
            try:
                path = Path(path)
            except Exception as e:
                raise ValueError(f"'{path}' is not a valid path") from e

        if not isinstance(path, Path):
            raise ValueError("'path' must be a string or Path object.")

        if not path.exists():
            raise FileNotFoundError(f"Specfication file not found: {path}")

        with open(path) as f:
            spec = yaml.safe_load(f)

        return cls(spec)

    def __getitem__(self, argname: str):
        """Return the value of the specified argument.

        For ease of use, when retrieving the value of an argument for a sub-module, dot
        notation can be used to specify the path to the desired argument.

        Args:
            argname: The name of the argument.

        Returns:
            The value of the argument.

        Raises:
            TypeError: If argname is not a non-empty string.
            KeyError: If the argument is not found in the specification.

        Example:
        >>> spec = TransformerSpec(
        ...     {
        ...         'embedding_size': 1024,
        ...         'block': {
        ...             'hidden_size': 512
        ...         }
        ...     }
        ... )
        >>> spec['block.hidden_size']
        512
        """
        if not isinstance(argname, str) or len(argname) == 0:
            raise TypeError("Argument name must be a non-empty string.")

        key_components = argname.split(".")
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

        value = _get(self.args, parents, target)

        if value is None:
            raise KeyError(f"Argument '{argname}' not found in specification.")

        return value

    def get(self, argname: str) -> Any:
        """Return the value of the specified argument in this specification.

        This method is equivalent to reading the argument using square bracket notation
        but returns `None` when the specified argument could not be found, instead of
        raising a KeyError.

        """
        value = None
        try:
            value = self.__getitem__(argname)
        except KeyError:
            pass

        return value

    def __setitem__(self, argname: str, value: Any) -> None:
        """Set the value of the specified argument.

        For ease of use, when setting the value of an argument for a sub-module, dot
        notation can be used to specify the path to the desired argument.

        Args:
            argname: The name of the argument to be modified.
            value: The desired value of the argument.

        Raises:
            TypeError: If argname is not a non-empty string.

        Example:
        >>> spec = ModelSpec(
        ...     {
        ...         'embedding_size': 1024,
        ...         'block': {
        ...             'hidden_size': 512
        ...         }
        ...     }
        ... )
        >>> spec['block.feedforward.d_proj_up'] = 10

        """
        if not isinstance(argname, str) or len(argname) == 0:
            raise TypeError("Argument name must be a non-empty string.")

        key_components = argname.split(".")

        obj = self.args
        for component in key_components[:-1]:
            if (next_obj := obj.get(component)) is None:
                obj[component] = dict()
                next_obj = obj[component]

            obj = next_obj

        obj[key_components[-1]] = value

    def __delitem__(self, key: str):
        """Delete the specified argument.

        This method removes the specified argument from this specification's argument
        dict.

        For ease of use, when deleting an argument for a sub-module, dot notation
        can be used to specify the path to the desired argument.

        Args:
            key: The name of the argument to be deleted.

        Raises:
            TypeError: If key is not a non-empty string.
            KeyError: If the argument is not found in the specification.

        """
        if not isinstance(key, str) or len(key) == 0:
            raise TypeError("Key must be a non-empty string.")

        key_components = key.split(".")

        obj = self.args
        for component in key_components[:-1]:
            obj = obj.get(component)

            if obj is None:
                raise KeyError(f"Argument '{key}' not found in specification")

        del obj[key_components[len(key_components) - 1]]


class TransformerBuilder:
    """A builder for constructing transformer models from a specification."""

    @staticmethod
    def build(spec: ModelSpec, container=None) -> nn.Module:
        """Create a model according to the provided specification.

        Args:
            spec: The specification of the desired transformer.

        Returns:
            A transformer built according to the provided specification.

        Raises:
            ValueError: If an unrecognized module type is specified or if a required
                parameter for a module is not found in the specification.

        """
        transformer_args = deepcopy(spec.args)

        def _resolve_nested_factories(label, spec, parent_params, container=container):
            """Perform a single-pass depth-first traversal of the spec tree.

            Replaces child specs with factory functions that produce modules
            according to that spec. Parameters are inherited from ancestors by
            passing accumulated context downward.
            """
            # Merge parent params with current spec (current takes precedence)
            available_params = {**parent_params, **spec}

            # Recursively resolve child specs
            for key, value in spec.items():
                if isinstance(value, dict):  # This value is a child spec
                    resolved = _resolve_nested_factories(
                        key,
                        value,
                        available_params,  # Pass down accumulated params
                        container=container,  # Pass down container
                    )
                    spec[key] = resolved
                    # Update available_params so siblings can access resolved factories
                    available_params[key] = resolved

            # If this spec defines a module type and name, create a factory for it
            if "type" in spec and "name" in spec:
                # Get the component class to check which params it accepts
                from lumiere.discover import get_component
                component_type = spec.get("type")
                component_name = spec.get("name")
                component_cls = get_component(component_type, component_name)

                if component_cls:
                    # Get the component's __init__ signature
                    sig = inspect.signature(component_cls.__init__)
                    # Build factory config with only accepted params
                    factory_config = {"type": component_type, "name": component_name}

                    for param_name in sig.parameters:
                        if param_name == "self":
                            continue
                        # Check spec first, then available_params
                        if param_name in spec:
                            factory_config[param_name] = spec[param_name]
                        elif param_name in available_params:
                            factory_config[param_name] = available_params[param_name]

                    factory = create_factory(factory_config, container=container)
                else:
                    raise ValueError(f"Component '{component_type}.{component_name}' not found")

                # Clean up type and name from spec
                del spec["type"]
                del spec["name"]

                return factory
            else:
                return spec

        _resolve_nested_factories("", transformer_args, {})

        # Filter transformer_args to only include parameters accepted by Transformer
        transformer_params = inspect.signature(Transformer).parameters
        filtered_args = {
            key: value
            for key, value in transformer_args.items()
            if key in transformer_params
        }

        return Transformer(**filtered_args)
