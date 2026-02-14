"""Classes for dynamically building a Transformer model from a specification."""

import copy
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from torch import nn

from lumiere.internal.di import DependencyContainer
from lumiere.internal.registry import get_component


def load(
    config: Mapping[str, Any], container: DependencyContainer | None = None
) -> nn.Module:
    """Load and return a Model instance from a hierarchical configuration.

    This loader uses the TransformerBuilder to construct models from hierarchical
    specifications with factory configurations. It supports dependency injection
    for all configuration values.

    Args:
        config: Hierarchical configuration dictionary. Factory fields (like
            'embedding_factory', 'block_factory') should contain 'type' and
            'name' fields plus component-specific parameters.
        container: Optional DependencyContainer for resolving dependencies.

    Returns:
        Initialized Model instance.

    Raises:
        ValueError: If a dependency cannot be resolved or component not found.
        RuntimeError: If an error occurs during model initialization.

    Example:
        ```python
        config = {
            "vocab_size": 30000,
            "context_size": 512,
            "num_blocks": 12,
            "embedding": {
                "name": "sinusoidal",
                "padding_id": 0
            },
            "block": {
                "name": "standard",
                "attention": {
                    "name": "multihead",
                    "num_heads": 8
                },
                "feedforward": {
                    "name": "linear",
                    "d_ff": 2048
                },
                "normalization": {
                    "name": "rms"
                }
            },
            "normalization": {
                "name": "rms"
            }
        }
        model = load(config)
        ```
    """
    try:
        # Build the model using TransformerBuilder
        spec = ModelSpec(config)
        return ModelBuilder.build(spec, container=container)
    except Exception as e:
        raise RuntimeError(f"Error building model from spec: {e}") from e


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

    def __init__(self, args: Mapping[str, Any]):
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
            ```python
            spec = TransformerSpec({
                'embedding_size': 1024,
                'block': {
                    'hidden_size': 512
                }
            })
            print(spec['block.hidden_size'])
            # Output: 512
            ```
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
            ```python
            spec = ModelSpec({
                'embedding_size': 1024,
                'block': {
                    'hidden_size': 512
                }
            })
            spec['block.feedforward.d_proj_up'] = 10
            ```

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


def _is_nested(value: Any) -> bool:
    return isinstance(value, (dict, list))


class ModelBuilder:
    """A builder for constructing a model from a specification."""

    @staticmethod
    def build(spec: ModelSpec, container=None) -> nn.Module:
        """Create a model according to the provided specification.

        Models are built by specifying a config. Configs are key->value mappings where keys
        correspond to argument names from a model's initializer and the values are the
        arguments. A config may not contain a key->value mapping for all possible args. If the
        missing arg is optional, then the default value specified in the model's iniatilizer
        will be used. If the arg is required, then a resolution process will take place. During
        resolution, the missing argument will be taken from the closest ancestor which specifies
        that argument. If no ancestors specify that argument, then the argument will be passed from
        the dependency container. If no such dependency exists then an error will be raised.

        A funky note on arg values. In cases where an arg value is itself a mapping or a container
        of mappings describing a component, then the arg should be treated as a factory. The model
        should be able to reference the factory multiple times to generate an unlimited number of
        instances of the defined component.

        Args:
            spec: The specification of the desired transformer.

        Returns:
            A transformer built according to the provided specification.

        Raises:
            ValueError: If an unrecognized module type is specified or if a required
                parameter for a module is not found in the specification.

        """

        def _build_module_factory(module_type, module_args, container):
            """Perform a single-pass depth-first traversal of the spec tree.

            Replaces child specs with factory functions that produce modules
            according to that spec. Parameters are inherited from ancestors by
            passing accumulated context downward.
            """
            if (variant := module_args.get("type")) is None:
                raise ValueError(f"No 'type' defined for component {module_type}")

            if (module_cls := get_component(module_type, variant)) is None:
                # Exit early if the component could not be found.
                raise ValueError(
                    f"Variant '{variant}' for module type '{module_type}' could not be"
                    + " found in the registry."
                )

            resolved_args = {}
            nested_args = []
            for arg_name, arg_value in module_args.items():
                if _is_nested(arg_value):
                    nested_args.append((arg_name, arg_value))
                else:
                    resolved_args[arg_name] = arg_value

            for arg_name, arg_value in nested_args:
                resolved_args[arg_name] = _build_module_factory(
                    arg_name,
                    {**copy.deepcopy(resolved_args), **arg_value},
                    container=container,
                )

            factory_args = {}
            for param_name, param in inspect.signature(
                module_cls.__init__
            ).parameters.items():
                if param_name == "self":
                    continue

                arg_value = resolved_args.get(param_name)

                if arg_value is None and param.default is inspect.Parameter.empty:
                    if container is not None:
                        arg_value = container.get(param_name)

                    if arg_value is None:
                        # Required parameter not found anywhere
                        raise ValueError(
                            f"Required parameter '{param_name}' not found for {module_type}.{variant}"
                        )

                factory_args[param_name] = arg_value

            return lambda: module_cls(**factory_args)

        return _build_module_factory(
            "model", copy.deepcopy(spec.args), container=container
        )()
