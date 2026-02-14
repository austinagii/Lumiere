"""Classes for dynamically building a Transformer model from a specification."""

import copy
import inspect
from collections.abc import Mapping
from typing import Any

from torch import nn

from lumiere.internal.registry import get_component


def _is_nested(value: Any) -> bool:
    return isinstance(value, (dict, list))


class ModelBuilder:
    """A builder for constructing a model from a specification."""

    @staticmethod
    def build(spec: Mapping[str, Any], container=None) -> nn.Module:
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
            container: Optional DependencyContainer for resolving dependencies.
                If None, uses the global dependency container.

        Returns:
            A transformer built according to the provided specification.

        Raises:
            ValueError: If an unrecognized module type is specified or if a required
                parameter for a module is not found in the specification.

        """
        # Use global container if none provided
        if container is None:
            from lumiere.internal.di import get_global_container
            container = get_global_container()

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
            "model", copy.deepcopy(spec), container=container
        )()
