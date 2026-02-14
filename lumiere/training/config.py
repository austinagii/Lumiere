from pathlib import Path
from typing import Any

import yaml


SUBKEY_SEPARATOR = "."


class Config:
    """Configuration object with dot notation access and singleton pattern.

    Provides hierarchical configuration access using dot notation (e.g., `config["model.vocab_size"]`).
    Implements the singleton pattern to ensure only one configuration instance exists.

    Args:
        data: Dictionary containing the configuration data.
        override: Whether to override an existing singleton instance. Defaults to `False`.

    Example:
        ```python
        config = Config({"model": {"vocab_size": 30000}})
        vocab_size = config["model.vocab_size"]
        print(vocab_size)
        # Output: 30000
        ```
    """

    _instance = None

    @classmethod
    def is_initialized(cls) -> bool:
        """Check whether the configuration singleton has been initialized.

        Returns:
            `True` if the singleton is initialized, `False` otherwise.
        """
        return (
            cls._instance is not None
            and hasattr(cls._instance, "_initialized")
            and cls._instance._initialized
        )

    @classmethod
    def get_instance(cls):
        """Return the configuration singleton instance.

        Returns:
            The `Config` singleton instance, or `None` if not initialized.
        """
        return cls._instance

    @classmethod
    def from_yaml(cls, path: str | Path, override=False):
        """Create a `Config` instance from a YAML file.

        Args:
            path: Path to the YAML configuration file.
            override: Whether to override an existing singleton instance. Defaults to `False`.

        Returns:
            Initialized `Config` instance.

        Raises:
            ValueError: If the path is invalid.
            FileNotFoundError: If the file does not exist.
        """
        if isinstance(path, str):
            try:
                path = Path(path)
            except Exception:
                raise ValueError(f"'{path}' is not a valid path")

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            config = yaml.safe_load(f)

        return cls(config, override)

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, data: dict[str, Any], override=False):
        if not hasattr(self, "_initialized") or override:  # Prevent re-initialization
            self.data = data
            self._initialized = True

    def get(self, key: str) -> Any:
        """Get a configuration value using dot notation, returning `None` if not found.

        Args:
            key: The configuration key using dot notation (e.g., `"model.vocab_size"`).

        Returns:
            The configuration value, or `None` if the key is not found.
        """
        value = None
        try:
            value = self.__getitem__(key)
        except KeyError:
            pass

        return value

    def __getitem__(self, key: str):
        """Get a configuration value using dot notation.

        Args:
            key: The configuration key using dot notation (e.g., `"model.vocab_size"`).

        Returns:
            The configuration value.

        Raises:
            TypeError: If the key is not a non-empty string.
            KeyError: If the key is not found in the configuration.
        """
        if not isinstance(key, str) or len(key) == 0:
            raise TypeError("Key must be a non-empty string.")

        key_components = key.split(SUBKEY_SEPARATOR)

        obj = self.data
        for component in key_components:
            obj = obj.get(component)

            if obj is None:
                raise KeyError(f"Field '{key}' not found in config")

        return obj

    def __setitem__(self, key: str, value: Any):
        """Set a configuration value using dot notation.

        Args:
            key: The configuration key using dot notation (e.g., `"model.vocab_size"`).
            value: The value to set.

        Raises:
            TypeError: If the key is not a non-empty string.
        """
        if not isinstance(key, str) or len(key) == 0:
            raise TypeError("Key must be a non-empty string.")

        key_components = key.split(SUBKEY_SEPARATOR)

        obj = self.data
        for component in key_components[:-1]:
            if (next_obj := obj.get(component)) is None:
                obj[component] = dict()
                next_obj = obj[component]

            obj = next_obj

        obj[key_components[len(key_components) - 1]] = value

    def __str__(self):
        """Return a YAML string representation of the configuration.

        Returns:
            YAML-formatted string of the configuration data.
        """
        return yaml.dump(self.data, default_flow_style=False)

    def __iter__(self):
        """Iterate over configuration items.

        Yields:
            Tuples of `(key, value)` pairs from the configuration.
        """
        yield from self.data.items()

    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Create a `Config` instance from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Initialized `Config` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        return cls(config)


# class ModelSpec:
#     """A specification for dynamically constructing a model.
#
#     This class represents a hierarchical configuration that maps directly to the
#     initialization parameters of a model and its nested components. The specification
#     structure should mirror the model's component hierarchy, with each nested dictionary
#     defining both the desired sub-module type and its initialization parameters.
#
#     For example, if a transformer model requires 'context_size', 'embedding_size', and
#     'block' (where 'block' is a callable producing transformer block instances),
#     and a 'standard' block type requires 'hidden_size' and 'dropout', the
#     specification would then include both sets of parameters in a nested structure:
#
#         context_size: 10
#         embedding_size: 16
#         block:
#             type: standard
#             hidden_size: 24
#             dropout: 0.1
#
#     The 'type' field specifies which module to instantiate, while other nested
#     parameters are passed to that module's initializer. Top-level parameters are
#     passed to the transformer's initializer.
#
#     In cases where a sub-module's initializer shares parameters in common with
#     an ancestor, then those parameters can be omitted from the sub-module's spec
#     and the corresponding values will be inherited from the nearest ancestor that
#     defines them.
#
#         context_size: 10
#         embedding_size: 16
#         block:
#             type: swiglu
#             # embedding_size: 16  <-- This value will be inherited if omitted.
#             hidden_size: 24
#             dropout: 0.1
#
#     """
#
#     def __init__(self, args: Mapping[str, Any]):
#         """Initialize a model specification.
#
#         Args:
#             args: The arguments for the model and its components.
#
#         """
#         if args is None:
#             raise ValueError("A dict of spec arguments is required.")
#
#         self.args = args
#
#     @classmethod
#     def from_yaml(cls, path: str | Path):
#         """Create a model spec instance from a YAML file.
#
#         Args:
#             path: The path to the desired YAML file.
#
#         Returns:
#             A spec matching the contents of the YAML file.
#
#         Raises:
#             ValueError: If the path is not a valid path.
#             FileNotFoundError: If the YAML file does not exist.
#
#         """
#         if isinstance(path, str):
#             try:
#                 path = Path(path)
#             except Exception as e:
#                 raise ValueError(f"'{path}' is not a valid path") from e
#
#         if not isinstance(path, Path):
#             raise ValueError("'path' must be a string or Path object.")
#
#         if not path.exists():
#             raise FileNotFoundError(f"Specfication file not found: {path}")
#
#         with open(path) as f:
#             spec = yaml.safe_load(f)
#
#         return cls(spec)
#
#     def __getitem__(self, argname: str):
#         """Return the value of the specified argument.
#
#         For ease of use, when retrieving the value of an argument for a sub-module, dot
#         notation can be used to specify the path to the desired argument.
#
#         Args:
#             argname: The name of the argument.
#
#         Returns:
#             The value of the argument.
#
#         Raises:
#             TypeError: If argname is not a non-empty string.
#             KeyError: If the argument is not found in the specification.
#
#         Example:
#             ```python
#             spec = TransformerSpec({
#                 'embedding_size': 1024,
#                 'block': {
#                     'hidden_size': 512
#                 }
#             })
#             print(spec['block.hidden_size'])
#             # Output: 512
#             ```
#         """
#         if not isinstance(argname, str) or len(argname) == 0:
#             raise TypeError("Argument name must be a non-empty string.")
#
#         key_components = argname.split(".")
#         parents = key_components[:-1]
#         target = key_components[-1]
#
#         def _get(obj, components, target):
#             if len(components) == 0:
#                 value = obj.get(target)
#             else:
#                 value = _get(obj[components[0]], components[1:], target)
#                 if value is None:
#                     value = obj.get(target)
#             return value
#
#         value = _get(self.args, parents, target)
#
#         if value is None:
#             raise KeyError(f"Argument '{argname}' not found in specification.")
#
#         return value
#
#     def get(self, argname: str) -> Any:
#         """Return the value of the specified argument in this specification.
#
#         This method is equivalent to reading the argument using square bracket notation
#         but returns `None` when the specified argument could not be found, instead of
#         raising a KeyError.
#
#         """
#         value = None
#         try:
#             value = self.__getitem__(argname)
#         except KeyError:
#             pass
#
#         return value
#
#     def __setitem__(self, argname: str, value: Any) -> None:
#         """Set the value of the specified argument.
#
#         For ease of use, when setting the value of an argument for a sub-module, dot
#         notation can be used to specify the path to the desired argument.
#
#         Args:
#             argname: The name of the argument to be modified.
#             value: The desired value of the argument.
#
#         Raises:
#             TypeError: If argname is not a non-empty string.
#
#         Example:
#             ```python
#             spec = ModelSpec({
#                 'embedding_size': 1024,
#                 'block': {
#                     'hidden_size': 512
#                 }
#             })
#             spec['block.feedforward.d_proj_up'] = 10
#             ```
#
#         """
#         if not isinstance(argname, str) or len(argname) == 0:
#             raise TypeError("Argument name must be a non-empty string.")
#
#         key_components = argname.split(".")
#
#         obj = self.args
#         for component in key_components[:-1]:
#             if (next_obj := obj.get(component)) is None:
#                 obj[component] = dict()
#                 next_obj = obj[component]
#
#             obj = next_obj
#
#         obj[key_components[-1]] = value
#
#     def __delitem__(self, key: str):
#         """Delete the specified argument.
#
#         This method removes the specified argument from this specification's argument
#         dict.
#
#         For ease of use, when deleting an argument for a sub-module, dot notation
#         can be used to specify the path to the desired argument.
#
#         Args:
#             key: The name of the argument to be deleted.
#
#         Raises:
#             TypeError: If key is not a non-empty string.
#             KeyError: If the argument is not found in the specification.
#
#         """
#         if not isinstance(key, str) or len(key) == 0:
#             raise TypeError("Key must be a non-empty string.")
#
#         key_components = key.split(".")
#
#         obj = self.args
#         for component in key_components[:-1]:
#             obj = obj.get(component)
#
#             if obj is None:
#                 raise KeyError(f"Argument '{key}' not found in specification")
#
#         del obj[key_components[len(key_components) - 1]]
#
