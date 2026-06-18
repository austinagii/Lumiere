import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


SUBKEY_SEPARATOR = "."


class Config:
    """Configuration object with dot notation access.

    Provides hierarchical configuration access using dot notation (e.g., `config["model.vocab_size"]`).

    Args:
        data: Dictionary containing the configuration data.

    Example:
        ```python
        config = Config({"model": {"vocab_size": 30000}})
        vocab_size = config["model.vocab_size"]
        print(vocab_size)
        # Output: 30000
        ```
    """

    def __init__(self, data: dict[str, Any]):
        """Initialize a Config instance with the given data.

        Args:
            data: Dictionary containing the configuration data.
        """
        self.data = data

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

    # TODO: Improve performance using synchronized dfs.
    def update(self, o: "Config") -> "Config":
        config = copy.deepcopy(self)

        for param, value in o.flatten():
            config[param] = value

        return config

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

        if key.endswith("type"):
            for k in list(obj.keys()):
                if k != "type":
                    del obj[k]

    def __delitem__(self, key: str):
        """Delete a configuration value using dot notation.

        Args:
            key: The configuration key using dot notation (e.g., `"model.vocab_size"`).

        Raises:
            TypeError: If key is not a non-empty string.
            KeyError: If the key is not found in the configuration.

        Example:
            ```python
            config = Config({"model": {"vocab_size": 30000}})
            del config["model.vocab_size"]
            ```
        """
        if not isinstance(key, str) or len(key) == 0:
            raise TypeError("Key must be a non-empty string.")

        key_components = key.split(SUBKEY_SEPARATOR)

        obj = self.data
        for component in key_components[:-1]:
            obj = obj.get(component)

            if obj is None:
                raise KeyError(f"Field '{key}' not found in config")

        final_key = key_components[-1]
        if final_key not in obj:
            raise KeyError(f"Field '{key}' not found in config")

        del obj[final_key]

    def __str__(self):
        """Return a YAML string representation of the configuration.

        Returns:
            YAML-formatted string of the configuration data.
        """
        return yaml.dump(self.data, default_flow_style=False)

    def __repr__(self):
        return json.dumps(self.data, indent=2)

    def __iter__(self):
        """Iterate over configuration items.

        Yields:
            Tuples of `(key, value)` pairs from the configuration.
        """
        yield from self.data.items()

    def flatten(self):
        def _iter(prefix, v):
            if isinstance(v, Mapping):
                for nk, nv in v.items():
                    yield from _iter(prefix + [nk], nv)
            else:
                yield ".".join(prefix), v

        yield from _iter([], self.data)

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """Create a `Config` instance from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Initialized `Config` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if isinstance(path, str):
            try:
                path = Path(path)
            except Exception as e:
                raise ValueError(f"'{path}' is not a valid path") from e

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            config = yaml.safe_load(f)

        return cls(config)

    @classmethod
    def from_yaml(cls, s: str):
        """Create a `Config` instance from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Initialized `Config` instance.

        Raises:
            ValueError: If the path is invalid.
            FileNotFoundError: If the file does not exist.
        """
        return cls(yaml.safe_load(s))
