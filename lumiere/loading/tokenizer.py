"""Tokenizer loading functionality."""

from typing import Any

from lumiere.discover import get_registry
from lumiere.loading.loader import ConfigLoader
from lumiere.tokenizer import Tokenizer

# Create loader
_loader = ConfigLoader[Tokenizer](get_registry(Tokenizer))


def load(config: dict[str, Any], container: Any = None) -> Tokenizer:
    """Load and return a Tokenizer instance from a configuration dictionary.

    The configuration must contain a 'name' or 'type' field with the registered
    tokenizer identifier, plus any additional keyword arguments required for
    that tokenizer's initialization.

    Dependencies can be injected via a DependencyContainer. Values in the config
    that start with "@" (e.g., "@vocab_size") will be resolved from the container.

    Args:
        config: Configuration dictionary containing:
            - 'name' or 'type': Registered identifier of the tokenizer.
            - Additional key-value pairs for tokenizer-specific parameters.
            - Values starting with "@" will be resolved from the container.
        container: Optional DependencyContainer for resolving dependencies.

    Returns:
        Initialized Tokenizer instance.

    Example:
        >>> config = {"name": "bpe", "vocab_size": 30000}
        >>> tokenizer = load(config)
    """
    # Support both 'name' and 'type' for flexibility
    config_dict = dict(config)
    if "name" not in config_dict and "type" in config_dict:
        config_dict["name"] = config_dict.pop("type")

    return _loader.load(config_dict, container=container)
