"""Pipeline loading functionality."""

from typing import Any

from lumiere.data.pipeline import Pipeline
from lumiere.data.preprocessor import Preprocessor
from lumiere.discover import get_registry
from lumiere.loading.loader import ConfigLoader

# Create loaders
_pipeline_loader = ConfigLoader[Pipeline](get_registry(Pipeline))
_preprocessor_loader = ConfigLoader[Preprocessor](get_registry(Preprocessor))


def load(config: dict[str, Any], container: Any = None) -> Pipeline:
    """Load and return a Pipeline instance from a configuration dictionary.

    The configuration must contain a 'name' field with the registered pipeline
    identifier, plus any additional keyword arguments required for that pipeline's
    initialization. For nested components like preprocessors, the configuration
    should include their initialization details.

    Dependencies can be injected via a DependencyContainer. Values in the config
    that start with "@" (e.g., "@tokenizer") will be resolved from the container.

    Args:
        config: Configuration dictionary containing:
            - 'name': Registered identifier of the pipeline.
            - Additional key-value pairs for pipeline-specific parameters.
            - 'preprocessors': Optional list of preprocessor configurations.
        container: Optional DependencyContainer for resolving dependencies.

    Returns:
        Initialized Pipeline instance.

    Example:
        >>> config = {
        ...     "name": "text",
        ...     "tokenizer": tokenizer_instance,
        ...     "batch_size": 32,
        ...     "preprocessors": [
        ...         {"name": "autoregressive", "device": "cuda"}
        ...     ]
        ... }
        >>> pipeline = load(config)
    """
    # Convert config to dict and handle preprocessors specially
    config_dict = dict(config)

    # If preprocessors are specified, load them using the centralized loader
    if "preprocessors" in config_dict and config_dict["preprocessors"] is not None:
        preprocessors = [
            _preprocessor_loader.load(dict(pc), container=container)
            for pc in config_dict["preprocessors"]
        ]
        config_dict["preprocessors"] = preprocessors

    return _pipeline_loader.load(config_dict, container=container)
