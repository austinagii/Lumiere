"""Centralized loading for all component types."""

from collections.abc import Mapping
from typing import Any, TypeVar

import torch

from lumiere.data import DataLoader, Dataset, Pipeline, Preprocessor
from lumiere.internal.di import DependencyContainer, resolve_value
from lumiere.internal.registry import get
from lumiere.nn import ModelBuilder
from lumiere.tokenizers import Tokenizer


T = TypeVar("T")


def load(
    type_cls: type[T],
    config: dict[str, Any],
    container: Any = None,
    **required_params,
) -> T:
    """Universal load function for any registered component type.

    Args:
        type_cls: The type/protocol class (e.g., Tokenizer, Dataset)
        config: Configuration dict with 'name' field and parameters
        container: Optional DependencyContainer for resolving @variable syntax
        **required_params: Required parameters (e.g., params for optimizer,
            optimizer for scheduler)

    Returns:
        Initialized instance of the requested type

    Example:
        ```python
        tokenizer = load(Tokenizer, {"name": "bpe", "vocab_size": 4096})
        optimizer = load(
            torch.optim.Optimizer,
            {"name": "adamw", "lr": 0.001},
            params=model.parameters()
        )
        ```
    """
    # Get the implementation name
    name = config.get("name") or config.get("type")
    if not name:
        raise ValueError("Config must contain 'name' or 'type' field")

    # Look up the class in the registry
    impl_cls = get(type_cls, name)
    if not impl_cls:
        raise ValueError(f"{type_cls.__name__} '{name}' not found in registry")

    # Extract parameters (everything except name/type)
    params = {k: v for k, v in config.items() if k not in ("name", "type")}

    # Resolve dependencies from container
    if container:
        params = {k: resolve_value(v, container) for k, v in params.items()}

    # Merge with required params
    params.update(required_params)

    # Instantiate
    try:
        return impl_cls(**params)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize {name}: {e}") from e


def load_tokenizer(config: dict[str, Any], container: Any = None) -> Tokenizer:
    """Load a tokenizer from configuration.

    Args:
        config: Configuration with 'name' and tokenizer parameters
        container: Optional DependencyContainer for resolving dependencies

    Returns:
        Initialized Tokenizer instance

    Example:
        ```python
        tokenizer = load_tokenizer({"name": "bpe", "vocab_size": 30000})
        ```
    """
    return load(Tokenizer, config, container)


def load_optimizer(
    config: dict[str, Any],
    params,
    container: Any = None,
) -> torch.optim.Optimizer:
    """Load an optimizer from configuration.

    Args:
        config: Configuration with 'name' and optimizer parameters
        params: Model parameters to optimize (typically model.parameters())
        container: Optional DependencyContainer for resolving dependencies

    Returns:
        Initialized Optimizer instance

    Example:
        ```python
        optimizer = load_optimizer(
            {"name": "adamw", "lr": 0.001},
            model.parameters()
        )
        ```
    """
    return load(torch.optim.Optimizer, config, container, params=params)


def load_scheduler(
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    container: Any = None,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Load a learning rate scheduler from configuration.

    Args:
        config: Configuration with 'name' and scheduler parameters
        optimizer: The optimizer instance to schedule
        container: Optional DependencyContainer for resolving dependencies

    Returns:
        Initialized LRScheduler instance

    Example:
        ```python
        scheduler = load_scheduler(
            {"name": "cosine-annealing", "warmup_steps": 500},
            optimizer
        )
        ```
    """
    return load(
        torch.optim.lr_scheduler.LRScheduler, config, container, optimizer=optimizer
    )


def load_dataset(config: dict[str, Any], container: Any = None) -> DataLoader:
    """Load a dataset (or multiple datasets) from configuration.

    Args:
        config: Configuration with 'datasets' list and DataLoader parameters
        container: Optional DependencyContainer for resolving dependencies

    Returns:
        Initialized DataLoader instance wrapping one or more datasets

    Example:
        ```python
        dataloader = load_dataset({
            "datasets": [{"name": "wikitext"}],
            "merge_mode": "round_robin"
        })
        ```
    """
    if (dataset_configs := config.get("datasets")) is None:
        raise ValueError("Configuration must contain a 'datasets' field")

    # Load each dataset
    datasets = [load(Dataset, dc, container) for dc in dataset_configs]

    # Extract DataLoader parameters (everything except 'datasets')
    dataloader_params = {
        key: resolve_value(value, container)
        for key, value in config.items()
        if key != "datasets"
    }

    return DataLoader(datasets, **dataloader_params)


def load_pipeline(config: dict[str, Any], container: Any = None) -> Pipeline:
    """Load a pipeline from configuration.

    Args:
        config: Configuration with 'name', pipeline parameters, and optional
            'preprocessors' list
        container: Optional DependencyContainer for resolving dependencies

    Returns:
        Initialized Pipeline instance

    Example:
        ```python
        pipeline = load_pipeline({
            "name": "text",
            "tokenizer": tokenizer,
            "batch_size": 32,
            "preprocessors": [{"name": "autoregressive"}]
        })
        ```
    """
    # Handle nested preprocessors
    config_dict = dict(config)
    if "preprocessors" in config_dict and config_dict["preprocessors"]:
        preprocessors = [
            load(Preprocessor, pc, container) for pc in config_dict["preprocessors"]
        ]
        config_dict["preprocessors"] = preprocessors

    return load(Pipeline, config_dict, container)


def load_model(
    config: Mapping[str, Any], container: DependencyContainer | None = None
) -> torch.nn.Module:
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
        return ModelBuilder.build(config, container=container)
    except Exception as e:
        raise RuntimeError(f"Error building model from spec: {e}") from e
