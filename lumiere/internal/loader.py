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


class Loader:
    """Centralized component loader with static methods for all component types.

    This class provides a unified interface for loading various components used
    in training and inference, including models, tokenizers, optimizers, schedulers,
    datasets, and pipelines.

    All methods are static and use the underlying registry and dependency injection
    system to instantiate components from configuration dictionaries. When no
    container is provided, methods automatically use the global dependency container.

    Example:
        ```python
        from lumiere import Loader, register_dependency

        # Load a tokenizer
        tokenizer = Loader.tokenizer({"name": "bpe", "vocab_size": 30000})

        # Register it in the global container for dependency injection
        register_dependency("tokenizer", tokenizer)

        # Load a model that references the tokenizer
        model = Loader.model({"embedding": {"tokenizer": "@tokenizer"}})

        # Load an optimizer
        optimizer = Loader.optimizer(
            {"name": "adamw", "lr": 0.001},
            model.parameters()
        )

        # For testing, you can still pass a custom container
        from lumiere import DependencyContainer
        test_container = DependencyContainer()
        test_container.register("tokenizer", mock_tokenizer)
        model = Loader.model(config, test_container)
        ```
    """

    @staticmethod
    def component(
        type_cls: type[T],
        config: dict[str, Any],
        container: Any = None,
        **required_params,
    ) -> T:
        """Universal load method for any registered component type.

        Args:
            type_cls: The type/protocol class (e.g., Tokenizer, Dataset)
            config: Configuration dict with 'name' field and parameters
            container: Optional DependencyContainer for resolving @variable syntax
            **required_params: Required parameters (e.g., params for optimizer)

        Returns:
            Initialized instance of the requested type

        Example:
            ```python
            from lumiere.tokenizers import Tokenizer
            tokenizer = Loader.component(
                Tokenizer,
                {"name": "bpe", "vocab_size": 4096}
            )
            ```
        """
        return load(type_cls, config, container, **required_params)

    @staticmethod
    def tokenizer(config: dict[str, Any], container: Any = None) -> Tokenizer:
        """Load a tokenizer from configuration.

        Args:
            config: Configuration with 'name' and tokenizer parameters
            container: Optional DependencyContainer for resolving dependencies.
                If None, uses the global dependency container.

        Returns:
            Initialized Tokenizer instance

        Example:
            ```python
            # Simple usage
            tokenizer = Loader.tokenizer({"name": "bpe", "vocab_size": 30000})

            # With dependency injection
            from lumiere import register_dependency
            register_dependency("vocab_size", 30000)
            tokenizer = Loader.tokenizer({"name": "bpe", "vocab_size": "@vocab_size"})
            ```
        """
        return load_tokenizer(config, container)

    @staticmethod
    def optimizer(
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
            optimizer = Loader.optimizer(
                {"name": "adamw", "lr": 0.001},
                model.parameters()
            )
            ```
        """
        return load_optimizer(config, params, container)

    @staticmethod
    def scheduler(
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
            scheduler = Loader.scheduler(
                {"name": "cosine-annealing", "warmup_steps": 500},
                optimizer
            )
            ```
        """
        return load_scheduler(config, optimizer, container)

    @staticmethod
    def data(config: dict[str, Any], container: Any = None) -> DataLoader:
        """Load a dataset (or multiple datasets) from configuration.

        Args:
            config: Configuration with 'datasets' list and DataLoader parameters
            container: Optional DependencyContainer for resolving dependencies

        Returns:
            Initialized DataLoader instance wrapping one or more datasets

        Example:
            ```python
            dataloader = Loader.data({
                "datasets": [{"name": "wikitext"}],
                "merge_mode": "round_robin"
            })
            ```
        """
        return load_data(config, container)

    @staticmethod
    def pipeline(config: dict[str, Any], container: Any = None) -> Pipeline:
        """Load a pipeline from configuration.

        Args:
            config: Configuration with 'name', pipeline parameters, and optional
                'preprocessors' list
            container: Optional DependencyContainer for resolving dependencies

        Returns:
            Initialized Pipeline instance

        Example:
            ```python
            pipeline = Loader.pipeline({
                "name": "text",
                "tokenizer": tokenizer,
                "batch_size": 32,
                "preprocessors": [{"name": "autoregressive"}]
            })
            ```
        """
        return load_pipeline(config, container)

    @staticmethod
    def model(
        config: Mapping[str, Any], container: DependencyContainer | None = None
    ) -> torch.nn.Module:
        """Load a model from hierarchical configuration.

        This method uses the ModelBuilder to construct models from hierarchical
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
                    "attention": {"name": "multihead", "num_heads": 8},
                    "feedforward": {"name": "linear", "d_ff": 2048},
                    "normalization": {"name": "rms"}
                }
            }
            model = Loader.model(config)
            ```
        """
        return load_model(config, container)


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
        container: Optional DependencyContainer for resolving @variable syntax.
            If None, uses the global dependency container.
        **required_params: Required parameters (e.g., params for optimizer,
            optimizer for scheduler)

    Returns:
        Initialized instance of the requested type

    Example:
        ```python
        # Using explicit container
        tokenizer = load(Tokenizer, {"name": "bpe", "vocab_size": 4096}, container)

        # Using global container
        from lumiere import register_dependency
        register_dependency("vocab_size", 4096)
        tokenizer = load(Tokenizer, {"name": "bpe", "vocab_size": "@vocab_size"})
        ```
    """
    # Use global container if none provided
    if container is None:
        from lumiere.internal.di import get_global_container
        container = get_global_container()

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
        container: Optional DependencyContainer for resolving dependencies.
            If None, uses the global dependency container.

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
        container: Optional DependencyContainer for resolving dependencies.
            If None, uses the global dependency container.

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
        container: Optional DependencyContainer for resolving dependencies.
            If None, uses the global dependency container.

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


def load_data(config: dict[str, Any], container: Any = None) -> DataLoader:
    """Load a dataset (or multiple datasets) from configuration.

    Args:
        config: Configuration with 'datasets' list and DataLoader parameters
        container: Optional DependencyContainer for resolving dependencies.
            If None, uses the global dependency container.

    Returns:
        Initialized DataLoader instance wrapping one or more datasets

    Example:
        ```python
        dataloader = load_data({
            "datasets": [{"name": "wikitext"}],
            "merge_mode": "round_robin"
        })
        ```
    """
    # Use global container if none provided
    if container is None:
        from lumiere.internal.di import get_global_container
        container = get_global_container()

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
        container: Optional DependencyContainer for resolving dependencies.
            If None, uses the global dependency container.

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
    # Use global container if none provided
    if container is None:
        from lumiere.internal.di import get_global_container
        container = get_global_container()

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
