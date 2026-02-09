"""Training script for Lumière models."""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

from lumiere import DependencyContainer
from lumiere.loading import (
    load_dataset as load_dataloader,
    load_optimizer,
    load_pipeline,
    load_scheduler,
    load_tokenizer,
)
from lumiere.models import load as load_model
from lumiere.training import Trainer
from lumiere.training.loss import cross_entropy_loss
from lumiere.utils.device import get_device


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the training configuration.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def init_training_components(config: dict, device: torch.device):
    """Initialize all training components from configuration.

    Args:
        config: Training configuration dictionary.
        device: Device to use for training.

    Returns:
        Tuple of (model, dataloader, pipeline, optimizer, scheduler, tokenizer)
    """
    # Initialize dependency container
    container = DependencyContainer()

    # Load and train tokenizer
    logger.info("Loading tokenizer...")
    tokenizer_config = config.get("tokenizer", {})
    tokenizer = load_tokenizer(tokenizer_config)
    logger.info("Tokenizer loaded successfully")

    # Register tokenizer in container for pipeline dependency injection
    container.register("tokenizer", tokenizer)

    # Load dataloader
    logger.info("Loading dataset...")
    data_config = config.get("data", {})
    dataloader = load_dataloader(data_config, container)
    logger.info(f"Dataloader initialized with {len(dataloader.datasets)} dataset(s)")

    # Train tokenizer on training data
    logger.info("Training tokenizer on training data...")
    tokenizer.train(dataloader["train"])
    logger.info(f"Tokenizer trained successfully (vocab size: {tokenizer.vocab_size})")

    # Load pipeline with injected tokenizer
    logger.info("Loading pipeline...")
    pipeline_config = config.get("pipeline", {})
    pipeline = load_pipeline(pipeline_config, container)
    logger.info("Pipeline loaded successfully")

    # Build model
    logger.info("Building model...")
    model_config = config.get("model", {})
    model = load_model(model_config, container).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model built successfully - "
        f"Total params: {total_params:,}, Trainable: {trainable_params:,}"
    )

    # Load optimizer
    logger.info("Loading optimizer...")
    optimizer_config = config.get("optimizer", {})
    optimizer = load_optimizer(optimizer_config, model.parameters(), container)
    logger.info(f"Optimizer loaded: {type(optimizer).__name__}")

    # Load scheduler
    scheduler = None
    if "scheduler" in config:
        logger.info("Loading learning rate scheduler...")
        scheduler_config = config["scheduler"]
        scheduler = load_scheduler(scheduler_config, optimizer, container)
        logger.info(f"Scheduler loaded: {type(scheduler).__name__}")

    return model, dataloader, pipeline, optimizer, scheduler, tokenizer


def train(config_path: str):
    """Run training from a configuration file.

    Args:
        config_path: Path to the training configuration YAML file.
    """
    # Load configuration
    config = load_config(config_path)

    # Get device
    device = get_device()

    # Initialize all components
    model, dataloader, pipeline, optimizer, scheduler, tokenizer = (
        init_training_components(config, device)
    )

    # Get training parameters
    training_config = config.get("training", {})

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        pipeline=pipeline,
        loss_fn=cross_entropy_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        **training_config,
    )
    logger.info("Trainer initialized successfully")

    # Start training
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(
        description="Train a Lumière transformer model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the training configuration YAML file",
    )

    args = parser.parse_args()

    try:
        train(args.config_path)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
