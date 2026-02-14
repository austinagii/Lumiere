"""Training script for Lumière models."""

import argparse
import logging
import sys

import torch

from lumiere import Loader, register_dependency
from lumiere.training import (
    Checkpoint,
    CheckpointType,
    RunManager,
    Trainer,
)
from lumiere.training.config import Config
from lumiere.training.loss import cross_entropy_loss
from lumiere.utils import get_device, register_signal_handlers


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def init_training_run(config_path: str, device: torch.device):
    """Initialize a new training run and all training components from configuration.

    Args:
        config_path: Path to training configuration YAML file.
        device: Device to use for training.

    Returns:
        Tuple of (run_id, run_manager, model, dataloader, pipeline, optimizer, scheduler, tokenizer)
    """
    config = Config.from_yaml(config_path)
    # fmt: off
    assert config.get("tokenizer") is not None, "Config missing required 'tokenizer' section"  # NOQA: E501
    assert config.get("data") is not None, "Config missing required 'data' section"
    assert config.get("pipeline") is not None, "Config missing required 'pipeline' section"  # NOQA: E501
    assert config.get("model") is not None, "Config missing required 'model' section"
    assert config.get("optimizer") is not None, "Config missing required 'optimizer' section"  # NOQA: E501
    assert config.get("training") is not None, "Config missing required 'training' section"  # NOQA: E501
    # fmt: on

    run_manager = RunManager.from_config(config)
    run_id = run_manager.init_run(config)
    logger.info(f"Initialized new training run with ID: {run_id}")

    logger.info("Loading tokenizer...")
    tokenizer = Loader.tokenizer(config["tokenizer"])
    register_dependency("tokenizer", tokenizer)
    logger.info("Tokenizer loaded successfully")

    logger.info("Loading dataset...")
    dataloader = Loader.data(config["data"])
    logger.info(f"Dataloader initialized with {len(dataloader.datasets)} dataset(s)")

    logger.info("Training tokenizer on training data...")
    tokenizer.train(dataloader["train"])
    logger.info(f"Tokenizer trained successfully (vocab size: {tokenizer.vocab_size})")

    if run_manager is not None:
        logger.info("Saving tokenizer artifact...")
        run_manager.save_artifact("tokenizer", tokenizer)
        logger.info("Tokenizer artifact saved successfully")

    logger.info("Loading pipeline...")
    pipeline = Loader.pipeline(config["pipeline"])
    logger.info("Pipeline loaded successfully")

    logger.info("Building model...")
    model = Loader.model(config["model"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model built successfully - "
        f"Total params: {total_params:,}, Trainable: {trainable_params:,}"
    )

    logger.info("Loading optimizer...")
    optimizer = Loader.optimizer(config["optimizer"], model.parameters())
    logger.info(f"Optimizer loaded: {type(optimizer).__name__}")

    scheduler = None
    if (scheduler_config := config.get("scheduler")) is not None:
        logger.info("Loading learning rate scheduler...")
        scheduler = Loader.scheduler(scheduler_config, optimizer)
        logger.info(f"Scheduler loaded: {type(scheduler).__name__}")

    logger.info("All components initialized successfully")

    return (
        run_id,
        run_manager,
        model,
        dataloader,
        pipeline,
        optimizer,
        scheduler,
        tokenizer,
    )


def resume_training_run(run_id: str, checkpoint_tag: str, device: torch.device):
    """Resume training from a checkpoint.

    Args:
        run_id: ID of the run to resume.
        checkpoint_tag: Tag of the checkpoint to load (e.g., 'epoch:0001', 'best', 'final').
        config: Training configuration dictionary.
        device: Device to use for training.

    Returns:
        Tuple of (model, dataloader, pipeline, optimizer, scheduler, tokenizer, run_manager, checkpoint)
    """
    # Initialize RunManager
    config_obj = Config(config)
    run_manager = RunManager.from_config(config_obj)

    # Load checkpoint
    logger.info(f"Loading checkpoint '{checkpoint_tag}' from run '{run_id}'...")
    checkpoint = run_manager.load_checkpoint(run_id, checkpoint_tag)
    assert checkpoint is not None, "Failed to load checkpoint"
    logger.info("Checkpoint loaded successfully")

    # Update config from checkpoint if available
    if "config" in checkpoint:
        config = checkpoint["config"]

    # Load tokenizer from artifact
    logger.info("Loading tokenizer artifact...")
    tokenizer_bytes = run_manager.load_artifact(run_id, "tokenizer")
    assert tokenizer_bytes is not None, "Could not find tokenizer artifact"

    tokenizer_config = config.get("tokenizer") or {}
    # Assuming tokenizer has a from_bytes method
    from lumiere.tokenizers import Tokenizer

    tokenizer = Tokenizer.from_bytes(tokenizer_bytes)
    assert tokenizer is not None, "Failed to deserialize tokenizer"
    assert tokenizer.vocab_size > 0, "Tokenizer vocabulary is empty"
    logger.info("Tokenizer loaded successfully")

    # Register tokenizer in global container
    register_dependency("tokenizer", tokenizer)

    # Load dataloader
    logger.info("Loading dataset...")
    data_config = config.get("data") or {}
    dataloader = Loader.data(data_config)
    logger.info(f"Dataloader initialized with {len(dataloader.datasets)} dataset(s)")

    # Load pipeline
    logger.info("Loading pipeline...")
    pipeline_config = config.get("pipeline") or {}
    pipeline = Loader.pipeline(pipeline_config)
    logger.info("Pipeline loaded successfully")

    # Build model and load state
    logger.info("Building model...")
    model_config = checkpoint.get("model_config") or config.get("model", {})
    model = Loader.model(model_config).to(device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Model state loaded from checkpoint")

    # Load optimizer and state
    logger.info("Loading optimizer...")
    optimizer_config = config.get("optimizer") or {}
    optimizer = Loader.optimizer(optimizer_config, model.parameters())

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Optimizer state loaded from checkpoint")

    # Load scheduler and state
    scheduler = None
    scheduler_config = config.get("scheduler")
    if scheduler_config is not None:
        logger.info("Loading scheduler...")
        scheduler = Loader.scheduler(scheduler_config, optimizer)

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("Scheduler state loaded from checkpoint")

    return (
        model,
        dataloader,
        pipeline,
        optimizer,
        scheduler,
        tokenizer,
        run_manager,
        checkpoint,
    )


def create_checkpoint_saver(run_manager: RunManager, run_id: str, model, config: dict):
    """Create a checkpoint saving hook for the trainer.

    Args:
        run_manager: RunManager instance for saving checkpoints.
        run_id: ID of the current run.
        model: The model being trained.
        config: Training configuration dictionary.

    Returns:
        Hook function to save checkpoints.
    """

    def save_checkpoint_hook(trainer):
        """Save checkpoint at the end of each epoch."""
        checkpoint = Checkpoint(
            run_id=run_id,
            config=config,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=trainer.optimizer.state_dict(),
            epoch=trainer.state.current_epoch,
            global_step=trainer.state.global_step,
            best_loss=trainer.state.best_loss
            if hasattr(trainer.state, "best_loss")
            else None,
        )

        if trainer.scheduler is not None:
            checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()

        # Save epoch checkpoint
        logger.info(f"Saving checkpoint for epoch {trainer.state.current_epoch}...")
        run_manager.save_checkpoint(CheckpointType.EPOCH, checkpoint)
        logger.info("Checkpoint saved successfully")

        # Save best checkpoint if this is the best epoch
        if hasattr(trainer.state, "best_loss") and hasattr(trainer.state, "prev_loss"):
            if trainer.state.prev_loss == trainer.state.best_loss:
                logger.info("Saving 'best' checkpoint...")
                run_manager.save_checkpoint(CheckpointType.BEST, checkpoint)
                logger.info("Best checkpoint saved successfully")

    return save_checkpoint_hook


def train(
    config_path: str,
    run_id: str = None,
    checkpoint_tag: str = None,
):
    """Run training from a configuration file with checkpointing support.

    Args:
        config_path: Path to the training configuration YAML file.
        run_id: Optional run ID to resume from.
        checkpoint_tag: Optional checkpoint tag to resume from (requires run_id).
    """
    # Load configuration
    config = Config.from_yaml(config_path)

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Initialize or resume run
    if run_id is None:
        # Start new run
        logger.info("Starting new training run...")

        # Initialize components
        (
            run_id,
            run_manager,
            model,
            dataloader,
            pipeline,
            optimizer,
            scheduler,
            tokenizer,
        ) = init_training_run(config_path, device)

        starting_epoch = 0
    else:
        # Resume from checkpoint
        logger.info(
            f"Resuming training run '{run_id}' from checkpoint '{checkpoint_tag}'..."
        )

        if checkpoint_tag is None:
            checkpoint_tag = "latest"  # or "best", depending on your preference

        (
            model,
            dataloader,
            pipeline,
            optimizer,
            scheduler,
            tokenizer,
            run_manager,
            checkpoint,
        ) = resume_training_run(run_id, checkpoint_tag, config, device)

        starting_epoch = checkpoint.get("epoch", 0)
        logger.info(f"Resuming from epoch {starting_epoch}")

    # Get training parameters
    training_config = config.get("training") or {}

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

    # Register checkpoint saving hook
    checkpoint_saver = create_checkpoint_saver(run_manager, run_id, model, config)
    trainer.register_post_epoch_hook(checkpoint_saver)

    logger.info("Trainer initialized successfully")

    # Start training
    logger.info("=" * 60)
    logger.info(f"Starting training run: {run_id}")
    logger.info("=" * 60)

    try:
        metrics = trainer.train()

        # Save final checkpoint
        logger.info("Saving final checkpoint...")
        final_checkpoint = Checkpoint(
            run_id=run_id,
            config=config,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=trainer.optimizer.state_dict(),
            epoch=trainer.state.current_epoch,
            global_step=trainer.state.global_step,
            best_loss=trainer.state.best_loss
            if hasattr(trainer.state, "best_loss")
            else None,
        )

        if trainer.scheduler is not None:
            final_checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()

        run_manager.save_checkpoint(CheckpointType.FINAL, final_checkpoint)
        logger.info("Final checkpoint saved successfully")

        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        # Save interrupt checkpoint
        logger.info("Saving interrupt checkpoint...")
        interrupt_checkpoint = Checkpoint(
            run_id=run_id,
            config=config,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=trainer.optimizer.state_dict(),
            epoch=trainer.state.current_epoch,
            global_step=trainer.state.global_step,
        )
        if trainer.scheduler is not None:
            interrupt_checkpoint["scheduler_state_dict"] = (
                trainer.scheduler.state_dict()
            )

        run_manager.save_checkpoint(CheckpointType.EPOCH, interrupt_checkpoint)
        logger.info("Interrupt checkpoint saved")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


def main():
    """Main entry point for the training script."""
    register_signal_handlers()

    parser = argparse.ArgumentParser(
        description="Train a Lumière transformer model with checkpointing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the training configuration YAML file",
    )

    parser.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        help="The run ID to resume (required when loading from checkpoint)",
    )

    parser.add_argument(
        "--checkpoint-tag",
        dest="checkpoint_tag",
        default=None,
        help="The checkpoint tag to resume from (e.g., 'epoch:0001', 'best', 'final')",
    )

    args = parser.parse_args()

    try:
        train(
            config_path=args.config_path,
            run_id=args.run_id,
            checkpoint_tag=args.checkpoint_tag,
        )
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
