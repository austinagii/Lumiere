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
    TrainingState,
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

RUN_CONFIG_PATH = "lumiere.yaml"
CHECKPOINT_INTERVAL = 5


def init_training_run(config: Config, device: torch.device):
    """Initialize components for a new training run.

    Args:
        config: The configuration for the training run.
        device: The device to use for training.

    Returns:
        Tuple of (model, dataloader, pipeline, optimizer, scheduler, tokenizer)
    """
    # fmt: off
    assert config.get("tokenizer") is not None, "Config missing required 'tokenizer' section"  # NOQA: E501
    assert config.get("data") is not None, "Config missing required 'data' section"
    assert config.get("pipeline") is not None, "Config missing required 'pipeline' section"  # NOQA: E501
    assert config.get("model") is not None, "Config missing required 'model' section"
    assert config.get("optimizer") is not None, "Config missing required 'optimizer' section"  # NOQA: E501
    assert config.get("training") is not None, "Config missing required 'training' section"  # NOQA: E501
    # fmt: on

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

    return model, dataloader, pipeline, optimizer, scheduler, tokenizer


def resume_training_run(config: Config, checkpoint: Checkpoint, device: torch.device):
    """Load components from a previous training run.

    Args:
        config: The configuration from the previous training run.
        checkpoint: The checkpoint to resume from.
        device: The device to use for training.

    Returns:
        Tuple of (model, dataloader, pipeline, optimizer, scheduler, tokenizer)
    """
    # fmt: off
    assert config.get("tokenizer") is not None, "Config missing required 'tokenizer' section"  # NOQA: E501
    assert config.get("data") is not None, "Config missing required 'data' section"
    assert config.get("pipeline") is not None, "Config missing required 'pipeline' section"  # NOQA: E501
    assert config.get("model") is not None, "Config missing required 'model' section"
    assert config.get("optimizer") is not None, "Config missing required 'optimizer' section"  # NOQA: E501
    assert config.get("training") is not None, "Config missing required 'training' section"  # NOQA: E501
    # fmt: on

    logger.info("Loading tokenizer...")
    tokenizer = Loader.tokenizer(config["tokenizer"], state=checkpoint["tokenizer"])
    register_dependency("tokenizer", tokenizer)
    logger.info("Tokenizer loaded successfully")

    logger.info("Loading dataset...")
    dataloader = Loader.data(config["data"])
    logger.info(f"Dataloader initialized with {len(dataloader.datasets)} dataset(s)")

    logger.info("Loading pipeline...")
    pipeline = Loader.pipeline(config["pipeline"])
    logger.info("Pipeline loaded successfully")

    logger.info("Building model...")
    model = Loader.model(config["model"]).to(device)
    assert "model_state_dict" in checkpoint, (
        "Model state could not be found in checkpoint"
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model loaded successfully - "
        f"Total params: {total_params:,}, Trainable: {trainable_params:,}"
    )

    logger.info("Loading optimizer...")
    optimizer = Loader.optimizer(config["optimizer"], model.parameters())

    assert "optimizer_state_dict" in checkpoint, (
        "Optimizer state could not be found in checkpoint"
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info(f"Optimizer loaded: {type(optimizer).__name__}")

    scheduler = None
    scheduler_config = config.get("scheduler")
    if scheduler_config is not None:
        logger.info("Loading scheduler...")
        scheduler = Loader.scheduler(scheduler_config, optimizer)

        assert "scheduler_state_dict" in checkpoint, (
            "Optimizer state could not be found in checkpoint"
        )
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info(f"Scheduler loaded: {type(scheduler).__name__}")

    logger.info("All components loaded successfully from checkpoint")

    return model, dataloader, pipeline, optimizer, scheduler, tokenizer


def create_checkpoint_saver(run_manager: RunManager):
    """Create a hook to save checkpoints at the end of each epoch.

    Args:
        run_manager: The `RunManager` instance for saving checkpoints.

    Returns:
        Hook function to save checkpoints.
    """

    def save_checkpoint_hook(trainer):
        """Save checkpoint at the end of each epoch."""
        # Save best checkpoint if this is the best epoch
        if (
            trainer.state.current_epoch % CHECKPOINT_INTERVAL != 0
            and trainer.state.prev_loss != trainer.state.best_loss
        ):
            return

        checkpoint = Checkpoint(
            run_id=run_manager.run.id,
            training_state=trainer.state_dict(),
            tokenizer=bytes(trainer.pipeline.tokenizer),
            model_state_dict=trainer.model.state_dict(),
            optimizer_state_dict=trainer.optimizer.state_dict(),
        )

        if trainer.scheduler is not None:
            checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()

        # Save epoch checkpoint if its the correct epoch.
        if trainer.state.current_epoch % CHECKPOINT_INTERVAL == 0:
            logger.info(f"Saving checkpoint for epoch {trainer.state.current_epoch}...")
            run_manager.save_checkpoint(
                CheckpointType.EPOCH, checkpoint, epoch_no=trainer.state.current_epoch
            )
            logger.info("Checkpoint saved successfully")

        # Save best checkpoint if this is the best epoch
        if trainer.state.prev_loss == trainer.state.best_loss:
            logger.info("Saving 'best' checkpoint...")
            run_manager.save_checkpoint(CheckpointType.BEST, checkpoint)
            logger.info("Checkpoint saved successfully")

    return save_checkpoint_hook


def train(
    config_path: str | None = None,
    run_id: str | None = None,
    checkpoint_tag: str | None = None,
):
    """Execute a training run.

    Args:
        config_path: Path to the training configuration YAML file.
        run_id: Optional run ID to resume from.
        checkpoint_tag: Optional checkpoint tag to resume from (requires run_id).
    """
    device = get_device()
    logger.info(f"Using device: {device}")

    logger.info(f"Initializing RunManager from {RUN_CONFIG_PATH}...")
    run_manager = RunManager.from_config_file(RUN_CONFIG_PATH)
    logger.info("RunManager initialized")

    if run_id is None:
        logger.info("Starting new training run...")
        config = Config.from_yaml(config_path)
        run_id = run_manager.init_run(config)
        logger.info(f"Initialized new training run with ID: {run_id}")

        model, dataloader, pipeline, optimizer, scheduler, tokenizer = (
            init_training_run(config, device)
        )

        # TODO: Move this into the initialization function.
        logger.info("Saving tokenizer artifact...")
        run_manager.save_artifact("tokenizer", tokenizer)
        logger.info("Tokenizer artifact saved successfully")

        training_state = None
    else:
        logger.info(
            f"Resuming training run '{run_id}' from checkpoint '{checkpoint_tag}'..."
        )

        config, checkpoint = run_manager.resume_run(
            run_id, checkpoint_tag, device=device
        )
        logger.info("Checkpoint loaded successfully")

        model, dataloader, pipeline, optimizer, scheduler, tokenizer = (
            resume_training_run(config, checkpoint, device)
        )

        training_state = TrainingState.from_dict(checkpoint["training_state"])
        logger.info(f"Resuming from epoch {training_state.current_epoch}")

    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        pipeline=pipeline,
        loss_fn=cross_entropy_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        state=training_state,
        **config["training"],
    )
    trainer.register_post_epoch_hook(create_checkpoint_saver(run_manager))
    logger.info("Trainer initialized successfully")

    logger.info("=" * 60)
    logger.info(f"Starting training run: {run_id}")
    logger.info("=" * 60)

    train_metrics = trainer.train()

    logger.info("Saving final checkpoint...")
    final_checkpoint = Checkpoint(
        run_id=run_id,
        training_state=trainer.state_dict(),
        model_state_dict=model.state_dict(),
        optimizer_state_dict=trainer.optimizer.state_dict(),
    )

    if trainer.scheduler is not None:
        final_checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()

    run_manager.save_checkpoint(CheckpointType.FINAL, final_checkpoint)
    logger.info("Final checkpoint saved successfully")
    logger.info(
        f"Training completed in {_format_hours(train_metrics.total_time_taken)}!"
    )


def _format_hours(time: float) -> str:
    seconds = int(time)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}H {minutes}m {seconds}s"


def main():
    """Main entry point for the training script."""
    register_signal_handlers()

    parser = argparse.ArgumentParser(
        description="Train a Lumiére model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config-path",
        dest="config_path",
        type=str,
        default=None,
        help="The path to the training configuration file",
    )

    parser.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        help="The ID of the training run to be resumed",
    )

    parser.add_argument(
        "--checkpoint-tag",
        dest="checkpoint_tag",
        default="best",
        help="The tag of the checkpoint to resume from (e.g., 'epoch:0001', 'best', 'final')",
    )

    args = parser.parse_args()

    try:
        train(
            config_path=args.config_path,
            run_id=args.run_id,
            checkpoint_tag=args.checkpoint_tag,
        )
    except Exception as e:
        logger.error(f"Fatal error: {e}", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
