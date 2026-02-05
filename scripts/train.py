import argparse
import logging
import os
import signal
import sys

import deepscale as ds
import torch
from deepscale import Checkpoint, CheckpointType
from deepscale.storage.clients.azure_blob_storage_client import (
    disable_tokenizer_parallelism,
)

import wandb
from lumiere.config import TrainingConfiguration
from lumiere.data import DataLoader
from lumiere.data.pipelines import TextPipeline
from lumiere.data.tokenizer import Tokenizer, TokenizerLoader
from lumiere.model import ModelBuilder, ModelSpec
from lumiere.training import OptimizerLoader, SchedulerLoader, Trainer
from lumiere.utils import get_device


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)

# Disable Azure blob storage logging
logging.getLogger("azure.storage.blob").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _exit_run(sig, frame):
    """Handle training interruption gracefully."""
    print()
    logger.info("Training halted by user")
    sys.exit(0)


def _register_signal_handlers():
    """Register signal handlers for gracefully stopping training."""
    # Handle Ctrl+C gracefully.
    signal.signal(signal.SIGINT, _exit_run)
    # Handle `kill` command gracefully.
    signal.signal(signal.SIGTERM, _exit_run)


def _init_wandb(run_id: str, train_config):
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_entity is None or wandb_project is None or wandb_api_key is None:
        raise ValueError(
            "WANDB_ENTITY, WANDB_PROJECT, and WANDB_API_KEY environment variables "
            "must be set to enable logging to wandb"
        )

    with disable_tokenizer_parallelism():
        if not wandb.login(key=wandb_api_key, verify=True, relogin=True):
            raise ValueError("Failed to login to wandb")

        wandb_run = wandb.init(
            id=run_id,
            entity=wandb_entity,
            project=wandb_project,
            name=f"lumiere-run-{run_id}",
            config=dict(train_config),
            resume=True,
        )

    return wandb_run


def _init_training_run(config_path: str, device: torch.device):
    train_config = TrainingConfiguration.from_file(config_path)
    run_id, run_manager = ds.init_run(dict(train_config))

    logger.info("Loading the dataset...")
    dataloader = DataLoader.from_config(**train_config.data)
    pipeline = TextPipeline()
    logger.info("Dataset loaded successfully\n")

    logger.info(f"Training new tokenizer for run '{run_id}'...")
    tokenizer = TokenizerLoader.load(**train_config.tokenizer)
    tokenizer.train(dataloader["train"])
    logger.info("Tokenizer trained successfully.\n")
    logger.info("Saving tokenizer...")
    run_manager.save_artifact("tokenizer", bytes(tokenizer))
    logger.info("Tokenizer saved successfully.\n")

    logger.info("Initializing model...")
    spec = ModelSpec(train_config.model)
    model = ModelBuilder.build(spec).to(device)
    logger.info("Model initialized successfully.\n")

    optimizer = OptimizerLoader.load(train_config.optimizer, model.parameters())
    scheduler = SchedulerLoader.load(train_config.scheduler, model.optimizer)

    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        pipeline=pipeline,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        *train_args,
    )

    # TODO: Consider add config and state as properties of `Run`.
    return run_manager, trainer


def _load_training_run(run_id: str, checkpoint_tag: str, device: torch.device):
    train_config, checkpoint, run_manager = ds.resume_run(
        run_id, checkpoint_tag, device=device
    )
    train_config = checkpoint.train_configuration

    logger.info("Loading the dataset...")
    dataloader = DataLoader.from_config(**train_config.data)
    pipeline = TextPipeline()
    logger.info("Dataset loaded successfully\n")

    logger.info(
        f"Loading tokenizer for run '{run_id}' with config:\n{train_config.tokenizer}"
    )
    tokenizer_bytes = run_manager.load_artifact("tokenizer")
    if tokenizer_bytes is None:
        raise ValueError("Could not find tokenizer artifact.")

    tokenizer = Tokenizer.from_bytes(tokenizer_bytes, **train_config.tokenizer)
    logger.info("Tokenizer loaded successfully\n")

    logger.info("Loading model...")
    spec = ModelSpec(checkpoint["model_spec"])
    model = ModelBuilder.build(spec).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Model loaded successfully.")

    optimizer = OptimizerLoader.load(train_config.optimizer, model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler = SchedulerLoader.load(train_config.scheduler, model.optimizer)
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        pipeline=pipeline,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        *train_args,
    )

    return run_manager, train_config, state, model, tokenizer, dataloader


def _save_epoch_checkpoint(run_manager, trainer, model):
    checkpoint = Checkpoint(
        run_id=run_manager.run.id,
        train_config=dict(trainer.config),
        model_state_dict=model.state_dict(),
        optimizer_state_dict=model.optimizer.state_dict(),
        scheduler_state_dict=model.scheduler.state_dict(),
        epoch=trainer.state.current_epoch,
        global_step=trainer.state.global_step,
        prev_loss=trainer.state.prev_loss,
        best_loss=trainer.state.best_loss,
        patience_counter=trainer.state.patience_counter,
        time_taken=trainer.state.total_time_taken,
    )

    logger.info("Saving epoch checkpoint...")
    run_manager.save_checkpoint(CheckpointType.EPOCH, checkpoint)
    logger.info("Checkpoint saved successfully.")


def _save_best_checkpoint():
    logger.info("Saving 'best' checkpoint...")
    run_manager.save_checkpoint(CheckpointType.BEST, checkpoint)
    logger.info("Checkpoint saved successfully")


def _log_train_metrics():
    # Log the training stats to wandb.
    if wandb_run is not None and state.global_step % wandb_log_interval == 0:
        with disable_tokenizer_parallelism():
            wandb_run.log(
                {
                    "train/loss": batch_loss.item(),
                    "train/perplexity": batch_perplexity.item(),
                    "train/lr": current_lr,
                    "train/grad_norm": grad_norm,
                }
            )


def _log_eval_metrics():
    if wandb_run is not None:
        wandb_run.log(
            {
                "validation/loss": metrics.avg_loss,
                "validation/perplexity": metrics.avg_perplexity,
            }
        )


def _train(
    run_id: str = None,
    checkpoint_tag: str = None,
    log_wandb: bool = True,
):
    logger.info("Selecting the training device...")
    device = get_device()
    logger.info(f"Using '{device}' device for this training run.\n")

    if run_id is None:
        logger.info("Starting a new training run.")
        run_manager, trainer = _init_training_run(device)
    else:
        logger.info(f"Resuming previous training run with ID: {run_id}")
        run_manager, trainer = _load_training_run(run_id, checkpoint_tag, device)

    # wandb_run = None
    # if log_wandb:
    #     wandb_run = _init_wandb(run_manager.run.id, train_config)
    #     wandb_run.watch(model, log="all")

    # trainer.register_post_epoch_hook(_save_checkpoint)
    #
    # logger.info(
    #     f"Starting training run '{run_manager.run.id}' with config:\n{train_config}"
    #     f"Total params: {sum(p.numel() for p in model.parameters()):,}, "
    #     f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    # )

    trainer.train()

    # run_manager.save_checkpoint(CheckpointType.FINAL, checkpoint)


if __name__ == "__main__":
    _register_signal_handlers()

    parser = argparse.ArgumentParser(description="Train a model")

    parser.add_argument("config_path")
    parser.add_argument(
        "--checkpoint-tag",
        dest="checkpoint_tag",
        default=None,
        help="The checkpoint to resume training from",
    )
    parser.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        help="The run ID to use (required when loading from checkpoint)",
    )
    parser.add_argument(
        "--disable-wandb-logging",
        action="store_false",
        dest="log_wandb",
        default=True,
        help="Do not log to wandb",
    )

    args = parser.parse_args()

    _train(
        run_id=args.run_id,
        checkpoint_tag=args.checkpoint_tag,
        log_wandb=args.log_wandb,
    )
