import argparse
import logging
import os
import signal
import sys
from collections import namedtuple
from dataclasses import dataclass

import deepscale as ds
import torch
from deepscale import Checkpoint, CheckpointType
from deepscale.storage.clients.azure_blob_storage_client import (
    disable_tokenizer_parallelism,
)

import wandb
from lumiere.config.config import Config
from lumiere.data import DataLoader
from lumiere.data.preprocessing import to_training_batches
from lumiere.data.tokenizer import SPECIAL_TOKENS, Tokenizer
from lumiere.models.builder import TransformerBuilder, TransformerSpec
from lumiere.training import schedulers
from lumiere.training.eval import evaluate
from lumiere.training.train import train
from lumiere.utils import get_device


CONFIG_PATH = "configs/transformer.yaml"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)

# Disable Azure blob storage logging
logging.getLogger("azure.storage.blob").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

RestorePoint = namedtuple("RestorePoint", "run_id checkpoint_tag")


@dataclass
class RunState:
    total_time_taken: float = 0.0
    best_loss: float = float("inf")
    best_perplexity: float = float("inf")
    global_step: int = 0
    current_epoch: int = 1
    patience: int = 0
    patience_counter: int = 0
    stopping_threshold: int = 0
    max_epochs: int = 0


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


def _init_wandb(run_id: str, run_config):
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
            config=dict(run_config),
            resume=True,
        )

    return wandb_run


def _init_training_run(device: torch.device):
    run_config = Config.from_file(CONFIG_PATH)
    run_id, run_manager = ds.init_run(dict(run_config))
    run_state = RunState()

    logger.info("Loading the dataset...")
    dataloader = DataLoader(**run_config.dataset)
    logger.info("Dataset loaded successfully\n")

    logger.info(f"Training new tokenizer for run '{run_id}'...")
    tokenizer = Tokenizer(**run_config.tokenizer)
    tokenizer.train(dataloader["train"])
    logger.info("Tokenizer trained successfully.\n")
    logger.info("Saving tokenizer...")
    run_manager.save_artifact("tokenizer", bytes(tokenizer))
    logger.info("Tokenizer saved successfully.\n")

    logger.info("Initializing model...")
    spec = TransformerSpec(run_config.model)
    model = TransformerBuilder.build(spec).to(device)
    model.optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=run_config.training["learning_rate"],
        weight_decay=run_config.training["weight_decay"],
    )
    model.scheduler = schedulers.cosine_annealing_lr_scheduler(
        model.optimizer,
        run_config.training["warmup_steps"],
        run_config.training["max_epochs"],
        run_config.training["epoch_steps"],
    )
    logger.info("Model initialized successfully.")

    run_state.patience = run_config.training["patience"]
    run_state.stopping_threshold = run_config.training["stopping_threshold"]
    run_state.max_epochs = run_config.training["max_epochs"]

    # TODO: Consider add config and state as properties of `Run`.
    return run_manager, run_config, run_state, model, tokenizer, dataloader


def _load_training_run(run_id: str, checkpoint_tag: str, device: torch.device):
    run_config, checkpoint, run_manager = ds.resume_run(
        run_id, checkpoint_tag, device=device
    )
    run_state = RunState()
    model_config = Config(run_config)

    logger.info("Loading the dataset...")
    dataloader = DataLoader(**model_config.dataset)
    logger.info("Dataset loaded successfully\n")

    logger.info(
        f"Loading tokenizer for run '{run_id}' with config:\n{model_config.tokenizer}"
    )
    tokenizer_bytes = run_manager.load_artifact("tokenizer")
    if tokenizer_bytes is None:
        raise ValueError("Could not find tokenizer artifact.")

    tokenizer = Tokenizer.from_bytes(tokenizer_bytes, **model_config.tokenizer)
    logger.info("Tokenizer loaded successfully\n")

    logger.info("Loading model...")
    spec = TransformerSpec(checkpoint["transformer_spec"])
    model = TransformerBuilder.build(spec).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config.training["learning_rate"],
        weight_decay=model_config.training["weight_decay"],
    )
    model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.scheduler = schedulers.cosine_annealing_lr_scheduler(
        model.optimizer,
        model_config.training["warmup_steps"],
        model_config.training["max_epochs"],
        model_config.training["epoch_steps"],
    )
    model.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    logger.info("Model loaded successfully.")

    run_state.total_time_taken = checkpoint["time_taken"]
    run_state.best_loss = checkpoint["best_loss"]
    run_state.best_perplexity = torch.tensor(run_state.best_loss).exp().item()
    run_state.global_step = checkpoint["global_step"]
    run_state.current_epoch = checkpoint["epoch"]
    run_state.patience = checkpoint["patience"]
    run_state.patience_counter = checkpoint["patience_counter"]
    run_state.stopping_threshold = checkpoint["stopping_threshold"]
    run_state.max_epochs = checkpoint["max_epochs"]

    return run_manager, run_config, run_state, model, tokenizer, dataloader


def _train_model(
    model, tokenizer, dataloader, run_config, run_state, wandb_run, device
):
    train_batches = to_training_batches(
        corpus=dataloader["train"],
        tokenizer=tokenizer,
        context_size=run_config.model["context_size"] + 1,
        batch_size=run_config.training["batch_size"],
        pad_id=SPECIAL_TOKENS["padding"].id,
        sliding_window_size=run_config.training["sliding_window_size"],
    )

    train_state = train(
        model=model,
        data=train_batches,
        gradient_clip_norm=run_config.training["gradient_clip_norm"],
        global_step=run_state.global_step,
        device=device,
        wandb_run=wandb_run,
        wandb_log_interval=run_config.logging["interval"],
    )

    logger.info(
        f"EPOCH {run_state.current_epoch:04d} - {'TRAINING':<10}: "
        f"Loss: {train_state.avg_loss:.4f}, "
        f"Perplexity: {train_state.avg_perplexity:.4f}, "
        f"LR: {train_state.current_lr:.2e}, "
        f"Time: {train_state.time_taken:.2f}s, "
    )

    return train_state


def _eval_model(model, tokenizer, dataloader, run_config, run_state, wandb_run, device):
    validation_batches = to_training_batches(
        corpus=dataloader["validation"],
        tokenizer=tokenizer,
        context_size=run_config.model["context_size"] + 1,
        batch_size=run_config.training["batch_size"],
        pad_id=SPECIAL_TOKENS["padding"].id,
        sliding_window_size=run_config.training["sliding_window_size"],
    )

    eval_state = evaluate(
        model=model,
        data=validation_batches,
        device=device,
        wandb_run=wandb_run,
    )

    logger.info(
        f"EPOCH {run_state.current_epoch:04d} - {'VALIDATION':<10}: "
        f"Loss: {eval_state.avg_loss:.4f}, "
        f"Perplexity: {eval_state.avg_perplexity:.4f}, "
        f"Time: {eval_state.time_taken:.2f}s, "
    )

    if wandb_run is not None:
        wandb_run.log(
            {
                "validation/loss": eval_state.avg_loss,
                "validation/perplexity": eval_state.avg_perplexity,
            }
        )

    return eval_state


def _create_checkpoint(run_id, run_config, model, run_state, eval_state):
    return Checkpoint(
        run_id=run_id,
        model_config=dict(run_config),
        model_state_dict=model.state_dict(),
        optimizer_state_dict=model.optimizer.state_dict(),
        scheduler_state_dict=model.scheduler.state_dict(),
        epoch=run_state.current_epoch,
        global_step=run_state.global_step,
        max_epochs=run_state.max_epochs,
        prev_loss=eval_state.avg_loss,
        best_loss=run_state.best_loss,
        patience_counter=run_state.patience_counter,
        time_taken=run_state.total_time_taken,
    )


def _execute_epoch(
    model, tokenizer, dataloader, run_config, run_state, run_manager, wandb_run, device
):
    train_state = _train_model(
        model,
        tokenizer,
        dataloader,
        run_config,
        run_state,
        wandb_run,
        device,
    )
    run_state.global_step += train_state.global_step
    run_state.total_time_taken += train_state.time_taken

    eval_state = _eval_model(
        model, tokenizer, dataloader, run_config, run_state, wandb_run, device
    )
    run_state.total_time_taken += eval_state.time_taken

    checkpoint = _create_checkpoint(
        run_manager.run.id, run_config, model, run_state, eval_state
    )

    # Update training state.
    if eval_state.avg_loss < run_state.best_loss - run_state.stopping_threshold:
        run_state.best_loss = eval_state.avg_loss
        run_state.best_perplexity = eval_state.avg_perplexity
        run_state.patience_counter = 0
        logger.info(
            f"New best validation loss: {run_state.best_loss:.4f}, "
            f"perplexity: {run_state.best_perplexity:.4f}"
        )
        logger.info("Saving 'best' checkpoint...")
        run_manager.save_checkpoint(CheckpointType.BEST, checkpoint)
        logger.info("Checkpoint saved successfully")
    else:
        run_state.patience_counter += 1

    run_state.current_epoch += 1
    should_save_checkpoint = (
        run_state.current_epoch % run_config.training["checkpoint_interval"] == 0
    )
    if should_save_checkpoint:
        logger.info("Saving epoch checkpoint...")
        run_manager.save_checkpoint(CheckpointType.EPOCH, checkpoint)
        logger.info("Checkpoint saved successfully")

    return run_state, checkpoint


# @signaled
def _main(
    restore_point: RestorePoint | None,
    log_wandb: bool = True,
):
    logger.info("Selecting the training device...")
    device = get_device()
    logger.info(f"Using '{device}' device for this training run.")

    if restore_point is None:
        logger.info("Starting a new training run.")
        run_manager, run_config, run_state, model, tokenizer, dataloader = (
            _init_training_run(device)
        )
    else:
        logger.info(f"Resuming previous training run with ID: {restore_point.run_id}")
        run_manager, run_config, run_state, model, tokenizer, dataloader = (
            _load_training_run(
                restore_point.run_id, restore_point.checkpoint_tag, device
            )
        )

    wandb_run = None
    if log_wandb:
        wandb_run = _init_wandb(run_manager.run.id, run_config)
        wandb_run.watch(model, log="all")

    logger.info(
        f"Starting training run '{run_manager.run.id}' with config:\n{run_config}"
        f"Total params: {sum(p.numel() for p in model.parameters()):,}, "
        f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    logger.info("--------------------------------")

    while True:
        run_state, checkpoint = _execute_epoch(model, tokenizer, dataloader, run_config, run_state, run_manager, wandb_run, device)  # fmt: skip

        if run_state.patience_counter >= run_state.patience:
            logger.info(
                f"Training stopped after {run_state.patience} epochs without improvement."  # noqa: E501
            )
            break

        if run_state.current_epoch > run_state.max_epochs:
            logger.info(f"Training completed after {run_state.current_epoch} epochs.")
            break

        logger.info("--------------------------------")

    logger.info(
        f"Total time taken: {run_state.total_time_taken:.2f}s, "
        f"Best validation loss: {run_state.best_loss:.4f}, "
        f"Best validation perplexity: {run_state.best_perplexity:.4f}"
    )

    run_manager.save_checkpoint(CheckpointType.FINAL, checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")

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

    restore_point: RestorePoint | None = None
    if args.run_id is not None:
        restore_point = RestorePoint(
            args.run_id, args.checkpoint_tag if args.checkpoint_tag else "latest"
        )

    _register_signal_handlers()

    _main(
        restore_point=restore_point,
        log_wandb=args.log_wandb,
    )
