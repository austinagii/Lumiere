import argparse
import logging
import os
import signal
import sys
from collections import namedtuple

import deepscale as ds
import torch
from deepscale import Checkpoint, CheckpointType
from deepscale.storage.clients.azure_blob_storage_client import (
    disable_tokenizer_parallelism,
)
from torch.nn import functional as F

import wandb
from lumiere.config.config import Config
from lumiere.data import DataLoader
from lumiere.data.tokenizer import Tokenizer
from lumiere.models.builder import TransformerBuilder, TransformerSpec
from lumiere.training import schedulers
from lumiere.training.state import TrainingState
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


def loss_fn(logits):
    batch_loss = F.cross_entropy(
        logits.view(-1, model.vocab_size),
        y.reshape(-1),
        ignore_index=SPECIAL_TOKENS["padding"].id,
    )
    batch_perplexity = torch.exp(batch_loss)


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
    state = TrainingState()

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
    logger.info("Model initialized successfully.\n")

    # TODO: Consider add config and state as properties of `Run`.
    return run_manager, run_config, state, model, tokenizer, dataloader


def _load_training_run(run_id: str, checkpoint_tag: str, device: torch.device):
    run_config, checkpoint, run_manager = ds.resume_run(
        run_id, checkpoint_tag, device=device
    )
    state = TrainingState()
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

    state.total_time_taken = checkpoint["time_taken"]
    state.prev_loss = checkpoint["prev_loss"]
    state.best_loss = checkpoint["best_loss"]
    state.best_perplexity = torch.tensor(state.best_loss).exp().item()
    state.global_step = checkpoint["global_step"]
    state.current_epoch = checkpoint["epoch"]
    state.patience_counter = checkpoint["patience_counter"]

    return run_manager, run_config, state, model, tokenizer, dataloader


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


def _log_new_loss(state):
    logger.info(
        f"New best validation loss: {state.best_loss:.4f}, "
        f"perplexity: {state.best_perplexity:.4f}"
    )


def _save_best_checkpoint():
    logger.info("Saving 'best' checkpoint...")
    run_manager.save_checkpoint(CheckpointType.BEST, checkpoint)
    logger.info("Checkpoint saved successfully")


def _log_train_metrics():
    logger.info(
        f"EPOCH {state.current_epoch:04d} - {'TRAINING':<10}: "
        f"Loss: {metrics.avg_loss:.4f}, "
        f"Perplexity: {metrics.avg_perplexity:.4f}, "
        f"Time: {state.total_time_taken:.2f}s, "
    )
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


def log_training_metrics():
    logger.info(
        f"EPOCH {state.current_epoch:04d} - {'TRAINING':<10}: "
        f"Loss: {metrics.avg_loss:.4f}, "
        f"Perplexity: {metrics.avg_perplexity:.4f}, "
        f"Time: {state.total_time_taken:.2f}s, "
    )

    if wandb_run is not None:
        wandb_run.log(
            {
                "validation/loss": metrics.avg_loss,
                "validation/perplexity": metrics.avg_perplexity,
            }
        )


# @interruptible
def _train(
    restore_point: RestorePoint | None,
    log_wandb: bool = True,
):
    logger.info("Selecting the training device...")
    device = get_device()
    logger.info(f"Using '{device}' device for this training run.\n")

    trainer.register_post_epoch_hook(
        functools.bind(_save_epoch_checkpoint, run_manager),
        lambda epoch_no: epoch_no % epoch_interval,
    )

    trainer.register_improvement_hook(_log_new_loss)

    if restore_point is None:
        logger.info("Starting a new training run.")
        run_manager, run_config, state, model, tokenizer, dataloader = (
            _init_training_run(device)
        )
    else:
        logger.info(f"Resuming previous training run with ID: {restore_point.run_id}")
        run_manager, run_config, state, model, tokenizer, dataloader = (
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

    logger.info(
        f"Total time taken: {state.total_time_taken:.2f}s, "
        f"Best validation loss: {state.best_loss:.4f}, "
        f"Best validation perplexity: {state.best_perplexity:.4f}"
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

    _train(
        restore_point=restore_point,
        log_wandb=args.log_wandb,
    )
