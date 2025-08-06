import argparse
import itertools
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

import torch
from azure.storage.blob import BlobServiceClient

import wandb
from lumiere.config.config import Config
from lumiere.data.dataloader import DataLoaderFactory
from lumiere.models.transformer import Transformer
from lumiere.persistence.checkpoint_manager import CheckpointManager
from lumiere.persistence.storage_client import (
    LocalStorageClient,
    RemoteStorageClient,
    disable_tokenizer_parallelism,
)
from lumiere.persistence.tokenizer_manager import TokenizerManager
from lumiere.preprocessing.batch_manager import BatchManager
from lumiere.preprocessing.tokenizer import SPECIAL_TOKENS, Tokenizer
from lumiere.training import schedulers
from lumiere.training.checkpoint import Checkpoint, CheckpointType
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


def signal_handler(sig, frame):
    """Handle training interruption gracefully."""
    print()
    logger.info("Training halted by user")
    sys.exit(0)


def _find_run(run_id: str) -> str:
    """Find the run name for the given run ID."""
    for run_path in Path("runs").iterdir():
        if run_path.is_dir() and run_id in run_path.name:
            return run_path.name
    return None


def main(
    run_id: str = None,
    checkpoint_name: str = None,
    checkpoint_manager: CheckpointManager = None,
    tokenizer_manager: TokenizerManager = None,
    log_wandb: bool = True,
):
    # Handle Ctrl+C gracefully.
    signal.signal(signal.SIGINT, signal_handler)

    # Determine the run ID and name.
    if run_id is None:
        run_id = wandb.util.generate_id()
        run_name = f"run_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting new training run with ID: {run_id}")
    else:
        run_name = _find_run(run_id)
        if run_name is None:
            raise ValueError(f"Run with ID '{run_id}' not found")
        logger.info(f"Resuming training run with ID: {run_id}")

    # Select the device to use for training.
    logger.info("Selecting device...")
    device = get_device()
    logger.info(f"Using device: {device} for this training run.\n")

    # Load the checkpoint if one is specified.
    checkpoint = None
    if checkpoint_name is not None:
        if checkpoint_manager is None:
            raise ValueError("A checkpoint manager is required to load from checkpoint")

        logger.info(f"Loading checkpoint '{checkpoint_name}' for run '{run_id}'...")
        checkpoint = checkpoint_manager.load_checkpoint(
            run_name, checkpoint_name, device
        )
        logger.info("Checkpoint loaded successfully\n")

    # Load the model config.
    logger.info("Loading model training config...")
    if checkpoint is not None:
        model_config = Config(checkpoint["model_config"])
        logger.info(f"Resuming training run with config:\n{model_config}")
    else:
        model_config = Config.from_file(CONFIG_PATH)
        logger.info(f"Starting new training run with config:\n{model_config}")

    # Load the dataset.
    logger.info("Loading the dataset...")
    dataloader = DataLoaderFactory.get_data_loader(
        dataset_name=model_config.dataset["name"],
        train_dataset_portion=model_config.dataset["train_portion"],
        validation_dataset_portion=model_config.dataset["validation_portion"],
    )
    logger.info("Dataset loaded successfully\n")

    # Load the tokenizer.
    logger.info("Initializing the tokenizer...")
    if checkpoint is not None:
        if tokenizer_manager is None:
            raise ValueError(
                "A tokenizer manager is required to load tokenizer from checkpoint"
            )

        logger.info(
            f"Loading tokenizer for run '{run_id}' with config:\n{model_config.tokenizer}"
        )
        tokenizer = tokenizer_manager.load_tokenizer(run_name)
        logger.info("Tokenizer loaded successfully\n")
    else:
        logger.info(f"Training tokenizer for run '{run_id}'...")
        tokenizer = Tokenizer(**model_config.tokenizer).train(dataloader.iter_train())
        logger.info("Tokenizer trained successfully\n")

        if tokenizer_manager is not None:
            logger.info("Saving tokenizer")
            tokenizer_manager.save_tokenizer(run_name, tokenizer)
            logger.info("Tokenizer saved successfully\n")

    logger.info("Initializing model...")
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        embedding_size=model_config.model["embedding_size"],
        context_size=model_config.model["context_size"],
        num_layers=model_config.model["num_layers"],
        num_heads=model_config.model["num_heads"],
        d_key=model_config.model["d_key"],
        d_value=model_config.model["d_value"],
        d_ff=model_config.model["d_ff"],
        dropout=model_config.model["dropout"],
        padding_id=SPECIAL_TOKENS["padding"].id,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config.training["learning_rate"],
        weight_decay=model_config.training["weight_decay"],
    )

    scheduler = schedulers.cosine_annealing_lr_scheduler(
        optimizer,
        model_config.training["warmup_steps"],
        model_config.training["max_epochs"],
        model_config.training["epoch_steps"],
    )

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    total_time_taken = checkpoint["time_taken"] if checkpoint else 0.0
    best_loss = checkpoint["best_loss"] if checkpoint else float("inf")
    best_perplexity = (
        torch.tensor(best_loss).exp().item() if checkpoint else float("inf")
    )
    global_step = checkpoint["global_step"] if checkpoint else 0
    current_epoch = checkpoint["epoch"] if checkpoint else 0
    patience_counter = checkpoint["patience_counter"] if checkpoint else 0
    patience = model_config.training["patience"]
    stopping_threshold = model_config.training["stopping_threshold"]
    max_epochs = (
        checkpoint["max_epochs"] if checkpoint else model_config.training["max_epochs"]
    )

    logger.info(
        f"Starting training run '{run_name}' with model config:\n{model_config}"
        f"Total params: {sum(p.numel() for p in model.parameters()):,}, "
        f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    logger.info("--------------------------------")

    wandb_run = None
    if log_wandb:
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
                name=run_name,
                config=model_config.config,
                resume=True,
            )
            wandb_run.watch(model, log="all")

    context_batch_manager = BatchManager(
        context_size=model_config.model["context_size"] + 1,
        batch_size=model_config.training["batch_size"],
        padding_token=SPECIAL_TOKENS["padding"].token,
    )

    # Start the training loop.
    for epoch in itertools.count(current_epoch + 1):
        train_state = train(
            run=wandb_run,
            model=model,
            tokenizer=tokenizer,
            data=dataloader.iter_train(),
            current_epoch=epoch,
            global_step=global_step,
            max_epochs=max_epochs,
            batch_manager=context_batch_manager,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_clip_norm=model_config.training["gradient_clip_norm"],
            device=device,
        )
        global_step = train_state.global_step

        logger.info(
            f"EPOCH {epoch:04d} - {'TRAINING':<10}: "
            f"Loss: {train_state.avg_loss:.4f}, "
            f"Perplexity: {train_state.avg_perplexity:.4f}, "
            f"LR: {train_state.current_lr:.2e}, "
            f"Time: {train_state.time_taken:.2f}s, "
        )

        eval_state = evaluate(
            model=model,
            tokenizer=tokenizer,
            data=dataloader.iter_validation(),
            batch_manager=context_batch_manager,
            device=device,
        )
        logger.info(
            f"EPOCH {epoch:04d} - {'VALIDATION':<10}: "
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

        # Update training state.
        if eval_state.avg_loss < best_loss - stopping_threshold:
            best_loss = eval_state.avg_loss
            best_perplexity = eval_state.avg_perplexity
            patience_counter = 0
            logger.info(
                f"New best validation loss: {best_loss:.4f}, "
                f"perplexity: {best_perplexity:.4f}"
            )
        else:
            patience_counter += 1

        total_time_taken += train_state.time_taken + eval_state.time_taken

        checkpoint = Checkpoint(
            run_id=run_id,
            model_config=model_config.config,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            epoch=epoch,
            global_step=global_step,
            max_epochs=max_epochs,
            prev_loss=eval_state.avg_loss,
            best_loss=best_loss,
            patience_counter=patience_counter,
            time_taken=total_time_taken,
        )

        is_checkpoint_interval = (
            epoch % model_config.training["checkpoint_interval"] == 0
        )
        if is_checkpoint_interval and checkpoint_manager is not None:
            logger.info("Saving epoch checkpoint...")
            checkpoint_manager.save_checkpoint(
                run_name, CheckpointType.EPOCH, checkpoint
            )
            logger.info("Checkpoint saved successfully")

        if eval_state.avg_loss == best_loss and checkpoint_manager is not None:
            logger.info("Saving best checkpoint...")
            checkpoint_manager.save_checkpoint(
                run_name, CheckpointType.BEST, checkpoint
            )
            logger.info("Checkpoint saved successfully")

        # Determine if the training should be stopped.
        if epoch >= max_epochs:
            logger.info(f"Training completed after {epoch} epochs")
            break
        if patience_counter >= patience:
            logger.info(f"Training stopped after {patience} epochs without improvement")
            break

        logger.info("--------------------------------")

    logger.info(
        f"Total time taken: {total_time_taken:.2f}s, "
        f"Best validation loss: {best_loss:.4f}, "
        f"Best validation perplexity: {best_perplexity:.4f}"
    )

    if checkpoint_manager is not None:
        checkpoint_manager.save_checkpoint(run_name, CheckpointType.FINAL, checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")

    parser.add_argument(
        "--checkpoint-name",
        dest="checkpoint_name",
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
        "--disable-local-artifacts",
        action="store_false",
        dest="save_local_artifacts",
        default=True,
        help="Do not save artifacts to local storage",
    )
    parser.add_argument(
        "--disable-remote-artifacts",
        action="store_false",
        dest="save_remote_artifacts",
        default=True,
        help="Do not save artifacts to remote storage",
    )
    parser.add_argument(
        "--disable-artifacts",
        action="store_false",
        dest="save_artifacts",
        default=True,
        help="Do not save artifacts to local or remote storage",
    )
    parser.add_argument(
        "--disable-wandb-logging",
        action="store_false",
        dest="log_wandb",
        default=True,
        help="Do not log to wandb",
    )

    args = parser.parse_args()

    if not args.save_artifacts:
        args.save_local_artifacts = False
        args.save_remote_artifacts = False

    local_storage_client = None
    if args.save_local_artifacts:
        local_storage_client = LocalStorageClient()

    remote_storage_client = None
    if args.save_remote_artifacts:
        remote_storage_client = RemoteStorageClient(
            BlobServiceClient.from_connection_string(
                os.getenv("BLOB_STORAGE_CONNECTION_STRING")
            ),
            os.getenv("BLOB_STORAGE_CONTAINER_NAME"),
        )

    # Investigate using factory pattern.
    checkpoint_manager = None
    if remote_storage_client is not None or local_storage_client is not None:
        checkpoint_manager = CheckpointManager(
            remote_storage_client=remote_storage_client,
            local_storage_client=local_storage_client,
        )

    tokenizer_manager = None
    if remote_storage_client is not None or local_storage_client is not None:
        tokenizer_manager = TokenizerManager(
            remote_storage_client=remote_storage_client,
            local_storage_client=local_storage_client,
        )

    if args.checkpoint_name is not None and args.run_id is None:
        raise ValueError("run_id is required when loading from checkpoint")

    main(
        run_id=args.run_id,
        checkpoint_name=args.checkpoint_name,
        checkpoint_manager=checkpoint_manager,
        tokenizer_manager=tokenizer_manager,
        log_wandb=args.log_wandb,
    )
