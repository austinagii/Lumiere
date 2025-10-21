import argparse
import itertools
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
from lumiere.research.src.config.config import Config
from lumiere.research.src.data.dataloader import get_data_loader
from lumiere.research.src.data.preprocessing import to_training_batches
from lumiere.research.src.data.tokenizer import SPECIAL_TOKENS, Tokenizer
from lumiere.research.src.models.transformer import Transformer
from lumiere.research.src.training import schedulers
from lumiere.research.src.training.eval import evaluate
from lumiere.research.src.training.train import train
from lumiere.research.src.utils import get_device


CONFIG_PATH = "lumiere/research/configs/transformer.yaml"


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


def main(
    run_id: str = None,
    checkpoint_tag: str = None,
    log_wandb: bool = True,
):
    # Handle Ctrl+C gracefully.
    signal.signal(signal.SIGINT, signal_handler)

    # Select the device to use for training.
    logger.info("Selecting device...")
    device = get_device()
    logger.info(f"Using device: {device} for this training run.\n")

    # Determine the run ID and name.
    if run_id is None:
        model_config = Config.from_file(CONFIG_PATH)
        run_id, run_manager = ds.init_run(model_config.config)
        logger.info(f"Starting new training run with ID: {run_id}")
        checkpoint = None
    else:
        run_config, checkpoint, run_manager = ds.resume_run(
            run_id, checkpoint_tag, device=device
        )
        model_config = Config(run_config)
        logger.info(f"Resuming training run with ID: {run_id}")

    # Load the dataset.
    logger.info("Loading the dataset...")
    dataloader = get_data_loader(
        dataset_name=model_config.dataset["name"],
        train_dataset_percentage=model_config.dataset["train_portion"],
        validation_dataset_percentage=model_config.dataset["validation_portion"],
    )
    logger.info("Dataset loaded successfully\n")

    # Load the tokenizer.
    logger.info("Initializing the tokenizer...")
    if checkpoint is None:
        logger.info(f"Training new tokenizer for run '{run_id}'...")
        tokenizer = Tokenizer(**model_config.tokenizer).train(dataloader.iter_train())
        logger.info("Tokenizer trained successfully\n")

        logger.info("Saving tokenizer")
        run_manager.save_artifact("tokenizer", bytes(tokenizer))
    else:
        logger.info(
            f"Loading tokenizer for run '{run_id}' with config:\n"
            f"{model_config.tokenizer}"
        )

        tokenizer_bytes = run_manager.load_artifact("tokenizer")
        if tokenizer_bytes is None:
            raise ValueError("Could not find tokenizer artifact.")

        tokenizer = Tokenizer.from_bytes(tokenizer_bytes, **model_config.tokenizer)
        logger.info("Tokenizer loaded successfully\n")

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
        pre_norm=model_config.model["pre_norm"],
        post_norm=model_config.model["post_norm"],
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
        f"Starting training run '{run_id}' with model config:\n{model_config}"
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
                name=f"lumiere-run-{run_id}",
                config=model_config.config,
                resume=True,
            )
            wandb_run.watch(model, log="all")

    # Start the training loop.
    for epoch in itertools.count(current_epoch + 1):
        train_batches = to_training_batches(
            corpus=dataloader.iter_train(),
            tokenizer=tokenizer,
            context_size=model_config.model["context_size"] + 1,
            batch_size=model_config.training["batch_size"],
            pad_id=SPECIAL_TOKENS["padding"].id,
            sliding_window_size=model_config.dataset["sliding_window_size"],
        )

        train_state = train(
            model=model,
            data=train_batches,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_clip_norm=model_config.training["gradient_clip_norm"],
            global_step=global_step,
            device=device,
            wandb_run=wandb_run,
            wandb_log_interval=model_config.logging["interval"],
        )
        global_step = train_state.global_step

        logger.info(
            f"EPOCH {epoch:04d} - {'TRAINING':<10}: "
            f"Loss: {train_state.avg_loss:.4f}, "
            f"Perplexity: {train_state.avg_perplexity:.4f}, "
            f"LR: {train_state.current_lr:.2e}, "
            f"Time: {train_state.time_taken:.2f}s, "
        )

        validation_batches = to_training_batches(
            corpus=dataloader.iter_validation(),
            tokenizer=tokenizer,
            context_size=model_config.model["context_size"] + 1,
            batch_size=model_config.training["batch_size"],
            pad_id=SPECIAL_TOKENS["padding"].id,
            sliding_window_size=model_config.dataset["sliding_window_size"],
        )

        eval_state = evaluate(
            model=model,
            data=validation_batches,
            device=device,
            wandb_run=wandb_run,
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

        should_save_checkpoint = (
            epoch % model_config.training["checkpoint_interval"] == 0
        )
        if should_save_checkpoint:
            logger.info("Saving epoch checkpoint...")
            run_manager.save_checkpoint(CheckpointType.EPOCH, checkpoint)
            logger.info("Checkpoint saved successfully")

        if eval_state.avg_loss == best_loss:
            logger.info("Saving best checkpoint...")
            run_manager.save_checkpoint(CheckpointType.BEST, checkpoint)
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

    if args.checkpoint_tag is not None and args.run_id is None:
        raise ValueError("run_id is required when loading from checkpoint")

    main(
        run_id=args.run_id,
        checkpoint_tag=args.checkpoint_tag,
        log_wandb=args.log_wandb,
    )
