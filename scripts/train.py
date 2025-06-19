import argparse
import itertools
import logging
import os
import signal
import sys

import datasets
import torch

from lumiere.config.config import ModelConfig, TokenizerConfig
from lumiere.models.transformer import Transformer
from lumiere.preprocessing.tokenizer import Tokenizer
from lumiere.training import schedulers
from lumiere.training.eval import evaluate
from lumiere.training.persistence import (
    Checkpoint,
    CheckpointType,
    load_checkpoint,
    load_tokenizer,
    save_checkpoint,
    save_tokenizer,
)
from lumiere.training.train import train
from lumiere.utils import get_device
from scripts import cli


MODEL_CONFIG_PATH_TEMPLATE = "configs/models/{}.yaml"
TOKENIZER_CONFIG_PATH_TEMPLATE = "configs/tokenizers/{}.yaml"

DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_PORTION = 100
TEXT_COLUMN_NAME = "text"

CHECKPOINT_INTERVAL = 3

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
    logger.info("Training halted by user")
    sys.exit(0)


def load_configs(model_name: str) -> tuple[ModelConfig, TokenizerConfig]:
    """Load the model and tokenizer configurations."""
    model_config_path = MODEL_CONFIG_PATH_TEMPLATE.format(model_name)
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Config file not found: {model_config_path}")

    logger.info(f"Loading model config from '{model_config_path}'...")
    model_config = ModelConfig(model_config_path)
    logger.info("Model configuration loaded successfully")

    tokenizer_config_path = TOKENIZER_CONFIG_PATH_TEMPLATE.format(
        model_config.model["tokenizer"]
    )
    if not os.path.exists(tokenizer_config_path):
        raise FileNotFoundError(f"Config file not found: {tokenizer_config_path}")

    logger.info(f"Loading tokenizer config from '{tokenizer_config_path}'...")
    tokenizer_config = TokenizerConfig(tokenizer_config_path)
    logger.info("Tokenizer configuration loaded successfully")

    return model_config, tokenizer_config


def load_datasets():
    """Load the training and validation datasets."""
    logger.info("Loading dataset...")
    train_dataset = datasets.load_dataset(
        DATASET_NAME, DATASET_CONFIG, split=f"train[:{DATASET_PORTION}%]"
    )
    logger.info(f"Dataset loaded: {len(train_dataset)} samples")

    validation_dataset = datasets.load_dataset(
        DATASET_NAME, DATASET_CONFIG, split="validation"
    )
    logger.info(f"Validation dataset loaded: {len(validation_dataset)} samples")

    return train_dataset, validation_dataset


def main(model_name: str, checkpoint: Checkpoint = None):
    # Register signal handler for graceful Ctrl+C handling
    signal.signal(signal.SIGINT, signal_handler)

    should_resume_checkpoint = checkpoint is not None
    train_dataset, validation_dataset = load_datasets()

    device = get_device()
    logger.info(f"Using device: {device}")

    # Load the specified checkpoint or load the latest config for training.
    if should_resume_checkpoint:
        logger.info(f"Resuming from checkpoint: '{checkpoint}'...")
        try:
            checkpoint = load_checkpoint(model_name, checkpoint, device)
        except Exception as e:
            raise RuntimeError(f"Checkpoint '{checkpoint}' could not be found", e)
        model_config = checkpoint["model_config"]
        logger.info("Checkpoint loaded successfully")
    else:
        model_config, tokenizer_config = load_configs(model_name)

    # Load the checkpoint's configured tokenizer or train one from scratch.
    if should_resume_checkpoint:
        tokenizer_name = model_config.model.get("tokenizer")
        if tokenizer_name is None:
            raise Exception(
                "Tokenizer not configured for checkpoint '{checkpoint_name}'"
            )
        logger.info(f"Loading tokenizer: '{model_config.model['tokenizer']}")
        tokenizer = load_tokenizer(model_config.model["tokenizer"])
        logger.info("Tokenizer loaded successfully")
    else:
        logger.info(f"Training tokenizer with configuration:\n{tokenizer_config}")
        tokenizer = Tokenizer().train(
            train_dataset,
            TEXT_COLUMN_NAME,
            tokenizer_config.tokenizer["batch_size"],
            tokenizer_config.tokenizer["vocab_size"],
        )
        logger.info("Tokenizer trained successfully")
        logger.info("Saving tokenizer")
        save_tokenizer(model_config.model["tokenizer"], tokenizer)
        logger.info("Tokenizer saved successfully")

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
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config.training["learning_rate"],
        weight_decay=model_config.training["weight_decay"],
    )

    scheduler = schedulers.cosine_annealing_lr_scheduler(
        optimizer,
        model_config.training["warmup_steps"],
        model_config.training["num_epochs"],
    )

    if should_resume_checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    total_time_taken = checkpoint["time_taken"] if should_resume_checkpoint else 0.0
    best_loss = checkpoint["best_loss"] if should_resume_checkpoint else float("inf")
    best_perplexity = (
        torch.tensor(best_loss).exp().item()
        if should_resume_checkpoint
        else float("inf")
    )
    current_epoch = checkpoint["epoch"] if should_resume_checkpoint else 0
    patience_counter = checkpoint["patience_counter"] if should_resume_checkpoint else 0
    patience = model_config.training["patience"]
    max_epochs = (
        model_config.training["num_epochs"]
        if model_config.training["num_epochs"] > 0
        else float("inf")
    )

    logger.info(f"Training model with configuration:\n{model_config}")
    logger.info(
        f"Starting training for model '{model_name}', "
        f"Total params: {sum(p.numel() for p in model.parameters()):,}, "
        f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    logger.info("--------------------------------")

    # Start the training loop.
    for epoch in itertools.count(current_epoch + 1):
        train_state = train(
            model=model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            current_epoch=epoch,
            max_epochs=max_epochs,
            batch_size=model_config.training["batch_size"],
            context_size=model_config.model["context_size"],
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_clip_norm=model_config.training["gradient_clip_norm"],
            device=device,
        )
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
            validation_dataset=validation_dataset,
            batch_size=model_config.training["batch_size"],
            context_size=model_config.model["context_size"],
            device=device,
        )
        logger.info(
            f"EPOCH {epoch:04d} - {'VALIDATION':<10}: "
            f"Loss: {eval_state.avg_loss:.4f}, "
            f"Perplexity: {eval_state.avg_perplexity:.4f}, "
            f"Time: {eval_state.time_taken:.2f}s, "
        )

        total_time_taken += train_state.time_taken + eval_state.time_taken

        # Capture performance improvements.
        if eval_state.avg_loss < best_loss:
            best_loss = eval_state.avg_loss
            best_perplexity = eval_state.avg_perplexity
            patience_counter = 0
            logger.info(
                f"New best validation loss: {best_loss:.4f}, "
                f"perplexity: {best_perplexity:.4f}"
            )
            logger.info("Saving best checkpoint...")
            save_checkpoint(
                CheckpointType.BEST,
                model_name,
                model_config,
                model,
                optimizer,
                scheduler,
                epoch,
                eval_state.avg_loss,
                best_loss,
                patience_counter,
                total_time_taken,
            )
            logger.info("Checkpoint saved successfully")
        else:
            patience_counter += 1

        # Determine if the training should be stopped.
        if epoch >= max_epochs:
            logger.info(f"Training completed after {epoch} epochs")
            break
        if patience_counter >= patience:
            logger.info(f"Training stopped after {patience} epochs without improvement")
            logger.info("--------------------------------")
            break

        if epoch % CHECKPOINT_INTERVAL == 0:
            logger.info("Saving epoch checkpoint...")
            save_checkpoint(
                CheckpointType.EPOCH,
                model_name,
                model_config,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                prev_loss=eval_state.avg_loss,
                best_loss=best_loss,
                patience_counter=patience_counter,
                time_taken=total_time_taken,
            )
            logger.info("Checkpoint saved successfully")
        logger.info("--------------------------------")

    logger.info(
        f"Total time taken: {total_time_taken:.2f}s, "
        f"Best validation loss: {best_loss:.4f}, "
        f"Best validation perplexity: {best_perplexity:.4f}"
    )

    save_checkpoint(CheckpointType.FINAL, model_name, model_config, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "model", default="transformer-tiny", help="Name of the model to train"
    )
    parser.add_argument(
        "--checkpoint", default=None, help="The checkpoint to resume from"
    )

    args = parser.parse_args()

    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = cli.parse_checkpoint(args.checkpoint)
        if checkpoint is None:
            parser.error("The specified checkpoint is not a valid checkpoint")

    main(args.model, checkpoint=checkpoint)
