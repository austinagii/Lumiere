import argparse
import itertools
import logging
import os
import signal
import sys

import datasets
import torch
from azure.storage.blob import BlobServiceClient

from lumiere.config.config import ModelConfig, TokenizerConfig
from lumiere.models.transformer import Transformer
from lumiere.persistence.checkpoint_manager import CheckpointManager
from lumiere.persistence.errors import PersistenceError
from lumiere.persistence.storage_client import LocalStorageClient, RemoteStorageClient
from lumiere.persistence.tokenizer_manager import TokenizerManager
from lumiere.preprocessing.tokenizer import Tokenizer
from lumiere.training import schedulers
from lumiere.training.checkpoint import Checkpoint, CheckpointType
from lumiere.training.eval import evaluate
from lumiere.training.train import train
from lumiere.utils import get_device


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
    print()
    logger.info("Training halted by user")
    sys.exit(0)


def load_model_config(model_name: str) -> tuple[str, ModelConfig]:
    """Load the model and tokenizer configurations."""
    model_config_path = MODEL_CONFIG_PATH_TEMPLATE.format(model_name)
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Config file not found: {model_config_path}")

    model_config = ModelConfig(model_config_path)

    return model_config_path, model_config


def load_tokenizer_config(model_name: str) -> TokenizerConfig:
    """Load the tokenizer configuration."""
    tokenizer_config_path = TOKENIZER_CONFIG_PATH_TEMPLATE.format(model_name)
    if not os.path.exists(tokenizer_config_path):
        raise FileNotFoundError(f"Config file not found: {tokenizer_config_path}")

    logger.info(f"Loading tokenizer config from '{tokenizer_config_path}'...")
    tokenizer_config = TokenizerConfig(tokenizer_config_path)
    logger.info("Tokenizer configuration loaded successfully")

    return tokenizer_config


def load_datasets():
    """Load the training and validation datasets."""
    train_dataset = datasets.load_dataset(
        DATASET_NAME, DATASET_CONFIG, split=f"train[:{DATASET_PORTION}%]"
    )
    logger.info(f"Dataset loaded: {len(train_dataset)} samples")

    validation_dataset = datasets.load_dataset(
        DATASET_NAME, DATASET_CONFIG, split="validation"
    )
    logger.info(f"Validation dataset loaded: {len(validation_dataset)} samples")

    return train_dataset, validation_dataset

    # connection_string = os.getenv("BLOB_STORAGE_CONNECTION_STRING")
    # if not connection_string:
    #     raise PersistenceError(
    #         "Environment variable BLOB_STORAGE_CONNECTION_STRING is not set"
    #     )
    # blob_service_client = BlobServiceClient.from_connection_string(
    #     connection_string
    # )

    # container_name = os.getenv("BLOB_STORAGE_CONTAINER_NAME")
    # if not container_name:
    #     raise PersistenceError(
    #         "Environment variable BLOB_STORAGE_CONTAINER_NAME is not set"
    #     )


def main(
    model_name: str,
    checkpoint_name: str = None,
    checkpoint_manager: CheckpointManager = None,
    tokenizer_manager: TokenizerManager = None,
):
    # Register signal handler for graceful Ctrl+C handling
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("Selecting device...")
    device = get_device()
    logger.info(f"Using device: {device}\n")

    logger.info("Loading datasets...")
    train_dataset, validation_dataset = load_datasets()
    logger.info("Datasets loaded successfully\n")

    # Load the checkpoint if specified.
    checkpoint = None
    if checkpoint_name is not None:
        if checkpoint_manager is None:
            raise ValueError(
                "Checkpoint manager is required when loading from checkpoint"
            )

        logger.info(f"Loading checkpoint: '{checkpoint_name}'...")
        try:
            checkpoint = checkpoint_manager.load_checkpoint(
                model_name, checkpoint_name, device
            )
            logger.info("Checkpoint loaded successfully\n")
        except Exception as e:
            raise RuntimeError(
                f"Checkpoint '{checkpoint_name}' could not be found: {e}"
            ) from e

    # Load the model config.
    logger.info("Loading model config...")
    if checkpoint is not None:
        model_config = ModelConfig.from_dict(checkpoint["model_config"])
        logger.info(
            f"Model config loaded successfully from checkpoint:\n{model_config}"
        )
    else:
        model_config_path, model_config = load_model_config(model_name)
        logger.info(
            f"Model config loaded successfully from {model_config_path}:\n"
            f"{model_config}"
        )

    # Load the tokenizer.
    logger.info("Initializing tokenizer...")
    tokenizer_name = model_config.model.get("tokenizer")
    if tokenizer_name is None or len(tokenizer_name.strip()) == 0:
        raise ValueError(f"No tokenizer configured for model '{model_name}'")

    tokenizer = None
    if tokenizer_manager is not None:
        logger.info(f"Attempting to load tokenizer: '{tokenizer_name}'")
        try:
            tokenizer = tokenizer_manager.load_tokenizer(tokenizer_name)
            logger.info("Tokenizer loaded successfully\n")
        except PersistenceError as e:
            logger.warning(f"Tokenizer '{tokenizer_name}' could not be found: {e}")

    if tokenizer is None:
        logger.info("Training tokenizer...")
        try:
            tokenizer_config = load_tokenizer_config(tokenizer_name)
        except FileNotFoundError as e:
            raise ValueError(
                f"Tokenizer configuration not found for model '{tokenizer_name}': {e}"
            ) from e

        logger.info(f"Training tokenizer with configuration:\n{tokenizer_config}")
        tokenizer = Tokenizer().train(
            train_dataset,
            TEXT_COLUMN_NAME,
            tokenizer_config.tokenizer["batch_size"],
            tokenizer_config.tokenizer["vocab_size"],
        )
        logger.info("Tokenizer trained successfully\n")

        if tokenizer_manager is not None:
            logger.info("Saving tokenizer")
            tokenizer_manager.save_tokenizer(tokenizer_name, tokenizer)
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

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    total_time_taken = checkpoint["time_taken"] if checkpoint else 0.0
    best_loss = checkpoint["best_loss"] if checkpoint else float("inf")
    best_perplexity = (
        torch.tensor(best_loss).exp().item() if checkpoint else float("inf")
    )
    current_epoch = checkpoint["epoch"] if checkpoint else 0
    patience_counter = checkpoint["patience_counter"] if checkpoint else 0
    patience = model_config.training["patience"]
    stopping_threshold = model_config.training["stopping_threshold"]
    max_epochs = (
        checkpoint["max_epochs"]
        if checkpoint
        else (
            model_config.training["num_epochs"]
            if model_config.training["num_epochs"] > 0
            else float("inf")
        )
    )

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
            model_config=model_config.to_dict(),
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            epoch=epoch,
            max_epochs=max_epochs,
            prev_loss=eval_state.avg_loss,
            best_loss=best_loss,
            patience_counter=patience_counter,
            time_taken=total_time_taken,
        )

        is_checkpoint_interval = epoch % CHECKPOINT_INTERVAL == 0
        if is_checkpoint_interval and checkpoint_manager is not None:
            logger.info("Saving epoch checkpoint...")
            checkpoint_manager.save_checkpoint(
                model_name, CheckpointType.EPOCH, checkpoint
            )
            logger.info("Checkpoint saved successfully")

        if eval_state.avg_loss == best_loss and checkpoint_manager is not None:
            logger.info("Saving best checkpoint...")
            checkpoint_manager.save_checkpoint(
                model_name, CheckpointType.BEST, checkpoint
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
        checkpoint_manager.save_checkpoint(model_name, CheckpointType.FINAL, checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")

    parser.add_argument(
        "model_name",
        default="transformer-tiny",
        help="Name of the model to train",
    )
    parser.add_argument(
        "--checkpoint-name",
        dest="checkpoint_name",
        default=None,
        help="The checkpoint to resume training from",
    )
    parser.add_argument(
        "-l",
        "--disable-local-artifacts",
        action="store_false",
        dest="save_local_artifacts",
        default=True,
        help="Do not save artifacts to local storage",
    )
    parser.add_argument(
        "-r",
        "--disable-remote-artifacts",
        action="store_false",
        dest="save_remote_artifacts",
        default=True,
        help="Do not save artifacts to remote storage",
    )

    args = parser.parse_args()

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

    main(
        args.model_name,
        checkpoint_name=args.checkpoint_name,
        checkpoint_manager=checkpoint_manager,
        tokenizer_manager=tokenizer_manager,
    )
