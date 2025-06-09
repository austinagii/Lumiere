import logging
from typing import Any
from pathlib import Path
from enum import StrEnum, auto
import os

import torch
from azure.storage.blob import BlobServiceClient

from lumiere.config.config import Config, ModelConfig
from lumiere.preprocessing.tokenizer import Tokenizer


CHECKPOINT_DIR_TEMPLATE = "artifacts/checkpoints/{model_name}"
CHECKPOINT_NAME_TEMPLATE = "{checkpoint_name}.pth"
MODEL_OUTPUT_DIR = "artifacts/models"
TOKENIZER_OUTPUT_DIR = "artifacts/tokenizers"
TOKENIZER_PATH_TEMPLATE = "artifacts/tokenizers/{tokenizer_name}.json"

logger = logging.getLogger(__name__)


class PersistenceError(Exception):
    def __init__(self, message: str, e: Exception = None) -> None:
        super().__init__(message, e)


class CheckpointType(StrEnum):
    EPOCH = auto()
    BEST = auto()


def save_checkpoint(
    type: CheckpointType,
    model_name: str,
    model_config: dict[str, Any],
    model: torch.nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    prev_loss: float,
    best_loss: float,
    patience_counter: int,
    time_taken,
) -> None:
    checkpoint_directory_path = Path(
        CHECKPOINT_DIR_TEMPLATE.format(model_name=model_name))

    if not checkpoint_directory_path.exists():
        try:
            checkpoint_directory_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise PersistenceError(
                f"Failed to create checkpoint directory '{checkpoint_directory_path}'", e)

    match type:
        case CheckpointType.EPOCH:
            checkpoint_name = CHECKPOINT_NAME_TEMPLATE.format(
                checkpoint_name=f"{type}_{epoch:04d}")
        case CheckpointType.BEST:
            checkpoint_name = CHECKPOINT_NAME_TEMPLATE.format(
                checkpoint_name=f"{type}")
        case _:
            raise ValueError(f"Invalid checkpoint type: {type}")

    checkpoint = {
        'model_config': model_config._config,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'prev_loss': prev_loss,
        'best_loss': best_loss,
        'patience_counter': patience_counter,
        'time_taken': time_taken
    }
    checkpoint_path = checkpoint_directory_path / checkpoint_name
    try:
        torch.save(checkpoint, checkpoint_path)
    except Exception as e:
        raise PersistenceError("An error occurred while saving model checkpoint"
                               f"to {checkpoint_path}", e)


def save_tokenizer(model_config, tokenizer: Tokenizer) -> Path:
    logger.info("Saving tokenizer to disk...")
    tokenizer_directory_path = Path(TOKENIZER_OUTPUT_DIR)
    try:
        tokenizer_directory_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise PersistenceError(f"An error occurred while creating tokenizer directory '{tokenizer_directory_path}'", e)

    tokenizer_path = tokenizer_directory_path / f"{model_config.model['tokenizer']}.json"
    try:
        tokenizer.save(str(tokenizer_path))
    except Exception as e:
        raise PersistenceError(f"An error occurred while saving tokenizer to {tokenizer_path}", e)
    return tokenizer_path

def load_tokenizer(tokenizer_name: str) -> Tokenizer:
    tokenizer_path = TOKENIZER_PATH_TEMPLATE.format(tokenizer_name=tokenizer_name)
    tokenizer = Tokenizer.load(str(tokenizer_path))
    logger.info(
        f"Tokenizer loaded from {tokenizer_path}")
    return tokenizer

def save_model(model_name, model):
    logger.info("Saving model to disk...")
    save_path = Path(MODEL_OUTPUT_DIR)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / f"{model_name}.pth")
    logger.info(f"Model saved to {save_path / f'{model_name}.pth'}")


def load_checkpoint(model_name: str, checkpoint_name: str) -> dict[str, Any]:
    checkpoint_path = CHECKPOINT_DIR_TEMPLATE.format(model_name=model_name) \
        + "/"+ CHECKPOINT_NAME_TEMPLATE.format(checkpoint_name=checkpoint_name)
    try:
        checkpoint = torch.load(checkpoint_path)
    except Exception as e:
        raise PersistenceError(f"An error occurred while loading checkpoint from {checkpoint_path}", e)

    checkpoint['model_config'] = ModelConfig.from_dict(checkpoint['model_config'])
    return checkpoint

def sync_checkpoint_to_blob_storage(model_name: str, checkpoint_name: str) -> None:
    connection_string = os.getenv('BLOB_STORAGE_CONNECTION_STRING')
    if not connection_string:
        raise PersistenceError("Environment variable BLOB_STORAGE_CONNECTION_STRING is not set")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    container_name = os.getenv('BLOB_STORAGE_CONTAINER_NAME')
    if not container_name:
        raise PersistenceError("Environment variable BLOB_STORAGE_CONTAINER_NAME is not set")

    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=f"checkpoints/{model_name}/{checkpoint_name}.pth"
    )

    local_checkpoint_path = CHECKPOINT_DIR_TEMPLATE.format(model_name=model_name) \
        + "/"+ CHECKPOINT_NAME_TEMPLATE.format(checkpoint_name=checkpoint_name)
    try:
        with open(local_checkpoint_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
    except Exception as e:
        raise PersistenceError(f"An error occurred while syncing checkpoint to blob storage", e)

    logger.info(f"Checkpoint '{checkpoint_name}' synced to blob storage")
