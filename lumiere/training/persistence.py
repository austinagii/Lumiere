import logging
import os
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn
from azure.storage.blob import BlobServiceClient

from lumiere.config.config import ModelConfig
from lumiere.models import Transformer
from lumiere.preprocessing.tokenizer import Tokenizer


CHECKPOINT_DIR_TEMPLATE = "artifacts/checkpoints/{model_name}"
CHECKPOINT_NAME_TEMPLATE = "{checkpoint_name}.pth"

MODEL_OUTPUT_DIR = "artifacts/models"
MODEL_OUTPUT_PATH_TEMPLATE = "artifacts/models/{model_name}.pth"

LOCAL_MODEL_PATH_TEMPLATE = "artifacts/models/{model_name}.pth"
REMOTE_MODEL_PATH_TEMPLATE = "models/{model_name}.pth"
LOCAL_CHECKPOINT_PATH_TEMPLATE = (
    "artifacts/checkpoints/{model_name}/{checkpoint_name}.pth"
)
REMOTE_CHECKPOINT_PATH_TEMPLATE = "checkpoints/{model_name}/{checkpoint_name}.pth"
LOCAL_TOKENIZER_PATH_TEMPLATE = "artifacts/tokenizers/{tokenizer_name}.json"
REMOTE_TOKENIZER_PATH_TEMPLATE = "tokenizers/{tokenizer_name}.json"

logger = logging.getLogger(__name__)


class PersistenceError(Exception):
    def __init__(self, message: str, e: Exception = None) -> None:
        super().__init__(message, e)


class CheckpointType(StrEnum):
    EPOCH = auto()
    BEST = auto()
    FINAL = auto()


@dataclass
class Checkpoint:
    type_: CheckpointType
    value: Any = None

    def __str__(self):
        return f"{self.type_}:{self.value}"


def save_checkpoint(
    type: CheckpointType,
    model_name: str,
    model_config: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    epoch: int = None,
    prev_loss: float = None,
    best_loss: float = None,
    patience_counter: int = None,
    time_taken=None,
) -> None:
    checkpoint = {
        "model_config": model_config._config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "prev_loss": prev_loss,
        "best_loss": best_loss,
        "patience_counter": patience_counter,
        "time_taken": time_taken,
    }

    local_checkpoint_path, remote_checkpoint_path = get_checkpoint_path(
        model_name, checkpoint=Checkpoint(type, epoch)
    )

    # Create the checkpoint directory if it does not already exist.
    if not local_checkpoint_path.parent.exists():
        try:
            local_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise PersistenceError(
                f"Failed to create checkpoint directory '{local_checkpoint_path.parent}'",
                e,
            )

    try:
        torch.save(checkpoint, local_checkpoint_path)
    except Exception as e:
        raise PersistenceError(
            f"An error occurred while saving model checkpointto {local_checkpoint_path}",
            e,
        )

    upload_artifact(local_checkpoint_path, remote_checkpoint_path)


def load_checkpoint(
    model_name: str,
    checkpoint: Checkpoint,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    """Load the checkpoint from local storage or blob storage.

    If the checkpoint is not found in local storage, it will be first be downloaded
    to local storage from blob storage.
    """

    # Sync the checkpoint to the local device if it is not already available on device.
    local_checkpoint_path, remote_checkpoint_path = get_checkpoint_path(
        model_name, checkpoint
    )
    if not local_checkpoint_path.exists():
        download_artifact(remote_checkpoint_path, local_checkpoint_path)

    # Load the checkpoint.
    try:
        checkpoint = torch.load(local_checkpoint_path, map_location=device)
    except Exception as e:
        raise PersistenceError("An error occurred while loading the checkpoint", e)

    checkpoint["model_config"] = ModelConfig.from_dict(checkpoint["model_config"])
    return checkpoint


def upload_artifact(local_artifact_path: Path, remote_artifact_path: Path) -> None:
    """Uploads the artifact to the remote device"""
    # Temporarily disable tokenizer parallelism to prevent fork conflicts
    # with tokenizers library.
    original_parallelism = os.environ.get("TOKENIZERS_PARALLELISM")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        connection_string = os.getenv("BLOB_STORAGE_CONNECTION_STRING")
        if not connection_string:
            raise PersistenceError(
                "Environment variable BLOB_STORAGE_CONNECTION_STRING is not set"
            )
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )

        container_name = os.getenv("BLOB_STORAGE_CONTAINER_NAME")
        if not container_name:
            raise PersistenceError(
                "Environment variable BLOB_STORAGE_CONTAINER_NAME is not set"
            )
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=str(remote_artifact_path),
        )

        try:
            with open(local_artifact_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
        except Exception as e:
            raise PersistenceError(
                "An error occurred while syncing checkpoint to blob storage", e
            )
    finally:
        # Restore original tokenizer parallelism setting.
        if original_parallelism is not None:
            os.environ["TOKENIZERS_PARALLELISM"] = original_parallelism
        else:
            os.environ.pop("TOKENIZERS_PARALLELISM", None)


def download_artifact(remote_artifact_path: Path, local_artifact_path: Path) -> None:
    """Downloads the artifact to the local device"""
    # Temporarily disable tokenizer parallelism to prevent fork conflicts
    original_parallelism = os.environ.get("TOKENIZERS_PARALLELISM")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        connection_string = os.getenv("BLOB_STORAGE_CONNECTION_STRING")
        if not connection_string:
            raise PersistenceError(
                "Environment variable BLOB_STORAGE_CONNECTION_STRING is not set"
            )
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )

        container_name = os.getenv("BLOB_STORAGE_CONTAINER_NAME")
        if not container_name:
            raise PersistenceError(
                "Environment variable BLOB_STORAGE_CONTAINER_NAME is not set"
            )
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=str(remote_artifact_path)
        )

        # Create the directory if it does not already exist
        local_artifact_directory = local_artifact_path.parent
        if not local_artifact_directory.exists():
            try:
                local_artifact_directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise PersistenceError(
                    "Failed to create checkpoint directory ",
                    "'{local_artifact_directory}'",
                    e,
                )

        if not blob_client.exists():
            raise PersistenceError("Checkpoint could not be found in blob")

        mode = "w" if local_artifact_path.exists() else "x"
        try:
            with open(local_artifact_path, f"{mode}b") as file:
                file.write(blob_client.download_blob().readall())
        except Exception as e:
            raise PersistenceError(
                "An error occurred while syncing blob storage to local", e
            )
    finally:
        # Restore original parallelism setting
        if original_parallelism is not None:
            os.environ["TOKENIZERS_PARALLELISM"] = original_parallelism
        else:
            os.environ.pop("TOKENIZERS_PARALLELISM", None)


def load_model(
    model_name: str,
    checkpoint: Checkpoint = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, ModelConfig, Tokenizer]:
    """Load the model from local storage or blob storage.

    This function returns the model, fully constructed from the model config
    with it's weights loaded from either the final checkpoint of or the specified
    checkpoint. If the checkpoint is not found in local storage, it is synced from
    blob storage.

    Raises:
        - ConfigError: If the model config is not found.
        - PersistenceError: If the model cannot be loaded.
    """
    checkpoint = load_checkpoint(model_name, checkpoint=checkpoint, device=device)
    model_config = checkpoint["model_config"]
    tokenizer = load_tokenizer(model_config.model["tokenizer"])

    # Load and initialize the model.
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
    )

    # Initialize the model weights.
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict, strict=True)
    model.to(device)

    return model, model_config, tokenizer


def save_tokenizer(tokenizer_name, tokenizer: Tokenizer) -> Path:
    local_path, remote_path = get_tokenizer_path(tokenizer_name)
    if not local_path.parent.exists():
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise PersistenceError(
                "An error occurred while creating tokenizer directory "
                f"'{str(local_path)}'",
                e,
            )

    try:
        tokenizer.save(str(local_path))
    except Exception as e:
        raise PersistenceError(
            f"An error occurred while saving tokenizer to {str(local_path)}", e
        )

    upload_artifact(local_path, remote_path)


def load_tokenizer(tokenizer_name: str) -> Tokenizer:
    local_path, remote_path = get_tokenizer_path(tokenizer_name)
    if not local_path.exists():
        download_artifact(remote_path, local_path)
    tokenizer = Tokenizer.load(str(local_path))
    return tokenizer


def get_checkpoint_path(model_name: str, checkpoint: Checkpoint) -> Tuple[Path, Path]:
    """Returns the path to the specified checkpoint"""

    match checkpoint.type_:
        case CheckpointType.FINAL:
            local_checkpoint_path = Path(
                LOCAL_MODEL_PATH_TEMPLATE.format(model_name=model_name)
            )
            remote_checkpoint_path = Path(
                REMOTE_MODEL_PATH_TEMPLATE.format(model_name=model_name)
            )
        case CheckpointType.BEST:
            local_checkpoint_path = Path(
                LOCAL_CHECKPOINT_PATH_TEMPLATE.format(
                    model_name=model_name, checkpoint_name=CheckpointType.BEST.value
                )
            )
            remote_checkpoint_path = Path(
                REMOTE_CHECKPOINT_PATH_TEMPLATE.format(
                    model_name=model_name, checkpoint_name=CheckpointType.BEST.value
                )
            )
        case CheckpointType.EPOCH:
            checkpoint_name = f"{checkpoint.type_.value}_{checkpoint.value:04d}"
            local_checkpoint_path = Path(
                LOCAL_CHECKPOINT_PATH_TEMPLATE.format(
                    model_name=model_name, checkpoint_name=checkpoint_name
                )
            )
            remote_checkpoint_path = Path(
                REMOTE_CHECKPOINT_PATH_TEMPLATE.format(
                    model_name=model_name, checkpoint_name=checkpoint_name
                )
            )
    return local_checkpoint_path, remote_checkpoint_path


def get_tokenizer_path(tokenizer_name: str) -> Tuple[Path, Path]:
    """Returns a tuple of the local and remote paths to the specified tokenizer"""
    return (
        Path(LOCAL_TOKENIZER_PATH_TEMPLATE.format(tokenizer_name=tokenizer_name)),
        Path(REMOTE_TOKENIZER_PATH_TEMPLATE.format(tokenizer_name=tokenizer_name)),
    )
