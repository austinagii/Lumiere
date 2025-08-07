from pathlib import Path
from typing import Any, Optional

import torch

from lumiere.persistence.errors import PersistenceError
from lumiere.persistence.storage_client import LocalStorageClient, RemoteStorageClient
from lumiere.training.checkpoint import Checkpoint, CheckpointType


CHECKPOINT_PATH_TEMPLATE = "runs/{run_name}/checkpoints/{checkpoint_name}.pth"


class CheckpointManager:
    def __init__(
        self,
        remote_storage_client: RemoteStorageClient = None,
        local_storage_client: LocalStorageClient = None,
        should_cache: bool = True,
    ):
        self.remote_storage_client = remote_storage_client
        self.local_storage_client = local_storage_client
        self.should_cache = should_cache

    def save_checkpoint(
        self,
        run_name: str,
        checkpoint_type: CheckpointType,
        checkpoint: Checkpoint,
    ) -> None:
        checkpoint_bytes = bytes(checkpoint)
        checkpoint_name = _get_checkpoint_name(checkpoint_type, checkpoint)
        checkpoint_path = self._get_checkpoint_path(run_name, checkpoint_name)

        if self.local_storage_client is not None:
            self.local_storage_client.store(checkpoint_path, checkpoint_bytes)

        if self.remote_storage_client is not None:
            self.remote_storage_client.store(checkpoint_path, checkpoint_bytes)

    def load_checkpoint(
        self,
        run_name: str,
        checkpoint_name: str,
        device: torch.device = torch.device("cpu"),
    ) -> dict[str, Any]:
        """Load the checkpoint from local storage or blob storage.

        If the checkpoint is not found in local storage, it will be first be downloaded
        to local storage from blob storage.
        """
        checkpoint_path = self._get_checkpoint_path(run_name, checkpoint_name)

        # TODO: Allow checkpoint to be overwritten by remote.
        if self.local_storage_client is not None and self.local_storage_client.exists(
            checkpoint_path
        ):
            checkpoint_bytes = self.local_storage_client.retrieve(checkpoint_path)
        else:
            if (
                self.remote_storage_client is not None
                and self.remote_storage_client.exists(checkpoint_path)
            ):
                checkpoint_bytes = self.remote_storage_client.retrieve(checkpoint_path)
            else:
                raise PersistenceError("The specified checkpoint could not be found")

            if self.local_storage_client is not None:
                self.local_storage_client.store(checkpoint_path, checkpoint_bytes)

        try:
            loaded_checkpoint = Checkpoint.from_bytes(checkpoint_bytes, device)
        except Exception as e:
            raise PersistenceError("An error occurred while loading the checkpoint", e)

        return loaded_checkpoint

    def _get_checkpoint_path(self, run_name: str, checkpoint_name: str) -> Path:
        """Returns the path to the specified model checkpoint"""

        checkpoint_type, checkpoint_uid = _parse_checkpoint_name(checkpoint_name)

        match checkpoint_type:
            case CheckpointType.EPOCH:
                epoch = f"{checkpoint_uid:04d}"  # Format the uid as an epoch number.
                checkpoint_name = f"{checkpoint_type}_{epoch}"
                checkpoint_path = Path(
                    CHECKPOINT_PATH_TEMPLATE.format(
                        run_name=run_name, checkpoint_name=checkpoint_name
                    )
                )
            case _:
                checkpoint_path = Path(
                    CHECKPOINT_PATH_TEMPLATE.format(
                        run_name=run_name, checkpoint_name=checkpoint_type.value
                    )
                )
        return checkpoint_path


def _get_checkpoint_name(
    checkpoint_type: CheckpointType, checkpoint: Checkpoint
) -> str:
    match checkpoint_type:
        case CheckpointType.EPOCH:
            return f"{checkpoint_type.value}:{checkpoint['epoch']:04d}"
        case _:
            return str(checkpoint_type)


def _parse_checkpoint_name(name: str) -> tuple[CheckpointType, Optional[Any]]:
    if len(name.strip()) == 0:
        raise ValueError("The checkpoint name cannot be empty")

    checkpoint_parts = name.split(":")

    try:
        checkpoint_type = CheckpointType(checkpoint_parts[0])
    except ValueError:
        raise ValueError(f"The checkpoint type '{checkpoint_parts[0]}' is not valid")

    checkpoint_uid = None
    if len(checkpoint_parts) > 1:
        match checkpoint_type:
            case CheckpointType.EPOCH:
                try:
                    checkpoint_uid = int(checkpoint_parts[1])
                except ValueError:
                    raise ValueError(
                        f"The checkpoint uid '{checkpoint_parts[1]}' is not valid"
                    )
            case _:
                raise ValueError(
                    f"The checkpoint type '{checkpoint_type}' does not support a uid"
                )

    return checkpoint_type, checkpoint_uid
