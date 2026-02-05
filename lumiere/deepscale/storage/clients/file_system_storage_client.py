"""File System storage client for training artifact management.

This module provides the FileSystemStorageClient class which implements the storage
client interface for storing and retrieving training artifacts using the local file
system.

Example:
    >>> fs_storage_client = FileSystemStorageClient("./")
    >>> fs_storage_client.init_run("run-123", {max_epochs: 10})
    >>> fs_storage_client.save_checkpoint(
    >>>     "run-123", "best", Checkpoint(avg_loss=0.15324)
    >>> )
"""

import pickle
from pathlib import Path
from typing import Any

import yaml

from lumiere.deepscale.storage.errors import (
    CheckpointNotFoundError,
    RunNotFoundError,
    StorageError,
)
from lumiere.deepscale.storage.storage_client import StorageClient


DEFAULT_BASE_DIR = Path(".")
RUN_CONFIG_PATH_TEMPLATE = "runs/{run_id}/config.yaml"
RUN_ARTIFACT_PATH_TEMPLATE = "runs/{run_id}/artifacts/{key}"
RUN_CHECKPOINT_PATH_TEMPLATE = "runs/{run_id}/checkpoints/{checkpoint_tag}.pt"


class FileSystemStorageClient(StorageClient):
    """File system implementation of StorageClient.

    Artifacts managed by this client are stored on the local file system according to
    the following structure:
        {base_dir}/runs/{run_id}/config.yaml            - Training configuration (YAML)
        {base_dir}/runs/{run_id}/checkpoints/{tag}.pt   - Model checkpoints (binary)

    """

    def __init__(self, base_dir: str | Path = None) -> None:
        """Initializes a new FileSystemStorageClient.

        Args:
            base_dir: The directory where artifacts managed by this client are to be
                stored.
        """
        if base_dir is None:
            base_dir = DEFAULT_BASE_DIR

        if isinstance(base_dir, str):
            base_dir = Path(base_dir)

        # TODO: Raise error if base_diir is invalid type.
        self._base_dir = base_dir

    def init_run(self, run_id: str, run_config: dict[Any, Any]) -> None:
        """Implements :meth:`StorageClient.init_run`."""
        run_config_file_path = self.base_dir / RUN_CONFIG_PATH_TEMPLATE.format(
            run_id=run_id
        )

        # TODO: Add proper exception handling. What happens if we cant write to
        # the specified base directory.
        run_config_file_path.parent.mkdir(parents=True, exist_ok=True)

        run_config_file_path.write_text(yaml.dump(run_config))

    def resume_run(
        self, run_id: str, checkpoint_tag: str
    ) -> tuple[dict[Any, Any], bytes]:
        """Implements :meth:`StorageClient.resume_run`."""
        run_config_path = self.base_dir / RUN_CONFIG_PATH_TEMPLATE.format(run_id=run_id)

        if not run_config_path.exists():
            raise RunNotFoundError(f"No run with id '{run_id}' could be found.")

        run_config = yaml.safe_load(run_config_path.read_text())

        checkpoint = self.load_checkpoint(run_id, checkpoint_tag)

        return run_config, checkpoint

    def save_checkpoint(
        self, run_id: str, checkpoint_tag: str, checkpoint: bytes
    ) -> None:
        """Implements :meth:`StorageClient.save_checkpoint`."""
        checkpoint_path = self.base_dir / RUN_CHECKPOINT_PATH_TEMPLATE.format(
            run_id=run_id, checkpoint_tag=checkpoint_tag
        )
        # Create the checkpoint directory if it does not already exist.
        if not checkpoint_path.parent.exists():
            try:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise StorageError(
                    f"Failed to create artifact directory '{checkpoint_path.parent}'",
                    e,
                )

        mode = "w" if checkpoint_path.exists() else "x"
        try:
            with open(checkpoint_path, f"{mode}b") as f:
                f.write(checkpoint)
        except Exception as e:
            raise StorageError(
                f"An error occurred while saving the artifact to '{checkpoint_path}'",
                e,
            )

    def load_checkpoint(self, run_id: str, checkpoint_tag: str) -> bytes:
        """Implements :meth:`StorageClient.load_checkpoint`."""
        checkpoint_path = self.base_dir / RUN_CHECKPOINT_PATH_TEMPLATE.format(
            run_id=run_id, checkpoint_tag=checkpoint_tag
        )

        if not checkpoint_path.exists():
            raise CheckpointNotFoundError(
                f"No checkpoint with name '{checkpoint_tag}' could be found."
            )

        return checkpoint_path.read_bytes()

    def save_artifact(
        self, run_id: str, key: str, artifact: Any, overwrite: bool = False
    ) -> None:
        """Save an artifact for the specified training run to the local file system.

        The artifact can be any arbitrary Python object. It will be serialized usiing the
        `pickle` module and stored as a file on the local file system at:
            "{base_dir}/runs/{run_id}/artifacts/{key}"

        Implements :meth:`StorageClient.save_artifact`.
        """
        artifact_path = self.base_dir / RUN_ARTIFACT_PATH_TEMPLATE.format(
            run_id=run_id, key=key
        )

        if not overwrite and artifact_path.exists():
            raise KeyError(f"An artifact with key '{key}' already exists.")

        try:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(pickle.dumps(artifact))
        except Exception as e:
            raise StorageError("An error occurred while saving the artifact", e)

    def load_artifact(self, run_id: str, key: str) -> Any | None:
        """Load an artifact for the specified training run from the local file system.

        The artifact will be loaded from the file system and reconstructed to its
        original type using the `pickle` module.

        Implements :meth:`StorageClient.load_artifact`.
        """
        artifact_path = self.base_dir / RUN_ARTIFACT_PATH_TEMPLATE.format(
            run_id=run_id, key=key
        )

        if not artifact_path.exists():
            return None

        try:
            return pickle.loads(artifact_path.read_bytes())
        except Exception as e:
            raise StorageError("An error occurred while loading the artifact", e)

    @property
    def base_dir(self):
        """The directory where artifacts managed by this client are stored."""
        return self._base_dir
