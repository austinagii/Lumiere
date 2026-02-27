import logging
import os
import secrets
import string
from collections.abc import Iterable
from concurrent import futures
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import Any

import torch
from azure.storage.blob import BlobServiceClient

from lumiere.persistence.clients import (
    AzureBlobStorageClient,
    FileSystemStorageClient,
)
from lumiere.persistence.errors import (
    ArtifactNotFoundError,
    CheckpointNotFoundError,
    RunNotFoundError,
    StorageError,
)
from lumiere.persistence.storage_client import StorageClient
from lumiere.training.checkpoint import Checkpoint, CheckpointType
from lumiere.training.config import Config


LOGGER = logging.getLogger(__name__)
MAX_THREAD_POOL_SIZE: int = 5


@dataclass
class Run:
    """A training run with a unique identifier and configuration.

    Attributes:
        _id: The unique identifier for this run.
        _config: The training configuration for this run.
    """

    _id: str
    _config: Config

    @classmethod
    def from_config(cls, config: Config) -> None:
        """Create a new run with a generated ID from a configuration.

        Args:
            config: The training configuration dictionary.

        Returns:
            A new `Run` instance with a randomly generated ID.
        """
        return cls(generate_run_id(), config)

    @property
    def id(self):
        """The unique identifier for this training run.

        Returns:
            The run ID as a string.
        """
        return self._id

    @property
    def config(self):
        """The training configuration for this run.

        Returns:
            Configuration dictionary containing training parameters.
        """
        return self._config


def generate_run_id(n: int = 8):
    """Generate a random alphanumeric run identifier.

    Args:
        n: The length of the identifier. Defaults to `8`.

    Returns:
        A random string of length `n` containing letters and digits.
    """
    alphabet = string.ascii_letters + string.digits
    return "".join([secrets.choice(alphabet) for _ in range(n)])


class StorageSources(StrEnum):
    AZURE_BLOB = "azure-blob"
    FILESYSTEM = auto()


def init_azure_blob_storage_client():
    """Initialize an Azure Blob Storage client from environment variables.

    Reads the Azure connection string and container name from the environment
    and creates an `AzureBlobStorageClient` instance.

    Returns:
        Configured `AzureBlobStorageClient` instance.

    Raises:
        RuntimeError: If required environment variables are not set.

    Environment Variables:
        AZURE_BLOB_CONNECTION_STRING: Azure Storage account connection string.
        AZURE_BLOB_CONTAINER_NAME: Name of the blob container to use.
    """
    if (connection_string := os.getenv("AZURE_BLOB_CONNECTION_STRING")) is None:
        raise RuntimeError(
            "Required environment variable 'AZURE_BLOB_CONNECTION_STRING' not found."
        )

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    if (blob_container_name := os.getenv("AZURE_BLOB_CONTAINER_NAME")) is None:
        raise RuntimeError(
            "Required environment variable 'AZURE_BLOB_CONTAINER_NAME' not found."
        )

    return AzureBlobStorageClient(blob_service_client, blob_container_name)


class RunManager:
    """Manages training runs with multi-backend storage support.

    Handles initialization, resumption, checkpointing, and artifact storage for
    training runs. Supports multiple storage backends (filesystem, Azure Blob Storage)
    with automatic failover for loading operations.

    Attributes:
        storage_clients: List of storage clients for saving checkpoints and artifacts.
        retrieval_clients: List of storage clients for loading checkpoints and artifacts.
    """

    def __init__(
        self,
        sources: Iterable[StorageClient],
        destinations: Iterable[StorageClient],
        fs_base_path: str | Path = None,
    ):
        client_by_location = {
            StorageSources.AZURE_BLOB: init_azure_blob_storage_client(),
            StorageSources.FILESYSTEM: FileSystemStorageClient(fs_base_path),
        }
        # TODO: Fix this initializer, just pass storage clients directly. Current
        # iteration is clunky.
        self.storage_clients = [client_by_location[source] for source in sources]
        self.retrieval_clients = [
            client_by_location[destination] for destination in destinations
        ]

    @classmethod
    def from_config(cls, config: Config) -> "RunManager":
        """Create a `RunManager` from a configuration object.

        Args:
            config: Configuration object containing storage and checkpoint settings.

        Returns:
            Configured `RunManager` instance.
        """
        fs_base_path = config.get("storage.clients.filesystem.basedir")

        return cls(
            sources=config["runs.checkpoints.sources"],
            destinations=config["runs.checkpoints.destinations"],
            fs_base_path=fs_base_path,
        )

    @classmethod
    def from_config_file(cls, file_path: str | Path) -> "RunManager":
        """Create a `RunManager` from a YAML configuration file.

        Args:
            file_path: Path to the YAML configuration file.

        Returns:
            Configured `RunManager` instance.
        """
        config = Config.from_yaml(file_path)

        return RunManager.from_config(config)

    def init_run(self, run_config: Config) -> str:
        """Initialize a new training run with the specified configuration.

        Creates a new run with a unique identifier and stores the configuration
        across all configured storage destinations. Executes initialization in
        parallel across all destinations using a thread pool.

        Args:
            run_config: Configuration object or dictionary containing the training configuration.

        Returns:
            The unique identifier of the newly initialized run.

        Raises:
            Exception: If initialization fails in any destination.
        """
        self.run = Run.from_config(run_config)

        with futures.ThreadPoolExecutor(
            max_workers=max(len(self.storage_clients), MAX_THREAD_POOL_SIZE)
        ) as executor:
            jobs = [
                executor.submit(
                    client.save_artifact,
                    self.run.id,
                    "config.yaml",
                    self.run.config,
                    False,
                )
                for client in self.storage_clients
            ]

            # Raise any exceptions that occur.
            for job in futures.as_completed(jobs):
                if (exception := job.exception()) is not None:
                    raise exception

        return self.run.id

    def resume_run(
        self, run_id: str, checkpoint_tag: str | None = None, device=None
    ) -> tuple[Config, Checkpoint]:
        """Resume the specified training run.

        This method sequentially iterates over this run manager's configured sources in
        the order that they've been defined, attempting to download the latest state of
        the run. Once the run is found, this run manager is configured with the
        identified run and is ready for use. All future calls to `save_checkpoint` and
        `load_checkpoint` will automatically use this run.

        If `checkpoint_tag` is not specified, then the run will resume from the latest
        checkpoint.

        Args:
            run_id: The unique identifier of the specified run.
            checkpoint_tag: The tag of the checkpoint that the run is to be resumed
                from. Defaults to `None`.

        Returns:
            A tuple containing the following:
                - The configuration used for the specified training run.
                - The checkpoint corresponding to the specified tag.

        Raises:
            RunNotFoundError: If the specified training run could not be found in any
                of the configured sources.
            CheckpointNotFoundError: If the specified checkpoint could not be found in
                any of the configured sources.
        """
        run_config, checkpoint_bytes = None, None

        # Load the run config from the first available source
        for client in self.retrieval_clients:
            try:
                run_config = client.load_artifact(run_id, "config.yaml")
                if run_config is not None:
                    break
            except Exception as e:
                LOGGER.info(f"Could not load config from source: {e}")
                continue

        if run_config is None:
            raise RunNotFoundError(
                f"Run config for run '{run_id}' could not be found in any source"
            )

        # Load the checkpoint from the first available source
        for client in self.retrieval_clients:
            try:
                checkpoint_bytes = client.load_checkpoint(run_id, checkpoint_tag)
                if checkpoint_bytes is not None:
                    break
            except Exception as e:
                LOGGER.info(f"Could not load checkpoint from source: {e}")
                continue

        if checkpoint_bytes is None:
            raise CheckpointNotFoundError(
                f"Checkpoint '{checkpoint_tag}' for run '{run_id}' could not be found"
            )

        self.run = Run(run_id, run_config)
        checkpoint = Checkpoint.from_bytes(checkpoint_bytes, device=device)
        return run_config, checkpoint

    def save_checkpoint(
        self,
        checkpoint_type: CheckpointType,
        checkpoint: Checkpoint,
        epoch_no: int = None,
    ) -> None:
        """Save a checkpoint for the current training run.

        Saves the checkpoint to all configured storage destinations in parallel
        using a thread pool. The checkpoint tag is generated based on the type
        (e.g., `"epoch:0001"` for epoch checkpoints, `"best"` for best checkpoint).

        Args:
            checkpoint_type: Type of checkpoint (EPOCH, BEST, or FINAL).
            checkpoint: The checkpoint object to save.
        """
        checkpoint_tag = self._create_checkpoint_tag(checkpoint_type, epoch_no)

        with futures.ThreadPoolExecutor(
            max_workers=max(len(self.storage_clients), MAX_THREAD_POOL_SIZE)
        ) as executor:
            # Just fire and forget.
            for client in self.storage_clients:
                executor.submit(
                    client.save_checkpoint,
                    self.run.id,
                    checkpoint_tag,
                    bytes(checkpoint),
                )

    def load_checkpoint(
        self,
        checkpoint_tag: str,
        device: torch.device = torch.device("cpu"),
    ) -> dict[str, Any]:
        """Load a checkpoint for the current training run.

        Attempts to load the checkpoint from each configured retrieval source in
        order until successful. If the checkpoint is not found in any source,
        raises an error.

        Args:
            checkpoint_tag: The tag identifying the checkpoint (e.g., `"best"`, `"epoch:0001"`).
            device: The device to load the checkpoint onto. Defaults to CPU.

        Returns:
            Deserialized checkpoint dictionary.

        Raises:
            CheckpointNotFoundError: If the checkpoint is not found in any source.
        """
        for client in self.retrieval_clients:
            try:
                checkpoint_bytes = client.load_checkpoint(self.run.id, checkpoint_tag)
                break
            except (RunNotFoundError, CheckpointNotFoundError):
                continue

        if checkpoint_bytes is None:
            raise CheckpointNotFoundError(
                f"Checkpoint with tag '{checkpoint_tag}' could not be found in any of "
                "the configured sources."
            )

        return Checkpoint.from_bytes(checkpoint_bytes, device)

    def save_artifact(
        self,
        key: str,
        artifact: Any,
        overwrite: bool = False,
        raise_exception: bool = False,
    ) -> None:
        """Save an artifact for the current training run.

        The specified artifact can be an arbitrary python object.

        If an artifact with the specified key already exists, then the `overwrite` flag
        will be used to determine if the previous artifact should be overwritten. If
        `overwrite` is `False` and an artifact with the specified key already exists,
        then an exception will be raised. If `overwrite` is `True` and an artifact with
        the specified key already exists, then the previous artifact will be overwritten.

        If `raise_exception` is `True` then this method will raise the first exception
        that occurs while storing the artifact in the configured sources. Else, it
        will return immediately upon initiating the storage of the artifact, ignoring
        any exceptions that occur.

        Args:
            key: The key that identifies the artifact being stored.
            artifact: The artifact to be stored.
            overwrite: Whether to overwrite any previous artifacts with the same key.
                Defaults to False.
            raise_exception: Whether to raise an exception if an error occurs while
                storing the artifact. Defaults to False.

        Raises:
            StorageError: If an error occurred while storing the artifact. The specific
                error will be nested.
        """
        with futures.ThreadPoolExecutor(
            max_workers=max(len(self.storage_clients), MAX_THREAD_POOL_SIZE)
        ) as executor:
            jobs = []
            for client in self.storage_clients:
                jobs.append(
                    executor.submit(
                        client.save_artifact,
                        self.run.id,
                        key,
                        artifact,
                        overwrite=overwrite,
                    )
                )

            if raise_exception:
                for job in futures.as_completed(jobs):
                    if (exception := job.exception()) is not None:
                        raise StorageError(
                            "An error occurred while storing the artifact.", exception
                        )

            return

    def load_artifact(self, key: str) -> Any:
        """Load an artifact for the current training run.

        The artifact will be returned as it's original type.

        Args:
            key: The key of the object to be loaded.

        Returns:
            Any | None: The artifact as its original type or `None` if no matching
                artifact could be found.
        """
        artifact: Any = None

        # Load the artifact from the first available source.
        for client in self.retrieval_clients:
            try:
                artifact = client.load_artifact(self.run.id, key)
            except Exception as e:
                LOGGER.info(
                    f"An error occurred while attempting to retrieve artifact: {key}", e
                )
                continue

            if artifact is not None:
                break

        # If the artifact could not be found in any of the configured sources then raise
        # an error.
        # TODO: Consider changing this behavior to return None. The arifact not being
        # found should not be an exceptional case.
        if artifact is None:  # Do we want this behavior or just return None?
            raise ArtifactNotFoundError(
                f"Artifact '{key}' could not be found in the configured locations."
            )

        return artifact

    @staticmethod
    def _create_checkpoint_tag(
        checkpoint_type: CheckpointType, epoch_no: int = None
    ) -> str:
        if checkpoint_type == CheckpointType.EPOCH:
            return f"{checkpoint_type.value}:{epoch_no:04d}"
        else:
            return str(checkpoint_type)
