"""Azure Blob Storage client for training artifact management.

This module provides the AzureBlobStorageClient class which implements the StorageClient
interface for storing and retrieving training configurations and model checkpoints in
Azure Blob Storage.

Example:
    ```python
    from azure.storage.blob import BlobServiceClient
    from os import environ

    blob_client = BlobServiceClient.from_connection_string(
        environ["AZURE_BLOB_CONNECTION_STRING"]
    )
    storage = AzureBlobStorageClient(blob_client, "my-container")
    storage.save_artifact("run-123", "config.yaml", {"max_epochs": 10})
    ```
"""

import logging
import os
import pickle
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient

from lumiere.persistence.errors import (
    ArtifactNotFoundError,
    CheckpointNotFoundError,
    StorageError,
)


# Disable Azure blob storage logging
logging.getLogger("azure.storage.blob").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)

logger = logging.getLogger(__file__)


class AzureBlobStorageClient:
    """Azure Blob Storage implementation of StorageClient.

    Artifacts managed by this client are stored in Azure Blob Storage as blobs using
    this structure:
        runs/{run_id}/artifacts/{key}       - Arbitrary artifacts (pickled)
        runs/{run_id}/checkpoints/{tag}.pt  - Model checkpoints (binary)

    Attributes:
        blob_service_client: The Azure client used to interact with blob storage.
        container_name: The name of the blob storage container where artifacts are
            stored and retrieved from.
    """

    def __init__(self, blob_service_client: BlobServiceClient, container_name: str):
        """Initialize a new AzureBlobStorageInstance.

        Args:
            blob_service_client: The blob service client instance that this client
                should use to store and retrieve artifacts.
            container_name: The blob storage container where artifacts should be stored
                and retrieved from.
        """
        # TODO: Add argument validation. e.g. blob service client not closed, container
        # name not empty or None etc...
        self.blob_service_client = blob_service_client
        self.container_name = container_name

    def save(): ...

    def save_checkpoint(
        self, run_id: str, checkpoint_tag: str, checkpoint: bytes
    ) -> None:
        """An implementation of :meth:`StorageClient.save_checkpoint`."""
        checkpoint_path = RUN_CHECKPOINT_PATH_TEMPLATE.format(
            run_id=run_id, checkpoint_tag=checkpoint_tag
        )

        self._upload_blob(
            self.blob_service_client, self.container_name, checkpoint_path, checkpoint
        )

    def load_checkpoint(self, run_id: str, checkpoint_tag: str) -> bytes:
        """An implementation of :meth:`StorageClient.load_checkpoint`."""
        checkpoint_path = RUN_CHECKPOINT_PATH_TEMPLATE.format(
            run_id=run_id, checkpoint_tag=checkpoint_tag
        )

        try:
            checkpoint = self._download_blob(
                self.blob_service_client, self.container_name, checkpoint_path
            )
        except ArtifactNotFoundError:
            raise CheckpointNotFoundError("Checkpoint could not be found in blob")
        return checkpoint

    def save_artifact(
        self, run_id: str, key: str, artifact: Any, overwrite: bool = False
    ) -> None:
        """Save an artifact for the given run to Azure Blob Storage.

        `artifact` can be any arbitrary Python object. It will be serialized using
        the pickle module and stored in azure blob.

        Implements :meth:`StorageClient.save_artifact`.
        """
        try:
            self._upload_blob(
                self.blob_service_client,
                self.container_name,
                RUN_ARTIFACT_PATH_TEMPLATE.format(run_id=run_id, key=key),
                pickle.dumps(artifact),
                overwrite=overwrite,
            )
        except ResourceExistsError as e:
            raise KeyError("An artifact with the specified key already exists.", e)

    def load(self, path: str | Path) -> bytes | None:
        """Load data from a blob.

        Args:
            path: The path to the blob to be loaded.

        Returns:
            The data contained in the blob as bytes.

        Raises:
            StorageError: If an error occurred while reading data from the blob.

        """
        data = None

        with (
            disable_tokenizer_parallelism()
            and self.blob_service_client.get_blob_client(
                container=self.container_name, blob=path
            ) as blob_client
        ):
            try:
                data = blob_client.download_blob().readall()
            except ResourceNotFoundError:
                logger.exception(f"Blob could not be found at '{path}'")
            except Exception as e:
                raise StorageError(
                    "An error occurred while downloading the blob from blob storage"
                ) from e

        return data

    @staticmethod
    def _upload_blob(
        blob_service_client: BlobServiceClient,
        container_name: str,
        blob_name: str,
        blob: str | bytes,
        overwrite=True,
    ) -> None:
        with disable_tokenizer_parallelism() and blob_service_client.get_blob_client(
            container=container_name, blob=blob_name
        ) as blob_client:
            try:
                blob_client.upload_blob(blob, overwrite=overwrite)
            except ResourceExistsError as e:
                raise e
            except Exception as e:
                raise StorageError(
                    "An error occurred while uploading the blob to blob storage",
                    e,
                )


@contextmanager
def disable_tokenizer_parallelism():
    """Temporarily disable parallelism for the huggingface tokenizers library.

    This prevents fork conflicts with the tokenizers library during blob operations.
    """
    original_parallelism = os.environ.get("TOKENIZERS_PARALLELISM")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        yield
    finally:
        # Restore original tokenizer parallelism setting
        if original_parallelism is not None:
            os.environ["TOKENIZERS_PARALLELISM"] = original_parallelism
        else:
            os.environ.pop("TOKENIZERS_PARALLELISM", None)
