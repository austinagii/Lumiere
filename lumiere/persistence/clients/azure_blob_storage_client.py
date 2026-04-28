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
    storage_client = AzureBlobStorageClient(blob_client, "my-container")
    storage_client.save("desired/path/to/artifact", b"some byte data")
    ```
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient

from lumiere.persistence.errors import (
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

    def save(self, path: str | Path, data: bytes, overwrite: bool = False) -> int:
        """Save data as an Azure blob.

        Args:
            path: The desired path of the blob to be created.
            data: The data to be contained within the blob.
            overwrite: Whether to overwrite an existing blob at the specified path.

        Returns:
            int: The number of bytes successfully written to the blob.

        Raise:
            StorageError: If any of the following occur:
                - The specified path corresponds to an existing blob but the overwrite
                  flag is set to `False`
                - An error occurred while uploading the data to blob storage.
        """
        with (
            disable_tokenizer_parallelism()
            and self.blob_service_client.get_blob_client(
                container=self.container_name, blob=path
            ) as blob_client
        ):
            try:
                blob_client.upload_blob(data, overwrite=overwrite)
            except ResourceExistsError as e:
                raise StorageError(
                    f"Attempting to overwrite blob at '{path}' but overwrite flag is"
                    + " not set."
                ) from e
            except Exception as e:
                raise StorageError(
                    "An error occurred while uploading the blob to blob storage",
                ) from e

        return len(data)

    def load(self, path: str | Path) -> bytes | None:
        """Load data from an Azure blob.

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
