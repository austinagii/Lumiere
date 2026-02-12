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
    storage.init_run("run-123", {"max_epochs": 10})
    ```
"""

import logging
import os
import pickle
from contextlib import contextmanager
from typing import Any

import yaml
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient

from lumiere.persistence.errors import (
    ArtifactNotFoundError,
    CheckpointNotFoundError,
    RunNotFoundError,
    StorageError,
)
from lumiere.persistence.storage_client import StorageClient


# Disable Azure blob storage logging
logging.getLogger("azure.storage.blob").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)

# TODO: Rename these constants. These arent really 'path' templates but instead blob name
# templates. It's just that the storage browser in azure visualizes them like paths in a
# filesyste.
RUN_BASE_PATH_TEMPLATE = "runs/{run_id}"
RUN_CONFIG_PATH_TEMPLATE = "runs/{run_id}/config.yaml"
RUN_ARTIFACT_PATH_TEMPLATE = "runs/{run_id}/{key}"
RUN_CHECKPOINT_PATH_TEMPLATE = "runs/{run_id}/checkpoints/{checkpoint_tag}.pt"

# Disable Azure blob storage logging
logging.getLogger("azure.storage.blob").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)


class AzureBlobStorageClient(StorageClient):
    """Azure Blob Storage implementation of StorageClient.

    Artifacts managed by this client are stored in Azure Blob Storage as blobs using
    this structure:
        runs/{run_id}/config.yaml           - Training configuration (YAML)
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

    def init_run(self, run_id: str, train_config: dict[Any, Any]) -> None:
        """An implementation of :meth:`StorageClient.init_run`."""
        config_blob_path = RUN_CONFIG_PATH_TEMPLATE.format(run_id=run_id)

        self._upload_blob(
            self.blob_service_client,
            self.container_name,
            config_blob_path,
            yaml.dump(train_config),
        )

    def resume_run(
        self, run_id: str, checkpoint_tag: str
    ) -> tuple[dict[Any, Any], bytes]:
        """An implementation of :meth:`StorageClient.resume_run`."""
        run_config_blob_name = RUN_CONFIG_PATH_TEMPLATE.format(run_id=run_id)
        try:
            run_config_data = self._download_blob(
                self.blob_service_client, self.container_name, run_config_blob_name
            )
        except ArtifactNotFoundError:
            raise RunNotFoundError("The run config could not be found in Azure Blob.")

        run_config = yaml.safe_load(run_config_data)

        checkpoint_blob_name = RUN_CHECKPOINT_PATH_TEMPLATE.format(
            run_id=run_id, checkpoint_tag=checkpoint_tag
        )
        try:
            checkpoint = self._download_blob(
                self.blob_service_client, self.container_name, checkpoint_blob_name
            )
        except ArtifactNotFoundError:
            raise CheckpointNotFoundError(
                "The checkpoint could not be found in Azure Blob."
            )

        return run_config, checkpoint

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

    def load_artifact(self, run_id: str, key: str) -> Any:
        """Load an artifact for the given run from Azure Blob Storage.

        Implements :meth:`StorageClient.load_artifact`.
        """
        artifact = None

        try:
            artifact_data = self._download_blob(
                self.blob_service_client,
                self.container_name,
                RUN_ARTIFACT_PATH_TEMPLATE.format(run_id=run_id, key=key),
            )
            artifact = pickle.loads(artifact_data)
        except ArtifactNotFoundError:
            pass
            # Do nothing if the artifact isnt found.
            # TODO: Consider just logging an info message.

        return artifact

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

    @staticmethod
    def _download_blob(
        blob_service_client: BlobServiceClient, container_name: str, blob_name: str
    ) -> bytes:
        with blob_service_client.get_blob_client(
            container=container_name, blob=blob_name
        ) as blob_client:
            try:
                return blob_client.download_blob().readall()
            except ResourceNotFoundError:
                raise ArtifactNotFoundError(
                    "Artifact could not be found in blob storage."
                )
            except Exception as e:
                raise StorageError(
                    "An error occurred while downloading the blob from blob storage",
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
