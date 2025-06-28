import os
from contextlib import contextmanager
from pathlib import Path

from azure.storage.blob import BlobServiceClient

from lumiere.persistence.errors import PersistenceError


class LocalStorageClient:
    def __init__(self, base_dir: Path = Path(".")):
        self.base_dir = base_dir

    def exists(self, artifact_path: Path) -> bool:
        return (self.base_dir / artifact_path).exists()

    def store(self, artifact_path: Path, artifact: bytes) -> None:
        artifact_path = self.base_dir / artifact_path
        # Create the checkpoint directory if it does not already exist.
        if not artifact_path.parent.exists():
            try:
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise PersistenceError(
                    f"Failed to create artifact directory '{artifact_path.parent}'",
                    e,
                )

        mode = "w" if artifact_path.exists() else "x"
        try:
            with open(artifact_path, f"{mode}b") as f:
                f.write(artifact)
        except Exception as e:
            raise PersistenceError(
                f"An error occurred while saving the artifact to '{artifact_path}'",
                e,
            )

    def retrieve(self, artifact_path: Path) -> bytes:
        """Downloads the artifact from the local device"""
        with open(self.base_dir / artifact_path, "rb") as f:
            return f.read()


@contextmanager
def disable_tokenizer_parallelism():
    """Context manager to temporarily disable tokenizers parallelism.

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


class RemoteStorageClient:
    def __init__(self, blob_service_client: BlobServiceClient, container_name: str):
        self.blob_service_client = blob_service_client
        self.container_name = container_name

    def exists(self, artifact_path: Path) -> bool:
        with disable_tokenizer_parallelism():
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=str(artifact_path)
            )
            try:
                return blob_client.exists()
            except Exception as e:
                raise PersistenceError(
                    "An error occurred while checking if the artifact exists in blob storage",
                    e,
                )

    def store(self, artifact_path: Path, artifact: bytes) -> None:
        """Uploads the artifact to the remote device"""
        with disable_tokenizer_parallelism():
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=str(artifact_path)
            )

            try:
                blob_client.upload_blob(artifact, overwrite=True)
            except Exception as e:
                raise PersistenceError(
                    "An error occurred while syncing checkpoint to blob storage", e
                )

    def retrieve(self, artifact_path: Path) -> bytes:
        """Downloads the artifact from the remote device."""
        with disable_tokenizer_parallelism():
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=str(artifact_path)
            )

            try:
                if not blob_client.exists():
                    raise PersistenceError("Checkpoint could not be found in blob")
                artifact = blob_client.download_blob().readall()
            except Exception as e:
                raise PersistenceError(
                    "An error occurred while downloading the artifact from blob storage",
                    e,
                )

            return artifact
