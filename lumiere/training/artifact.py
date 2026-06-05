import logging
from pathlib import Path

from lumiere.persistence.clients import StorageClient
from lumiere.persistence.errors import StorageError


ARTIFACT_PATH_TEMPLATE = "runs/{run_name}/artifacts/{artifact_name}"

logger = logging.getLogger(__name__)


class ArtifactStore:
    def __init__(self, client: StorageClient):
        self.client = client

    def add(
        self,
        run_name: str,
        artifact_name: Path,
        artifact: bytes,
    ) -> None:
        """Save an arbitrary artifact for the current training run.

        The specified artifact can be an arbitrary python object.

        Args:
            name: The name that identifies the artifact being stored.
            artifact: The artifact to be stored.
            overwrite: Whether to overwrite any previous artifacts with the same name.
                Defaults to False.
            raise_exception: Whether to raise an exception if an error occurs while
                storing the artifact in any of the configured storage locations.
                Defaults to False.

        Raises:
            StorageError: If an error occurred while storing the artifact and
            `raise_exception` is True.
        """
        artifact_path = ARTIFACT_PATH_TEMPLATE.format(
            run_name=run_name, artifact_name=artifact_name
        )

        try:
            num_bytes_written = self.client.save(
                artifact_path, artifact, overwrite=True
            )
        except Exception as e:
            raise StorageError("An error occurred while storing the artifact.") from e

        if num_bytes_written != len(artifact):
            raise StorageError(
                f"Failed to save full data for artifact '{artifact_name}'."
            )

    def get(self, run_name: str, artifact_name: str) -> bytes | None:
        """Load an artifact for the current training run.

        Args:
            name: The name of the object to be loaded.

        Returns:
            bytes | None: The matching artifact.

        """
        artifact_path = ARTIFACT_PATH_TEMPLATE.format(
            run_name=run_name, artifact_name=artifact_name
        )

        try:
            artifact = self.client.load(artifact_path)
        except Exception as e:
            raise StorageError(
                "An error occurred while attempting to retrieve artifact:"
                + " '{artifact_name}'.",
            ) from e

        return artifact
