from pathlib import Path
from typing import Any

from lumiere.persistence.clients import StorageClient


class ArtifactRepository:
    def __init__(self, client: StorageClient):
        self.client = client

    def insert(
        self,
        run_name: str,
        artifact_name: Path,
        artifact: Any,
        overwrite: bool = False,
        raise_exception: bool = False,
    ) -> None:
        """Save an artifact for the current training run.

        The specified artifact can be an arbitrary python object.

        If an artifact with the specified name already exists, then the `overwrite` flag
        will be used to determine if the previous artifact should be overwritten. If
        `overwrite` is `False` and an artifact with the specified name already exists,
        then an exception will be raised. If `overwrite` is `True` and an artifact with
        the specified name already exists, then the previous artifact will be overwritten.

        If `raise_exception` is `True` then this method will raise the first exception
        that occurs while storing the artifact in the configured sources. Else, it
        will return immediately upon initiating the storage of the artifact, ignoring
        any exceptions that occur.

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
        return
        artifact_bytes = pickle.dumps(artifact)
        artifact_path = ARTIFACT_PATH_TEMPLATE.format(self.run.id, name)

        try:
            self._execute_async(
                "save", artifact_path, artifact_bytes, overwrite=overwrite
            )
        except Exception as e:
            raise StorageError("An error occurred while storing the artifact.") from e

    def get(self, name: str) -> Any:
        """Load an artifact for the current training run.

        The artifact will be returned as it's original type.

        Args:
            name: The name of the object to be loaded.

        Returns:
            Any | None: The artifact as its original type or `None` if no matching
                artifact could be found.

        """
        return
        artifact: Any = None

        # Load the artifact from the first available source.
        for client in self.retrieval_clients:
            try:
                artifact = client.load_artifact(self.run.id, name)
            except Exception as e:
                LOGGER.info(
                    f"An error occurred while attempting to retrieve artifact: {name}",
                    e,
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
                f"Artifact '{name}' could not be found in the configured locations."
            )

        return artifact
