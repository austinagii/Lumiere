from typing import Any, Protocol

from lumiere.training.checkpoint import Checkpoint


class StorageClient(Protocol):
    """Stores and retrieves training artifacts in a storage location.

    This class provides an interface for classes inteded to be used to store and
    retrieve artifacts (e.g. training configurations, checkpints, etc...) from a
    configured storage location. (e.g. Amazon S3 bucket).
    """

    def init_run(self, run_id: str, run_config: bytes) -> None:
        """Intialize a new training run with the specifed configuration.

        Args:
            run_id: The unique identifier of the run to be initialized.
            run_config: The training configuration for this run.

        Raises:
            StorageError: If an error occurred while attempting to create the training
                in the storage location.
        """
        ...

    def resume_run(
        self, run_id: str, checkpoint_tag: str = "latest"
    ) -> tuple[dict[Any, Any], Checkpoint | None]:
        """Load a previous training run.

        If a checkpoint tag is specified then the checkpoint with the matching tag is
        returned. If no checkpoint tag is specified then the checkpoint tagged as
        'latest' will be returned instead.

        If no checkpoint tag is specified and there are also no previously saved
        checkpoints then none will be returned.

        Args:
            run_id: The unique identifier of the run to be resumed.
            checkpoint_tag: The tag of the checkpoing where the run should be resumed
                from. Defaults to None.

        Returns:
            Tuple[dict[Any, Any], Checkpoint]:
                A tuple containing:
                    - The training configuration used for the specified training run.
                    - The checkpoint from which the training run should be resumed.

        Raises:
            RunNotFoundError: If the training run or its configuration cannot be found.
            CheckpointNotFoundError: If the specified checkpoint could not be found.
        """
        ...

    def save_checkpoint(
        self, run_id: str, checkpoint_tag: str, checkpoint: bytes
    ) -> None:
        """Save the specified checkpoint of the given training run.

        Args:
            run_id: The unique identifier of the training run.
            checkpoint_tag: The tag to be used for the specified checkpoint.
            checkpoint: The checkpoint to be saved (in bytes).

        Raises:
            StorageError: If an error occurred while attempting to save the checkpoint.
        """
        ...

    def load_checkpoint(self, run_id: str, checkpoint_tag: str) -> bytes:
        """Load the specified checkpoint.

        Args:
            run_id: The unique identifier of the training run for which the checkpoint
                was saved.
            checkpoint_tag: The tag of the checkpoint to be loaded.

        Returns:
            bytes: The matching checkpoint as a raw bytestream.

        Raises:
            StorageError: If an error occurred while attempting to load the checkpoint.
            CheckpointNotFoundError: If the specified checkpoint could not be found.
        """
        ...

    def save_artifact(
        self, run_id: str, key: str, artifact: Any, overwrite: bool = False
    ) -> None:
        """Save an artifact for the specified training run.

        The specified artifact can be an arbitrary python object. This object will be
        serialized and stored in the storage location.

        If an artifact with the specified key already exists, then the `overwrite` flag
        will be used to determine if the previous artifact should be overwritten.

        Args:
            run_id: The unique identifier of the training run.
            key: The key that identifies the artifact being stored.
            artifact: The artifact to be stored.
            overwrite: Whether to overwrite any previous artifacts with the same key.
                Defaults to False.

        Raises:
            StorageError: If an error occurred while storing the artifact. The specific
                error will be nested.
            KeyError: If an artifact with the specified key already exists and
                overwriting is not allowed.
        """
        ...

    def load_artifact(self, run_id: str, key: str) -> Any | None:
        """Load an artifact for the specified training run.

        The object returned will be equivalent but not identical to the original object 
        stored. i.e. the values of the properties of both objects will be the same but 
        the ids of the objects and their properties may not be.

        Args:
            run_id: The unique identifier of the training run.
            key: The key of the object to be loaded.

        Returns:
            Any | None: The artifact as its original type or `None` if no matching
                artifact could be found.
        """
        ...

