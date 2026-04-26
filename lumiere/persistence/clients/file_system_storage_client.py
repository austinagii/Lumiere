"""A client for storing and retrieving data from the local file system.

Example:
    ```python
    from lumiere.training.checkpoint import Checkpoint

    checkpoint = Checkpoint(epoch=10, train_loss=0.015, weights=torch.randn(1, 5, 10))
    checkpoint_bytes = bytes(checkpoint)

    fs_client = FileSystemStorageClient("/tmp/runs/umvc3")
    fs_client.save("checkpoints/epoch_001.pt", checkpoint_bytes)
    ```
"""

from pathlib import Path

from lumiere.persistence.errors import StorageError


DEFAULT_BASE_DIR = Path(".")


class FileSystemStorageClient:
    """Client for storing and retrieving data from the local file system."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        """Initializes a new `FileSystemStorageClient`.

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

    def save(self, path: str | Path, data: bytes, overwrite: bool = False):
        """Write data to a file.

        If a file does not exist at the specified path, then one will be created. If one
        does exist, then an error will be raised to prevent overwriting the contents of
        that file. This behaviour can be disabled by setting the `overwrite` flag to
        `True`.

        Args:
            path: The destination file path.
            data: The data to be written to the file.
            overwrite: Whether to overwrite the contents of an existing file at `path`.

        Raises:
            StorageError: If any of the following occur:
                - A file exists at the specified path and `overwrite` is `False`.
                - All bytes of the data could not be written to the file system.
                - An error occurred while writing the data to the file system.
        """
        if not isinstance(path, Path):
            try:
                path = Path(path)
            except Exception as e:
                raise ValueError(f"Invalid path: '{path}'") from e

        assert len(data) > 0, "No data provided to be written"

        fullpath = self._base_dir / path
        if not overwrite and fullpath.exists():
            raise StorageError(f"Attempting overwrite data at '{path}'.")

        try:
            fullpath.parent.mkdir(parents=True, exist_ok=True)
            return fullpath.write_bytes(data)
        except Exception as e:
            raise StorageError("An error occurred while saving the data", e) from e

    def load(self, path: str | Path) -> bytes | None:
        """Read data from a file.

        Args:
            path: The path to the file to be read.

        Returns:
            The data read from the file (as bytes) or `None` if no file could be found
            at the specified path.

        Raises:
            TypeError: If `path` is not a string or path-like object.
            StorageError: If an error occurred while reading data from the file.

        """
        if isinstance(path, str):
            try:
                path = Path(path)
            except Exception as e:
                raise ValueError(f"Invalid path: '{path}'") from e

        if not isinstance(path, Path):
            raise TypeError(
                "Expected 'path' to be either a string or path-like object, "
                + f"got '{type(path).__name__}'"
            )

        path = self.base_dir / path
        if not path.exists():
            return None

        try:
            return path.read_bytes()
        except Exception as e:
            raise StorageError(
                f"An error occurred while reading file '{path}'.", e
            ) from e

    @property
    def base_dir(self):
        """The directory where artifacts managed by this client are stored."""
        return self._base_dir
