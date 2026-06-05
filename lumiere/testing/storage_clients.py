from lumiere.persistence.errors import StorageError


class MemoryStorageClient:
    """A simple storage client using an in-memory backend."""

    def __init__(self) -> None:
        self._storage: dict[str, bytes] = {}

    def save(self, path: str, data: bytes, overwrite: bool = False) -> int:
        if not overwrite and path in self._storage:
            raise StorageError(f"Data already exists at '{path}'")
        self._storage[path] = data
        return len(data)

    def load(self, path: str) -> bytes | None:
        return self._storage.get(path)
