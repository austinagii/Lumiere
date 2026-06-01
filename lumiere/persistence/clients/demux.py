from pathlib import Path

from .base import StorageClient


class StorageClientDemux:
    def __init__(self, *storage_clients: StorageClient):
        pass

    def save(self, path: Path | str, data: bytes, overwrite: bool = False):
        pass

    def load(self, path: Path | str):
        pass
