from typing import Any

from lumiere.persistence.clients import StorageClient


class EventRepository:
    def __init__(self, client: StorageClient):
        self.client = client

    def insert(self, event: dict[str, Any]):
        return
