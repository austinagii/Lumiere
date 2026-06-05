import json
from typing import Any

from lumiere.persistence.clients import StorageClient
from lumiere.persistence.errors import StorageError


EVENT_LOG_PATH_TEMPLATE = "runs/{run_name}/events.json"


class EventStore:
    def __init__(self, client: StorageClient):
        self.client = client

    def add(self, run_name: str, event: dict[str, Any]):
        event_log = self._load_event_log(run_name)
        if event_log is None:  # New runs won't have an event log yet.
            event_log = []

        event_log.append(event)

        self._save_event_log(run_name, event_log)

    def _load_event_log(self, run_name: str) -> list[dict]:
        event_log_bytes = self.client.load(
            EVENT_LOG_PATH_TEMPLATE.format(run_name=run_name)
        )
        if event_log_bytes is None:
            return None

        return json.loads(event_log_bytes)

    def _save_event_log(self, run_name: str, event_log: list) -> None:
        event_log_path = EVENT_LOG_PATH_TEMPLATE.format(run_name=run_name)
        event_log_json = json.dumps(event_log, indent=2)
        event_log_bytes = bytes(event_log_json, "utf-8")
        # Overwriting the entire event log seems dangerous considering the possibility
        # that all bytes may not be written and cause an error.
        num_bytes_written = self.client.save(
            event_log_path, event_log_bytes, overwrite=True
        )
        if len(event_log_bytes) != num_bytes_written:
            raise StorageError("Failed to save full event log for run '{run_name}'.")

    def list(self, run_name: str) -> list[dict]:
        return self._load_event_log(run_name)
