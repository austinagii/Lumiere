import json

import pytest

from lumiere.testing.storage_clients import MemoryStorageClient
from lumiere.training.event import EventStore
from lumiere.utils import randomizer


@pytest.fixture
def storage_client():
    return MemoryStorageClient()


@pytest.fixture
def event_store(storage_client):
    return EventStore(storage_client)


class TestEventStore:
    def test_add_inserts_event_into_event_log(self, event_store, storage_client):
        run_name = randomizer.random_name()
        event = {"era": "primordial", "action": "bigbang", "impact": "kickstart"}

        event_store.add(run_name, event)

        event_log_bytes = storage_client.load(f"runs/{run_name}/events.json")
        event_log = json.loads(event_log_bytes)

        assert len(event_log) == 1
        logged_event = event_log[0]
        assert logged_event["era"] == event["era"]
        assert logged_event["action"] == event["action"]
        assert logged_event["impact"] == event["impact"]
        assert len(logged_event.keys()) == 3

    def test_insert_raises_if_error_occurs_while_saving_to_storage(self, mocker):
        error_cls = RuntimeError
        mocker.patch.object(MemoryStorageClient, "save", side_effect=error_cls)
        store = EventStore(MemoryStorageClient())

        with pytest.raises(error_cls):
            store.add(randomizer.random_name(), {})

    def test_list_loads_all_events_from_storage(self, storage_client, event_store):
        run_name = randomizer.random_name()
        event_log = [
            {"id": 1, "event": "start"},
            {"id": 2, "event": "epoch"},
            {"id": 3, "event": "end"},
        ]

        event_log_bytes = bytes(json.dumps(event_log, indent=2), "utf-8")
        storage_client.save(f"runs/{run_name}/events.json", event_log_bytes)
        events = event_store.list(run_name)

        assert events[0]["event"] == "start"
        assert events[1]["event"] == "epoch"
        assert events[2]["event"] == "end"

    def test_list_returns_none_if_run_does_not_exist(self, event_store, storage_client):
        assert len(storage_client._storage) == 0
        assert event_store.list(randomizer.random_name()) is None

    def test_list_propagates_errors_that_occur_while_loading_events_from_store(
        self, mocker
    ):
        error_cls = RuntimeError
        mocker.patch.object(MemoryStorageClient, "load", side_effect=error_cls())

        client = MemoryStorageClient()
        store = EventStore(client)

        with pytest.raises(error_cls):
            store.list(randomizer.random_name())
