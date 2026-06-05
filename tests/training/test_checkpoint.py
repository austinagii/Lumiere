import functools
import json
import time
from pathlib import Path

import pytest
import torch

from lumiere.persistence.clients import StorageClient
from lumiere.persistence.errors import StorageError
from lumiere.training.checkpoint import (
    Checkpoint,
    CheckpointStore,
    decode_checkpoint,
)
from lumiere.utils import randomizer


@pytest.fixture
def checkpoint() -> Checkpoint:
    return Checkpoint(
        epoch=10, loss=1.59e-5, random_state=42, model_state=torch.tensor([1, 2, 3])
    )


class TestCheckpoint:
    def test_checkpoint_can_be_instantiated_from_arbitrary_key_value_pairs(self):
        pass

    def test_checkpoint_data_can_be_accessed_using_dot_notation(
        self, checkpoint: Checkpoint
    ):
        assert checkpoint.epoch == 10
        assert checkpoint.loss == 1.59e-5
        assert checkpoint.random_state == 42
        assert torch.equal(checkpoint.model_state, torch.tensor([1, 2, 3]))

    def test_checkpoint_can_be_converted_to_bytes(self, checkpoint: Checkpoint):
        checkpoint_bytes = checkpoint.to_bytes()
        loaded_checkpoint = Checkpoint.from_bytes(checkpoint_bytes)

        assert isinstance(loaded_checkpoint, Checkpoint)
        assert loaded_checkpoint.epoch == checkpoint.epoch
        assert loaded_checkpoint.loss == checkpoint.loss
        assert loaded_checkpoint.random_state == checkpoint.random_state
        assert torch.equal(loaded_checkpoint.model_state, checkpoint.model_state)


# TODO: Move to test utility package.
class MemoryStorageClient:
    """A simple storage client using an in-memory backend."""

    def __init__(self):
        self._storage = {}

    def save(self, path: str | Path, data: bytes, overwrite: bool = False) -> int:
        if not overwrite and path in self._storage:
            raise StorageError(f"Data already exists at '{path}'")
        self._storage[path] = data
        return len(data)

    def load(self, path: str | Path) -> bytes | None:
        return self._storage.get(path)


@pytest.fixture
def storage_client():
    return MemoryStorageClient()


@pytest.fixture
def checkpoint_store(storage_client: StorageClient):
    return CheckpointStore(storage_client)


class TestCheckpointStore:
    def test_insert_saves_checkpoint_to_storage_location(
        self,
        checkpoint: Checkpoint,
        checkpoint_store: CheckpointStore,
        storage_client: StorageClient,
    ):
        run_name = randomizer.random_name()

        checkpoint_store.add(run_name, checkpoint)

        checkpoint_bytes = storage_client.load(
            f"runs/{run_name}/artifacts/checkpoints/{checkpoint.id}.ckpt"
        )
        loaded_checkpoint = Checkpoint.from_bytes(checkpoint_bytes)
        assert isinstance(loaded_checkpoint, Checkpoint)
        assert loaded_checkpoint.epoch == checkpoint.epoch
        assert loaded_checkpoint.loss == checkpoint.loss
        assert loaded_checkpoint.random_state == checkpoint.random_state
        assert torch.equal(loaded_checkpoint.model_state, checkpoint.model_state)

    def test_insert_adds_checkpoint_to_index(
        self,
        checkpoint: Checkpoint,
        checkpoint_store: CheckpointStore,
        storage_client: StorageClient,
    ):
        run_name = randomizer.random_name()

        checkpoint_index_bytes = storage_client.load(
            f"runs/{run_name}/artifacts/checkpoints/index.json"
        )
        assert checkpoint_index_bytes is None

        checkpoint_store.get(run_name, checkpoint)
        checkpoint_index_bytes = storage_client.load(
            f"runs/{run_name}/artifacts/checkpoints/index.json"
        )
        assert checkpoint_index_bytes

        checkpoint_index = json.loads(
            checkpoint_index_bytes, object_hook=decode_checkpoint
        )

        epoch_checkpoint = checkpoint_index.get("epoch:10")
        assert epoch_checkpoint and _equal_checkpoints(checkpoint, epoch_checkpoint)

        latest_checkpoint = checkpoint_index.get("latest")
        assert latest_checkpoint and _equal_checkpoints(checkpoint, latest_checkpoint)

        best_checkpoint = checkpoint_index.get("best")
        assert best_checkpoint and _equal_checkpoints(checkpoint, best_checkpoint)

    @pytest.mark.slow
    def test_insert_updates_index_after_a_checkpoint_is_added(
        self,
        checkpoint: Checkpoint,
        checkpoint_store: CheckpointStore,
        storage_client: StorageClient,
    ):
        run_name = randomizer.random_name()
        checkpoint_index_bytes = storage_client.load(
            f"runs/{run_name}/artifacts/checkpoints/index.json"
        )
        assert checkpoint_index_bytes is None

        load_index = functools.partial(
            _load_index, storage_client=storage_client, run_name=run_name
        )

        current_time = time.time_ns()
        checkpoint_1 = Checkpoint(epoch=1, loss=1.0, created_at=current_time + 100_000)
        checkpoint_2 = Checkpoint(epoch=2, loss=0.5, created_at=current_time)
        checkpoint_3 = Checkpoint(epoch=3, loss=0.8, created_at=current_time + 200_000)

        checkpoint_store.get(run_name, checkpoint_1)
        checkpoint_index = load_index()
        assert _equal_checkpoints(checkpoint_index.get("epoch:1"), checkpoint_1)
        assert _equal_checkpoints(checkpoint_index.get("latest"), checkpoint_1)
        assert _equal_checkpoints(checkpoint_index.get("best"), checkpoint_1)

        checkpoint_store.get(run_name, checkpoint_2)
        checkpoint_index = load_index()
        assert _equal_checkpoints(checkpoint_index.get("epoch:2"), checkpoint_2)
        assert _equal_checkpoints(checkpoint_index.get("latest"), checkpoint_1)
        assert _equal_checkpoints(checkpoint_index.get("best"), checkpoint_2)

        checkpoint_store.get(run_name, checkpoint_3)
        checkpoint_index = load_index()
        assert _equal_checkpoints(checkpoint_index.get("epoch:3"), checkpoint_3)
        assert _equal_checkpoints(checkpoint_index.get("latest"), checkpoint_3)
        assert _equal_checkpoints(checkpoint_index.get("best"), checkpoint_2)

    def test_get_retrieves_checkpoint_from_storage(
        self,
        checkpoint_store: CheckpointStore,
        storage_client: StorageClient,
    ):
        current_time = time.time_ns()
        run_name = randomizer.random_name()
        checkpoint_1 = Checkpoint(
            epoch=1,
            loss=1.0,
            created_at=current_time + 100_000,
            model_state=torch.randn(3),
        )
        checkpoint_2 = Checkpoint(
            epoch=2,
            loss=0.5,
            created_at=current_time,
            model_state=torch.randn(3),
        )
        checkpoint_3 = Checkpoint(
            epoch=3,
            loss=0.8,
            created_at=current_time + 200_000,
            model_state=torch.randn(3),
        )

        checkpoint_store.add(run_name, checkpoint_1)
        checkpoint_store.add(run_name, checkpoint_2)
        checkpoint_store.add(run_name, checkpoint_3)

        def _assert_checkpoints_equal(a, b):
            assert a.id == b.id
            assert a.epoch == b.epoch
            assert a.loss == b.loss
            assert a.created_at == b.created_at
            assert torch.allclose(a.model_state, b.model_state)

        # fmt: off
        _assert_checkpoints_equal(checkpoint_store.get(run_name, "latest"), checkpoint_3)   # NOQA: E501
        _assert_checkpoints_equal(checkpoint_store.get(run_name, "best"), checkpoint_2)     # NOQA: E501
        _assert_checkpoints_equal(checkpoint_store.get(run_name, "epoch:1"), checkpoint_1)  # NOQA: E501
        _assert_checkpoints_equal(checkpoint_store.get(run_name, "epoch:2"), checkpoint_2)  # NOQA: E501
        _assert_checkpoints_equal(checkpoint_store.get(run_name, "epoch:3"), checkpoint_3)  # NOQA: E501
        # fmt: on


def _load_index(storage_client, run_name):
    checkpoint_index_bytes = storage_client.load(
        f"runs/{run_name}/artifacts/checkpoints/index.json"
    )
    if checkpoint_index_bytes is None:
        return None
    checkpoint_index = json.loads(checkpoint_index_bytes, object_hook=decode_checkpoint)
    return checkpoint_index


def _equal_checkpoints(a, b):
    return (
        a.id == b.id
        and a.epoch == b.epoch
        and a.loss == b.loss
        and a.created_at == b.created_at
    )
