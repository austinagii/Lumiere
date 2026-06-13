from pathlib import Path

import pytest

from lumiere.persistence.errors import StorageError
from lumiere.training.artifact import ArtifactStore
from lumiere.training.checkpoint import CheckpointStore
from lumiere.training.config import Config
from lumiere.training.event import EventStore
from lumiere.training.orchestrator import TrainingOrchestrator
from lumiere.training.run import RunStatus, RunStore


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
def run_store(storage_client):
    return RunStore(storage_client)


@pytest.fixture
def checkpoint_store(storage_client):
    return CheckpointStore(storage_client)


@pytest.fixture
def artifact_store(storage_client):
    return ArtifactStore(storage_client)


@pytest.fixture
def event_store(storage_client):
    return EventStore(storage_client)


@pytest.fixture
def config():
    config_yaml = """
        model:
          type: transformer
          vocab_size: 256
          context_size: 64
          embedding_size: 128
          num_blocks: 1
          normalization:
            type: rms
            normalized_shape: 128
          embedding:
            type: sinusoidal
            padding_id: 2
          block:
            type: standard
            attention:
              type: multihead
              num_heads: 4
              d_key: 32
              d_value: 32
            feedforward:
              type: linear
              d_ff: 256
            normalization:
              type: rms
              normalized_shape: 128
            dropout: 0.1
            pre_norm: True
            post_norm: False

        data:
          datasets:
            - name: wikitext
              split: "1:1:1"

        tokenizer:
          name: bpe
          vocab_size: 256
          min_frequency: 2

        pipeline:
          name: text
          tokenizer: "@tokenizer"
          batch_size: 32
          context_size: 64
          pad_id: 2
          sliding_window_size: 8
          preprocessors:
            - name: autoregressive
              device: cpu

        optimizer:
          name: adamw
          lr: 0.0003

        scheduler:
          name: cosine-annealing
          max_epochs: 1000
          epoch_steps: 1700
          warmup_steps: 1700

        training:
          max_epochs: 3
          stopping_threshold: 0.0001
          gradient_clip_norm: 1.0
    """
    return Config.from_yaml(config_yaml)


@pytest.fixture
def orchestrator(run_store, checkpoint_store, artifact_store, event_store):
    return TrainingOrchestrator(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        artifact_store=artifact_store,
        event_store=event_store,
        checkpoint_interval=1,
    )


class TestOrchestrator:
    def test_train_records_training_run(
        self,
        storage_client,
        run_store,
        checkpoint_store,
        artifact_store,
        event_store,
        config,
    ):
        orchestrator = TrainingOrchestrator(
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            artifact_store=artifact_store,
            event_store=event_store,
            checkpoint_interval=1,
        )

        run = orchestrator.train(config=config)

        saved_run = run_store.get(run.name)
        assert saved_run.status == RunStatus.COMPLETED
        assert saved_run.created_at
        assert isinstance(saved_run.created_at, int)
        assert saved_run.updated_at
        assert isinstance(saved_run.updated_at, int)
        assert saved_run.updated_at > saved_run.created_at
        assert saved_run.current_epoch == 3
        assert saved_run.current_step > 0
        assert saved_run.current_loss
        assert isinstance(saved_run.current_loss, float)

    def test_train_captures_training_checkpoints(
        self, config, orchestrator, checkpoint_store
    ):
        run = orchestrator.train(config)

        assert checkpoint_store.get(run.name, "epoch:1")
        assert checkpoint_store.get(run.name, "epoch:2")
        assert checkpoint_store.get(run.name, "epoch:3")

    def test_train_captures_training_events(self, orchestrator, event_store, config):
        run = orchestrator.train(config=config)

        events = event_store.list(run.name)

        assert len(events) == 3
        for e in events:
            assert "train_loss" in e
            assert "val_loss" in e
            assert "lr" in e
            assert "epoch" in e
            assert "global_step" in e

    def test_train_captures_training_artifacts(
        self, orchestrator, artifact_store, config, storage_client
    ):
        run = orchestrator.train(config=config)

        assert artifact_store.get(run.name, "tokenizer")
