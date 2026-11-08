import pytest

from lumiere.testing.storage_clients import MemoryStorageClient
from lumiere.training.artifact import ArtifactStore
from lumiere.training.checkpoint import CheckpointStore
from lumiere.training.config import Config
from lumiere.training.event import EventStore
from lumiere.training.orchestrator import TrainingArguments, TrainingOrchestrator
from lumiere.training.run import Run, RunStatus, RunStore
from lumiere.utils.device import get_device


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


@pytest.fixture(scope="module")
def device():
    return get_device()


@pytest.fixture
def config():
    config_yaml = """
        architecture:
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
    """
    return Config.from_yaml(config_yaml)


@pytest.fixture
def args(device):
    return TrainingArguments(
        max_epochs=3,
        patience=None,
        stopping_threshold=1e-3,
        gradient_clip_norm=1e-1,
        log_interval=10,
        checkpoint_interval=1,
        device=device,
    )


@pytest.fixture
def orchestrator(run_store, checkpoint_store, artifact_store, event_store):
    return TrainingOrchestrator(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        artifact_store=artifact_store,
        event_store=event_store,
    )


class TestOrchestrator:
    @pytest.mark.slow
    def test_train_stores_record_of_training_run(
        self,
        run_store: RunStore,
        checkpoint_store: CheckpointStore,
        artifact_store: ArtifactStore,
        event_store: EventStore,
        orchestrator: TrainingOrchestrator,
        config: Config,
        args: TrainingArguments,
    ):
        run = orchestrator.start(config, args)

        saved_run = run_store.get(run.name)
        assert saved_run.status == RunStatus.COMPLETED
        assert saved_run.created_at and isinstance(saved_run.created_at, int)
        assert saved_run.updated_at and isinstance(saved_run.updated_at, int)
        assert saved_run.updated_at > saved_run.created_at
        assert saved_run.current_epoch == 3
        assert saved_run.current_step > 0
        assert saved_run.current_loss and isinstance(saved_run.current_loss, float)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "max_epochs,checkpoint_interval",
        [(1, 1), (2, 1), (3, 1), (4, 1), (2, 2), (3, 2), (4, 2)],
    )
    def test_train_stores_checkpoints_at_specified_intervals(
        self,
        max_epochs: int,
        checkpoint_interval: int,
        orchestrator: TrainingOrchestrator,
        checkpoint_store: CheckpointStore,
        config: Config,
        args: TrainingArguments,
    ):
        args.max_epochs = max_epochs
        args.checkpoint_interval = checkpoint_interval

        run = orchestrator.start(config, args)

        checkpoint_epochs = [
            epoch
            for epoch in range(1, max_epochs + 1)
            if epoch % checkpoint_interval == 0
        ]
        for epoch in checkpoint_epochs:
            assert checkpoint_store.get(run.name, f"epoch:{epoch}")

    # TODO: Add test to verify no checkpoints are created if checkpoint_interval > max_epochs.

    def test_train_captures_training_events(
        self, orchestrator, event_store, config, args
    ):
        run = orchestrator.start(config, args)

        events = event_store.list(run.name)

        assert len(events) == 3
        for e in events:
            assert "train_loss" in e
            assert "val_loss" in e
            assert "lr" in e
            assert "epoch" in e
            assert "global_step" in e

    def test_train_captures_training_artifacts(
        self, orchestrator, artifact_store, config, args
    ):
        run = orchestrator.start(config, args)

        assert artifact_store.get(run.name, "tokenizer")


class _StubTrainer:
    """A trainer stand-in that records the run status seen during training."""

    def __init__(self, run_store, run_name, outcome=None):
        self.run_store = run_store
        self.run_name = run_name
        self.outcome = outcome
        self.status_during_training = None

    def train(self):
        self.status_during_training = self.run_store.get(self.run_name).status
        if self.outcome is not None:
            raise self.outcome

        class _Metrics:
            total_time_taken = 1.0

        return _Metrics()


class TestOrchestratorRunStatus:
    @pytest.fixture
    def orchestrator(self, run_store, checkpoint_store, artifact_store, event_store):
        return TrainingOrchestrator(
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            artifact_store=artifact_store,
            event_store=event_store,
        )

    @pytest.fixture
    def run(self, run_store, config):
        run = Run(config, name="status-test")
        run_store.add(run)
        return run

    def test_run_is_running_during_training_and_completed_after(
        self, orchestrator, run_store, run
    ):
        trainer = _StubTrainer(run_store, run.name)

        orchestrator._execute_training_run(run, trainer)

        assert trainer.status_during_training == RunStatus.RUNNING
        assert run_store.get(run.name).status == RunStatus.COMPLETED

    def test_run_is_stopped_when_training_is_interrupted(
        self, orchestrator, run_store, run
    ):
        trainer = _StubTrainer(run_store, run.name, outcome=KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            orchestrator._execute_training_run(run, trainer)

        assert run_store.get(run.name).status == RunStatus.STOPPED

    def test_run_is_error_when_training_fails(self, orchestrator, run_store, run):
        trainer = _StubTrainer(run_store, run.name, outcome=RuntimeError("boom"))

        with pytest.raises(RuntimeError):
            orchestrator._execute_training_run(run, trainer)

        assert run_store.get(run.name).status == RunStatus.ERROR
