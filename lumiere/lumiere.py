"""Provides convenience methods for starting or resuming lumiere training runs."""

from pathlib import Path

from .persistence.clients import FileSystemStorageClient
from .training.artifact import ArtifactStore
from .training.checkpoint import CheckpointStore
from .training.config import Config
from .training.event import EventStore
from .training.orchestrator import TrainingOrchestrator
from .training.run import RunStore
from .utils.device import get_device


def start(
    model: Config,
    name: str | None = None,
    max_epochs: int | None = None,
    stopping_threshold: float = 1e-3,
    patience: int | None = 5,
    gradient_clip_norm: float | None = 1e-6,
    log_interval: int | None = 50,
    checkpoint_interval: int | None = 3,
    # storage_locations: str = "filesystem:filesystem",
    device: str | None = None,
):
    """Start a new training run.

    If either max_epochs or patience are specified (default to None) then training
    will continue indefinitely until manually stopped by the user.

    Args:
        model: A specification of the model to be trained.
        name: The name of the training run. If a name is not provided one will be
            generated. Defaults to `None`.
        max_epochs: The maximum number epochs to be executed. If none is provided the
            run will execute new epochs until it is stopped.
        stopping_threshold: The minimum improvement to the validation performance for
            this epoch to count as an improvement.
        patience: The maximum number of consecutive epochs allowed without improvement.
            Exceeding this number halts the training run.
        gradient_clip_norm: The maximum normal
        log_interval: The number of steps between each capture or training metrics.
        checkpoint_interval: The number of epochs between saving each checkpoint.
        device: The device training should be executed on.

    """
    storage_client = _init_storage_client()
    run_store = RunStore(storage_client)
    checkpoint_store = CheckpointStore(storage_client)
    artifact_store = ArtifactStore(storage_client)
    event_store = EventStore(storage_client)
    device = get_device() if device is None else device

    orchestrator = TrainingOrchestrator(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        artifact_store=artifact_store,
        event_store=event_store,
        max_epochs=max_epochs,
        patience=patience,
        stopping_threshold=stopping_threshold,
        gradient_clip_norm=gradient_clip_norm,
        log_interval=log_interval,
        checkpoint_interval=checkpoint_interval,
        device=device,
    )

    orchestrator.train(config=model, run_name=name)


def resume(
    name: str,
    checkpoint_tag: str = "latest",
    max_epochs: int | None = None,
    patience: int | None = None,
    stopping_threshold: float = None,
    gradient_clip_norm: float | None = None,
    log_interval: int | None = None,
    checkpoint_interval: int | None = None,
    # storage_locations: str = "filesystem:filesystem",
    device: str | None = None,
):
    storage_client = _init_storage_client()
    run_store = RunStore(storage_client)
    checkpoint_store = CheckpointStore(storage_client)
    artifact_store = ArtifactStore(storage_client)
    event_store = EventStore(storage_client)
    device = get_device() if device is None else device

    orchestrator = TrainingOrchestrator(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        artifact_store=artifact_store,
        event_store=event_store,
        max_epochs=max_epochs,
        patience=patience,
        stopping_threshold=stopping_threshold,
        gradient_clip_norm=gradient_clip_norm,
        log_interval=log_interval,
        checkpoint_interval=checkpoint_interval,
        device=device,
    )

    orchestrator.train(run_name=name, checkpoint_tag=checkpoint_tag)


def _init_storage_client():
    return FileSystemStorageClient(Path(__file__).parent.parent)
