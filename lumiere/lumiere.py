"""Provides convenience methods for starting or resuming lumiere training runs."""

from pathlib import Path

from .persistence.clients import FileSystemStorageClient
from .training.artifact import ArtifactStore
from .training.checkpoint import CheckpointStore
from .training.config import Config
from .training.event import EventStore
from .training.orchestrator import TrainingArguments, TrainingOrchestrator
from .training.run import RunStore


def start(
    config: Config,
    args: TrainingArguments | None = None,
    name: str | None = None,
):
    """Start a new training run.

    If either max_epochs or patience are specified (default to None) then training
    will continue indefinitely until manually stopped by the user.

    Args:
        config: A specification of the training run and its components.
        args: The training controls. Defaults to `None`.
        name: The name of the training run. If a name is not provided one will be
            generated. Defaults to `None`.

    """
    # TODO: Consider simplifying initialization of stores.
    storage_client = FileSystemStorageClient(Path(__file__).parent.parent)
    run_store = RunStore(storage_client)
    checkpoint_store = CheckpointStore(storage_client)
    artifact_store = ArtifactStore(storage_client)
    event_store = EventStore(storage_client)

    orchestrator = TrainingOrchestrator(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        artifact_store=artifact_store,
        event_store=event_store,
    )

    return orchestrator.start(config, args=args, run_name=name)


def resume(name: str, tag: str = "latest"):
    """Resume a training run.

    By default, the resumed run will retail the training parameters specified during
    its creation, however, these values can be overridden by specifying the desired
    values during invocation.

    Args:
        name: The name of the run to be resumed.
        tag: The tag of the checkpoint to resume training from.

    """
    storage_client = FileSystemStorageClient(Path(__file__).parent.parent)
    run_store = RunStore(storage_client)
    checkpoint_store = CheckpointStore(storage_client)
    artifact_store = ArtifactStore(storage_client)
    event_store = EventStore(storage_client)

    orchestrator = TrainingOrchestrator(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        artifact_store=artifact_store,
        event_store=event_store,
    )

    orchestrator.resume(name, tag)
