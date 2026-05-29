"""Classes for managing training runs."""

import dataclasses
import json
import logging
import time
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path

import yaml

from lumiere.persistence.clients import StorageClient
from lumiere.utils import randomizer

from .config import Config


RUN_META_PATH_TEMPLATE = "runs/{run_name}/meta.json"
RUN_CONFIG_PATH_TEMPLATE = "runs/{run_name}/config.yaml"
RUN_ARTIFACT_PATH_TEMPLATE = "runs/{run_name}/artifacts/{artifact_name}"

LOGGER = logging.getLogger(__name__)


class RunStatus(StrEnum):
    """The status of a training run.

    Attributes:
        PENDING: Run has been created but not yet started.
        RUNNING: Run is actively executing.
        SUSPENDED: Run has been paused and can be resumed.
        STOPPED: Run was manually terminated before completion.
        COMPLETED: Run finished successfully.
        ERROR: Run terminated due to an unrecoverable failure.
    """

    PENDING = auto()
    RUNNING = auto()
    SUSPENDED = auto()
    STOPPED = auto()
    COMPLETED = auto()
    ERROR = auto()


@dataclass
class Run:
    """A training run for a given model.

    Attributes:
        id: The unique identifier for this run.
        name: The unique name for this run.
        config: The configuration used for this run.
        created_at: The instant this run was created (in nano since epoch)
        updated_at: The instant this run was last updated (in nano since epoch)
        current_epoch: The current epoch of this run.
        current_step: The current step of the training run.
        current_loss: The current validation loss.

    """

    id: str
    name: str
    status: RunStatus
    config: Config
    created_at: int
    updated_at: int | None = None
    current_epoch: int = 0
    current_step: int = 0
    current_loss: float = 0.0

    def __init__(self, config: Config, name: str | None = None):
        """Initialize a training run.

        Args:
            config: The configuration for the training run.
            name: An optional name for the training run. If none is provided
                one will be automatically generated. Defaults to `None`.
        """
        self.id = randomizer.random_id()
        self.name = randomizer.random_name() if name is None else name
        self.status = RunStatus.PENDING
        self.config = config
        self.created_at = time.time_ns()

    def to_dict(self):
        """Return a dict containing the properties."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d) -> "Run":
        """Create a `Run` from a dict."""
        # TODO: Add type checking for run values from config.
        run = cls(d["config"], name=d["name"])
        run.id = d["id"]
        run.status = RunStatus(d["status"])
        run.config = d["config"]
        run.created_at = d["created_at"]
        run.updated_at = d.get("updated_at")
        run.current_epoch = d.get("current_epoch", 0)
        run.current_step = d.get("current_step", 0)
        run.current_loss = d.get("current_loss", 0.0)
        return run


class RunRepository:
    """Stores and retrieves training runs.

    Training runs are stored as two separate artifacts:
        - meta.json: The properties of the training run.
        - config.yaml: The configuration of the training run.

    Attributes:
        client: The storage client for the desired storage location.

    """

    def __init__(self, client: StorageClient):
        """Initialize a run repository.

        Args:
            client: The client responsible for storing run data.

        """
        self.client = client

    def insert(self, run: Run) -> None:
        """Save a training run.

        Args:
            run: The run to be saved.

        Raises:
            StorageError: If an error occurred while saving the run to the storage
                location.

        """
        run_bytes = _run_to_json(run)
        self.client.save(f"runs/{run.name}/meta.json", run_bytes)

        run_config = dict(run.config)
        config_bytes = bytes(yaml.dump(run_config), "utf-8")
        self.client.save(f"runs/{run.name}/config.yaml", config_bytes)

    def get(self, run_name: str) -> Run | None:
        """Resume the specified training run.

        Args:
            run_name: The name of the training run.

        Returns:
            Run: The matching training run or `None` if not found.

        Raises:
            StorageError: If an error occurred while retrieving the run from the storage
                location.

        """
        run_bytes = self.client.load(RUN_META_PATH_TEMPLATE.format(run_name=run_name))
        if run_bytes is None:
            return None

        run_dict = json.loads(run_bytes)

        run_config_bytes = self.client.load(
            RUN_CONFIG_PATH_TEMPLATE.format(run_name=run_name)
        )
        if run_config_bytes is None:
            # Every run should have a config associated with it.
            raise RuntimeError(f"Could not retrieve config for run '{run_name}'.")

        run_config_dict = yaml.safe_load(run_config_bytes)
        run_dict["config"] = Config(run_config_dict)

        return Run.from_dict(run_dict)

    # TODO: Add docstring.
    def update(self, run: Run):
        run_bytes = _run_to_json(run)
        self.client.save(
            RUN_META_PATH_TEMPLATE.format(run_name=run.name),
            run_bytes,
            overwrite=True,
        )
        return run


def _run_to_json(run: Run) -> bytes:
    """Convert a run to byte encoded JSON."""
    run_dict = run.to_dict()
    run_dict = {k: v for k, v in run_dict.items() if k != "config"}
    return bytes(json.dumps(run_dict, indent=2), "utf-8")


class RunArtifactRepository:
    """Stores and retrieves artifacts generated during training runs."""

    def __init__(self, client: StorageClient):
        """Initialize an artifact repository.

        Args:
            client: The client for the desired storage destination.
        """
        self.client = client

    def insert(self, run_name: str, artifact_name: str | Path, artifact: bytes) -> None:
        """Save an artifact.

        Args:
            run_name: The name of the training run which generated the artifact.
            artifact_name: The name of the artifact.
            artifact: The artifact to be stored (as bytes).

        Raises:
            StorageError: If an error occurred while storing the artifact.

        """
        artifact_path = RUN_ARTIFACT_PATH_TEMPLATE.format(
            run_name=run_name, artifact_name=artifact_name
        )
        self.client.save(artifact_path, artifact)

    def get(self, run_name: str, artifact_name: str) -> bytes | None:
        """Load an artifact.

        Args:
            run_name: The name of the training run which generated the artifact.
            artifact_name: The name of the artifact.

        Returns:
            bytes | None: The artifact as bytes or `None` if the artifact could not be
                found.

        """
        artifact_path = RUN_ARTIFACT_PATH_TEMPLATE.format(
            run_name=run_name, artifact_name=artifact_name
        )
        return self.client.load(artifact_path)
