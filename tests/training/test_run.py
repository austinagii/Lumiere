import json
import time
from pathlib import Path

import pytest
import yaml

from lumiere.persistence.errors import StorageError
from lumiere.training import Config
from lumiere.training.run import Run, RunArtifactRepository, RunRepository, RunStatus


@pytest.fixture
def config(scope="module") -> Config:
    return Config(
        {
            "village": "Konoha",
            "hokage": "Naruto",
            "rank": ["Genin", "Chunin", "Jonin", "Kage"],
            "natures": ["Fire", "Wind", "Lightning", "Earth", "Water"],
            "team_7": ["Naruto", "Sasuke", "Sakura"],
            "kurama_tails": 9,
            "sage_mode": True,
        }
    )


@pytest.fixture
def run(config) -> Run:
    return Run(config)


class TestRun:
    """Tests suite for :class:`lumiere.training.run.Run`."""

    def test_to_dict_creates_a_dict_from_the_run(self, run: Run):
        run_dict = run.to_dict()

        assert run.id == run_dict["id"]
        assert run.name == run_dict["name"]
        assert run.status == run_dict["status"]
        assert run.config == run_dict["config"]
        assert run.created_at == run_dict["created_at"]
        assert run.updated_at == run_dict["updated_at"]
        assert run.current_epoch == run_dict["current_epoch"]
        assert run.current_step == run_dict["current_step"]
        assert run.current_loss == run_dict["current_loss"]

    def test_from_dict_creates_a_run_from_a_dict(self, config):
        run_dict = {
            "id": "676-425",
            "name": "tailed-beast-bomb-rasenshuriken",
            "status": "completed",
            "config": config,
            "created_at": 1253235735,
            "updated_at": 2354254782,
            "current_epoch": 10,
            "current_step": 442540,
            "current_loss": 0.00854345,
        }
        run = Run.from_dict(run_dict)

        assert run.id == run_dict["id"]
        assert run.name == run_dict["name"]
        assert run.status == run_dict["status"]
        assert isinstance(run.status, RunStatus)
        assert run.config == run_dict["config"]
        assert isinstance(run.config, Config)
        assert run.created_at == run_dict["created_at"]
        assert run.updated_at == run_dict["updated_at"]
        assert run.current_epoch == run_dict["current_epoch"]
        assert run.current_step == run_dict["current_step"]
        assert run.current_loss == run_dict["current_loss"]

    def test_from_dict_uses_defaults_for_fields_missing_from_dict(self, config):
        run_dict = {
            "id": "676-425",
            "name": "tailed-beast-bomb-rasenshuriken",
            "status": "completed",
            "config": config,
            "created_at": 1253235735,
        }
        run = Run.from_dict(run_dict)

        assert run.id == run_dict["id"]
        assert run.name == run_dict["name"]
        assert run.status == run_dict["status"]
        assert isinstance(run.status, RunStatus)
        assert run.config == run_dict["config"]
        assert isinstance(run.config, Config)
        assert run.created_at == run_dict["created_at"]
        assert run.updated_at is None
        assert run.current_epoch == 0
        assert run.current_step == 0
        assert run.current_loss == 0


class MemoryStorageClient:
    """A storage client using an in-memory backend."""

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
def run_repository(storage_client):
    return RunRepository(storage_client)


class TestRunRepository:
    """Test suite for the :meth:`lumiere.training.run.RunRepository` class."""

    def test_insert_saves_run_metadata_as_json(
        self, run, run_repository, storage_client
    ):
        run_repository.insert(run)

        run_bytes = storage_client.load(f"runs/{run.name}/meta.json")
        assert run_bytes is not None
        run_dict = json.loads(run_bytes)
        assert run_dict["id"] == run.id
        assert run_dict["name"] == run.name
        assert run_dict["status"] == run.status.value
        assert "config" not in run_dict
        assert run_dict["created_at"] == run.created_at
        assert run_dict["updated_at"] == run.updated_at
        assert run_dict["current_epoch"] == run.current_epoch
        assert run_dict["current_step"] == run.current_step
        assert run_dict["current_loss"] == run.current_loss

    def test_insert_saves_run_config_as_yaml(self, run, run_repository, storage_client):
        run_repository.insert(run)

        config_bytes = storage_client.load(f"runs/{run.name}/config.yaml")
        assert config_bytes is not None
        assert yaml.safe_load(config_bytes) == dict(run.config)

    def test_update_overwrites_run_metadata(self, run, run_repository, storage_client):
        run_repository.insert(run)
        run.status = RunStatus.RUNNING
        run.current_epoch = 5
        run.current_step = 12500
        run.current_loss = 0.3217

        run_repository.update(run)

        run_bytes = storage_client.load(f"runs/{run.name}/meta.json")
        run_dict = json.loads(run_bytes)
        assert run_dict["status"] == RunStatus.RUNNING
        assert run_dict["current_epoch"] == 5
        assert run_dict["current_step"] == 12500
        assert run_dict["current_loss"] == 0.3217

    def test_update_does_not_modify_config(self, run, run_repository, storage_client):
        run_repository.insert(run)
        original_config_bytes = storage_client.load(f"runs/{run.name}/config.yaml")

        run.status = RunStatus.RUNNING
        run_repository.update(run)

        assert (
            storage_client.load(f"runs/{run.name}/config.yaml") == original_config_bytes
        )

    def test_get_retrieves_run_by_name(self, storage_client, run_repository):
        run_dict = {
            "id": "kage",
            "name": "gaara",
            "status": "pending",
            "created_at": time.time_ns(),
            "updated_at": time.time_ns(),
            "current_epoch": 100,
            "current_step": 242700,
            "current_loss": 0.0535,
        }
        run_bytes = bytes(json.dumps(run_dict), "utf-8")
        storage_client.save(f"runs/{run_dict['name']}/meta.json", run_bytes)

        run_config_dict = {"nature": "earth"}
        run_config_bytes = bytes(yaml.dump(run_config_dict), "utf-8")
        storage_client.save(f"runs/{run_dict['name']}/config.yaml", run_config_bytes)

        run = run_repository.get("gaara")

        assert run_dict["id"] == run.id
        assert run_dict["name"] == run.name
        assert run_dict["status"] == run.status.value
        assert run_dict["created_at"] == run.created_at
        assert run_dict["updated_at"] == run.updated_at
        assert run_dict["current_epoch"] == run.current_epoch
        assert run_dict["current_step"] == run.current_step
        assert run_dict["current_loss"] == run.current_loss
        assert run_config_dict == run.config.to_dict()


@pytest.fixture
def artifact_repository(storage_client):
    return RunArtifactRepository(storage_client)


class TestRunArtifactRepository:
    """Test suite for the :class:`lumiere.training.run.Run` class."""

    # TODO: Implement remaining tests.
    def test_insert_saves_artifact_to_storage_location(
        self, storage_client, artifact_repository
    ):
        artifact = b"3534xcc-53345d-4ngfkb"

        artifact_repository.insert("ultimate-ninja", "raikiri-staff", artifact)

        assert (
            storage_client.load("runs/ultimate-ninja/artifacts/raikiri-staff")
            == artifact
        )

    def test_get_loads_artifact_from_storage_location(
        self, storage_client, artifact_repository
    ):
        artifact = b"3534xcc-53345d-4ngfkb"

        storage_client.save("runs/ultimate-ninja/artifacts/raikiri-staff", artifact)

        assert artifact_repository.get("ultimate-ninja", "raikiri-staff") == artifact

    def test_get_returns_none_if_artifact_does_not_exist(
        self, storage_client, artifact_repository
    ):
        artifact = b"3534xcc-53345d-4ngfkb"

        storage_client.save("runs/ultimate-ninja/artifacts/raikiri-staff", artifact)

        assert artifact_repository.get("ultimate-ninja", "sand-gourd") is None
