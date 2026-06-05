import pytest

from lumiere.testing.storage_clients import MemoryStorageClient
from lumiere.training.artifact import ArtifactStore


@pytest.fixture
def storage_client():
    return MemoryStorageClient()


@pytest.fixture
def artifact_store(storage_client):
    return ArtifactStore(storage_client)


class TestRunArtifactStore:
    def test_insert_saves_artifact_to_storage_location(
        self, storage_client, artifact_store
    ):
        artifact = b"3534xcc-53345d-4ngfkb"

        artifact_store.add("ultimate-ninja", "raikiri-staff", artifact)

        assert (
            storage_client.load("runs/ultimate-ninja/artifacts/raikiri-staff")
            == artifact
        )

    def test_get_loads_artifact_from_storage_location(
        self, storage_client, artifact_store
    ):
        artifact = b"3534xcc-53345d-4ngfkb"

        storage_client.save("runs/ultimate-ninja/artifacts/raikiri-staff", artifact)

        assert artifact_store.get("ultimate-ninja", "raikiri-staff") == artifact

    def test_get_returns_none_if_artifact_does_not_exist(
        self, storage_client, artifact_store
    ):
        artifact = b"3534xcc-53345d-4ngfkb"

        storage_client.save("runs/ultimate-ninja/artifacts/raikiri-staff", artifact)

        assert artifact_store.get("ultimate-ninja", "sand-gourd") is None
