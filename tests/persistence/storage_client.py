from pathlib import Path

import pytest
from azure.storage.blob import BlobServiceClient

from lumiere.persistence.storage_client import LocalStorageClient


@pytest.fixture
def local_storage_client(tmp_path):
    return LocalStorageClient(base_dir=tmp_path)


@pytest.fixture(params=["test.txt", "test/test.txt"])
def file_path(request):
    return Path(request.param)


class TestLocalStorageClient:
    def test_exists_returns_true_if_artifact_is_present(
        self, local_storage_client, file_path
    ):
        expected_file_path = local_storage_client.base_dir / file_path
        expected_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(expected_file_path, "x") as file:
            file.write("test")

        assert local_storage_client.exists(file_path)

    def test_exists_returns_false_if_artifact_is_not_present(
        self, local_storage_client, file_path
    ):
        assert not local_storage_client.exists(file_path)

    def test_store_creates_directory_if_it_does_not_exist(
        self, local_storage_client, file_path
    ):
        assert not (local_storage_client.base_dir / file_path).exists()

        local_storage_client.store(b"test", file_path)

        assert (local_storage_client.base_dir / file_path).exists()

    def test_store_correctly_stores_the_artifact(self, local_storage_client, file_path):
        local_storage_client.store(b"test", file_path)

        assert (local_storage_client.base_dir / file_path).exists()
        assert (local_storage_client.base_dir / file_path).read_text() == "test"

    def test_store_overwrites_existing_artifacts(self, local_storage_client, file_path):
        expected_file_path = local_storage_client.base_dir / file_path
        expected_file_path.parent.mkdir(parents=True, exist_ok=True)

        expected_file_path.write_text("test")
        local_storage_client.store(b"test2", file_path)

        assert expected_file_path.read_text() == "test2"

    def test_retrieve_successfully_retrieves_the_artifact(
        self, local_storage_client, file_path
    ):
        expected_file_path = local_storage_client.base_dir / file_path
        expected_file_path.parent.mkdir(parents=True, exist_ok=True)

        expected_file_path.write_text("test")

        retrieved_artifact = local_storage_client.retrieve(file_path)
        assert retrieved_artifact == b"test"


@pytest.fixture
def blob_client(mocker):
    blob_client = mocker.Mock()
    blob_client.exists.return_value = True
    return blob_client


@pytest.fixture
def blob_service_client(mocker, blob_client):
    blob_service_client = BlobServiceClient("testaccounturl")
    mocker.patch.object(
        blob_service_client, "get_blob_client", return_value=blob_client
    )
    return blob_service_client
