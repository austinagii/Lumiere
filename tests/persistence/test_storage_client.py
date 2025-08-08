from pathlib import Path

import pytest
from azure.storage.blob import BlobServiceClient

from lumiere.persistence.errors import PersistenceError
from lumiere.persistence.storage_client import LocalStorageClient, RemoteStorageClient


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

        local_storage_client.store(file_path, b"test")

        assert (local_storage_client.base_dir / file_path).exists()

    def test_store_correctly_stores_the_artifact(self, local_storage_client, file_path):
        local_storage_client.store(file_path, b"test")

        assert (local_storage_client.base_dir / file_path).exists()
        assert (local_storage_client.base_dir / file_path).read_text() == "test"

    def test_store_overwrites_existing_artifacts(self, local_storage_client, file_path):
        expected_file_path = local_storage_client.base_dir / file_path
        expected_file_path.parent.mkdir(parents=True, exist_ok=True)

        expected_file_path.write_text("test")
        local_storage_client.store(file_path, b"test2")

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


@pytest.fixture
def remote_storage_client(blob_service_client):
    return RemoteStorageClient(blob_service_client, "testcontainer")


class TestRemoteStorageClient:
    def test_exists_returns_true_if_artifact_is_present(self, remote_storage_client):
        assert remote_storage_client.exists("test.txt")

    def test_exists_returns_false_if_artifact_is_not_present(
        self, mocker, remote_storage_client
    ):
        blob_client = mocker.Mock()
        blob_client.exists.return_value = False

        mocker.patch.object(
            remote_storage_client.blob_service_client,
            "get_blob_client",
            return_value=blob_client,
        )

        assert not remote_storage_client.exists("test.txt")

    def test_exists_raises_an_error_if_an_error_occurs(
        self, mocker, remote_storage_client
    ):
        blob_client = mocker.Mock()
        blob_client.exists.side_effect = Exception("test")

        mocker.patch.object(
            remote_storage_client.blob_service_client,
            "get_blob_client",
            return_value=blob_client,
        )

        with pytest.raises(PersistenceError):
            remote_storage_client.exists("test.txt")

    def test_store_successfully_uploads_artifact(self, mocker, remote_storage_client):
        blob_client = mocker.Mock()

        mocker.patch.object(
            remote_storage_client.blob_service_client,
            "get_blob_client",
            return_value=blob_client,
        )

        test_artifact = b"test artifact content"
        remote_storage_client.store("test.txt", test_artifact)

        blob_client.upload_blob.assert_called_once_with(test_artifact, overwrite=True)

    def test_store_raises_error_when_upload_fails(self, mocker, remote_storage_client):
        blob_client = mocker.Mock()
        blob_client.upload_blob.side_effect = Exception("Upload failed")

        mocker.patch.object(
            remote_storage_client.blob_service_client,
            "get_blob_client",
            return_value=blob_client,
        )

        with pytest.raises(
            PersistenceError,
            match="An error occurred while syncing checkpoint to blob storage",
        ):
            remote_storage_client.store("test.txt", b"test")

    def test_retrieve_successfully_downloads_artifact(
        self, mocker, remote_storage_client
    ):
        blob_client = mocker.Mock()
        blob_client.exists.return_value = True

        # Mock the download_blob method and its readall method
        mock_download_blob = mocker.Mock()
        mock_download_blob.readall.return_value = b"downloaded content"
        blob_client.download_blob.return_value = mock_download_blob

        mocker.patch.object(
            remote_storage_client.blob_service_client,
            "get_blob_client",
            return_value=blob_client,
        )

        result = remote_storage_client.retrieve("test.txt")

        assert result == b"downloaded content"
        blob_client.exists.assert_called_once()
        blob_client.download_blob.assert_called_once()

    def test_retrieve_raises_error_when_artifact_not_found(
        self, mocker, remote_storage_client
    ):
        blob_client = mocker.Mock()
        blob_client.exists.return_value = False

        mocker.patch.object(
            remote_storage_client.blob_service_client,
            "get_blob_client",
            return_value=blob_client,
        )

        with pytest.raises(
            PersistenceError, match="Checkpoint could not be found in blob"
        ):
            remote_storage_client.retrieve("test.txt")

    def test_retrieve_raises_error_when_download_fails(
        self, mocker, remote_storage_client
    ):
        blob_client = mocker.Mock()
        blob_client.exists.return_value = True
        blob_client.download_blob.side_effect = Exception("Download failed")

        mocker.patch.object(
            remote_storage_client.blob_service_client,
            "get_blob_client",
            return_value=blob_client,
        )

        with pytest.raises(
            PersistenceError,
            match="An error occurred while downloading the artifact from the remote device",
        ):
            remote_storage_client.retrieve("test.txt")

    def test_retrieve_handles_exception_during_exists_check(
        self, mocker, remote_storage_client
    ):
        blob_client = mocker.Mock()
        blob_client.exists.side_effect = Exception("Exists check failed")

        mocker.patch.object(
            remote_storage_client.blob_service_client,
            "get_blob_client",
            return_value=blob_client,
        )

        with pytest.raises(
            PersistenceError,
            match="An error occurred while downloading the artifact from the remote device",
        ):
            remote_storage_client.retrieve("test.txt")
