import os
import pickle
import random
import string
from collections import namedtuple

import pytest
from azure.core.exceptions import AzureError
from azure.storage.blob import BlobServiceClient

from lumiere.persistence.clients import AzureBlobStorageClient
from lumiere.persistence.clients.azure_blob_storage_client import (
    disable_tokenizer_parallelism,
)
from lumiere.persistence.errors import (
    StorageError,
)
from lumiere.utils import randomizer


def random_bytes() -> bytes:
    return b"somerandombytes"


@pytest.fixture(scope="module")
def blob_service_client():
    """Azure Blob Service client configured from connection string."""
    az_blob_conn_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
    if az_blob_conn_string is None:
        raise KeyError("Connection string not found for Azure Blob Storage.")

    yield (client := BlobServiceClient.from_connection_string(az_blob_conn_string))

    client.close()


@pytest.fixture
def container(blob_service_client):
    """Creates a random container in Azure Blob that will be cleaned up after use."""
    test_id = "".join([random.choice(string.ascii_lowercase) for _ in range(6)])
    container_name = f"test-{test_id}"
    container_client = blob_service_client.create_container(
        container_name, metadata={"Category": "test"}
    )

    Container = namedtuple("Container", ["name", "client"])
    yield Container(name=container_name, client=container_client)

    container_client.delete_container()


@pytest.fixture
def client(blob_service_client, container) -> AzureBlobStorageClient:
    container_name, _ = container

    return AzureBlobStorageClient(blob_service_client, container_name)


@pytest.fixture
def container_client(container):
    """Return a client for an ephemeral test container."""
    _, container_client = container
    return container_client


@pytest.fixture
def run_id():
    """Return a randomly generated run id using :meth:`randomizer.random_id`."""
    return randomizer.random_id()


@pytest.fixture
def artifact_key():
    """Return a randomly generated artifact key."""
    vocab = string.ascii_uppercase + string.ascii_lowercase
    return "".join(random.choice(vocab) for _ in range(6))


@pytest.fixture
def az_storage_client(blob_service_client, container):
    """AzureBlobStorageClient instance for testing."""
    return AzureBlobStorageClient(blob_service_client, container.name)


def _add_target_checkpoint(container_client):
    run_id = randomizer.random_id()
    checkpoint_tags = ["epoch_0001", "epoch_0002", "epoch_0003", "best", "final"]
    target_checkpoint_tag = random.choice(checkpoint_tags)

    for checkpoint_tag in checkpoint_tags:
        checkpoint_path = f"runs/{run_id}/checkpoints/{checkpoint_tag}.pt"
        checkpoint_data = (
            b"target" if checkpoint_tag == target_checkpoint_tag else b"test"
        )
        container_client.upload_blob(checkpoint_path, checkpoint_data)

    TargetCheckpoint = namedtuple("TargetCheckpoint", ["run_id", "name"])
    return TargetCheckpoint(run_id=run_id, name=target_checkpoint_tag)


class TestAzureBlobStorageClient:
    """Test suite for :class:`lumiere.persistence.clients.AzureBlobStorageClient`."""

    @pytest.mark.slow
    def test_save_checkpoint_successfully_uploads_artifact(
        self, az_storage_client, container
    ):
        run_id = randomizer.random_id(n=8)
        checkpoint_tag = "test"
        checkpoint_data = b"test"

        az_storage_client.save_checkpoint(run_id, checkpoint_tag, checkpoint_data)

        assert container.client.get_blob_client(
            f"runs/{run_id}/checkpoints/{checkpoint_tag}.pt"
        ).exists()

    # TODO: Add test to verify behavior when run does not exist.

    def test_save_checkpoint_raises_error_when_upload_fails(
        self, mocker, container, blob_service_client
    ):
        mocker.patch(
            "azure.storage.blob.BlobClient.upload_blob",
            side_effect=Exception("Upload failed"),
        )

        az_storage_client = AzureBlobStorageClient(blob_service_client, container.name)

        with pytest.raises(StorageError):
            az_storage_client.save_checkpoint("abc", "best", b"test")

    # -----------------------------------
    # ------- SAVE_ARTIFACT TESTS -------
    # -----------------------------------

    @pytest.mark.slow
    @pytest.mark.parametrize("artifact", [[1, 2, 3], {"test": "data"}])
    def test_save_artifact_uploads_object_to_blob_storage(
        self, az_storage_client, container, artifact
    ):
        run_id = randomizer.random_id()
        key = "testkey"
        expected_blob_name = f"runs/{run_id}/artifacts/{key}"
        _, container_client = container
        blob_client = container_client.get_blob_client(expected_blob_name)

        # Verify the artifact doesn't already exist.
        assert not blob_client.exists()

        az_storage_client.save_artifact(run_id, key, artifact)

        # Verify the artifact has been created.
        assert blob_client.exists()
        assert blob_client.get_blob_properties().size != 0

    @pytest.mark.slow
    def test_save_artifact_overwrites_existing_artifact_is_overwrite_is_true(
        self, az_storage_client, container
    ):
        run_id = randomizer.random_id()
        key = "testkey"
        expected_blob_name = f"runs/{run_id}/artifacts/{key}"
        _, container_client = container
        blob_client = container_client.get_blob_client(expected_blob_name)

        # Verify the artifact doesn't already exist.
        assert not blob_client.exists()

        # Store the first version of the artifact and verify it's stored correctly.
        original_artifact = {"data": "original"}
        az_storage_client.save_artifact(run_id, key, original_artifact)

        assert blob_client.exists()
        reconstructed_original = pickle.loads(blob_client.download_blob().readall())
        assert original_artifact == reconstructed_original

        # Store another version of the artifact and verify it overwrites the previous.
        new_artifact = {"data": "new"}
        az_storage_client.save_artifact(run_id, key, new_artifact, overwrite=True)

        reconstructed_new = pickle.loads(blob_client.download_blob().readall())
        assert reconstructed_new != original_artifact

    @pytest.mark.slow
    def test_save_artifact_raises_error_if_existing_key_and_overwrite_is_false(
        self, az_storage_client, container_client, run_id, artifact_key
    ):
        blob_client = container_client.get_blob_client(
            f"runs/{run_id}/artifacts/{artifact_key}"
        )

        # Verify the artifact doesn't already exist.
        assert not blob_client.exists()

        # Store the artifact and verify it's been store successfully.
        artifact = {"secret_formula": "e=mc^2"}
        az_storage_client.save_artifact(run_id, artifact_key, artifact, overwrite=False)

        assert blob_client.exists()

        # Attempt to overwrite the previous artifact.
        new_artifact = {"secret_formula": "e=mc^(1/2)"}
        with pytest.raises(KeyError):
            az_storage_client.save_artifact(
                run_id, artifact_key, new_artifact, overwrite=False
            )

        assert pickle.loads(blob_client.download_blob().readall()) == artifact

    @pytest.mark.slow
    def test_save_artifact_does_not_overwite_existing_artifacts_by_default(
        self, az_storage_client, container_client, run_id, artifact_key
    ):
        blob_client = container_client.get_blob_client(
            f"runs/{run_id}/artifacts/{artifact_key}"
        )

        assert not blob_client.exists()

        az_storage_client.save_artifact(run_id, artifact_key, ["launch", "codes"])
        assert blob_client.exists()

        with pytest.raises(KeyError):
            az_storage_client.save_artifact(run_id, artifact_key, ["<redacted>"])

    def test_save_artifact_raises_error_if_error_occurs_during_upload(
        self, mocker, az_storage_client, run_id, artifact_key
    ):
        mocker.patch(
            "azure.storage.blob.BlobClient.upload_blob", side_effect=Exception()
        )

        with pytest.raises(StorageError):
            az_storage_client.save_artifact(run_id, artifact_key, ("apple", 42))

    # -----------------------------------
    # ---------- `LOAD` TESTS -----------
    # -----------------------------------
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "path", ["anthropic", "companies/anthropic", "companies/ai/labs/anthropic"]
    )
    def test_load_retrieves_data_from_blob(self, client, container_client, path):
        expected_data = random_bytes()
        container_client.upload_blob(path, expected_data)

        actual_data = client.load(path)

        assert actual_data == expected_data

    @pytest.mark.slow
    def test_load_returns_none_if_blob_could_not_be_found(self, client):
        assert client.load("viltrum") is None

    @pytest.mark.slow
    def test_load_raises_error_if_blob_container_does_not_exist(
        self, blob_service_client
    ):
        client = AzureBlobStorageClient(blob_service_client, "planet vegeta")

        with pytest.raises(StorageError):
            client.load("Bardock")

    def test_load_raises_error_if_error_occurs_while_retrieving_blob(
        self, mocker, client
    ):
        mocker.patch(
            "azure.storage.blob.BlobClient.download_blob",
            side_effect=AzureError("A random error occurred."),
        )

        with pytest.raises(StorageError):
            client.load("randomblob")


@pytest.mark.slow
class TestDisableTokenizerParallelism:
    """Test suite for :func:`lumiere.persistence.clients.AzureBlobStorageClient.disable_tokenizer_parallelism`."""  # noqa: E501

    def setup_method(self):
        # Ensure clean state
        if "TOKENIZERS_PARALLELISM" in os.environ:
            del os.environ["TOKENIZERS_PARALLELISM"]

    def test_sets_tokenizers_parallelism_to_false(self):
        with disable_tokenizer_parallelism():
            assert os.getenv("TOKENIZERS_PARALLELISM") == "false"

    def test_restores_original_value_when_set(self):
        original_value = "true"
        os.environ["TOKENIZERS_PARALLELISM"] = original_value

        with disable_tokenizer_parallelism():
            assert os.getenv("TOKENIZERS_PARALLELISM") == "false"

        assert os.getenv("TOKENIZERS_PARALLELISM") == original_value

    def test_removes_env_var_when_not_originally_set(self):
        assert os.getenv("TOKENIZERS_PARALLELISM") is None

        with disable_tokenizer_parallelism():
            assert os.getenv("TOKENIZERS_PARALLELISM") == "false"

        assert os.getenv("TOKENIZERS_PARALLELISM") is None

    def test_restores_state_on_exception(self):
        original_value = "true"
        os.environ["TOKENIZERS_PARALLELISM"] = original_value

        with pytest.raises(ValueError), disable_tokenizer_parallelism():
            assert os.getenv("TOKENIZERS_PARALLELISM") == "false"
            raise ValueError("Test exception")

        assert os.getenv("TOKENIZERS_PARALLELISM") == original_value

    def test_removes_env_var_on_exception_when_not_originally_set(self):
        if "TOKENIZERS_PARALLELISM" in os.environ:
            del os.environ["TOKENIZERS_PARALLELISM"]

        with pytest.raises(ValueError), disable_tokenizer_parallelism():
            assert os.getenv("TOKENIZERS_PARALLELISM") == "false"
            raise ValueError("Test exception")

        assert os.getenv("TOKENIZERS_PARALLELISM") is None

    def test_handles_edge_case_values(self):
        # Test with empty string (should be preserved)
        os.environ["TOKENIZERS_PARALLELISM"] = ""

        with disable_tokenizer_parallelism():
            assert os.getenv("TOKENIZERS_PARALLELISM") == "false"

        assert os.getenv("TOKENIZERS_PARALLELISM") == ""

        # Test with None (env var not set)
        del os.environ["TOKENIZERS_PARALLELISM"]
        assert os.getenv("TOKENIZERS_PARALLELISM") is None

        with disable_tokenizer_parallelism():
            assert os.getenv("TOKENIZERS_PARALLELISM") == "false"

        assert os.getenv("TOKENIZERS_PARALLELISM") is None
