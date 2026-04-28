import os
from collections import namedtuple

import pytest
from azure.core.exceptions import AzureError
from azure.storage.blob import BlobServiceClient

from lumiere.persistence.clients import AzureBlobStorageClient
from lumiere.persistence.clients.azure_blob_storage_client import (
    disable_tokenizer_parallelism,
)
from lumiere.persistence.errors import StorageError
from lumiere.utils import randomizer


@pytest.fixture(scope="module")
def blob_service_client():
    """An azure blob storage client."""
    az_blob_conn_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
    if az_blob_conn_string is None:
        raise KeyError("Connection string not found for Azure Blob Storage.")

    yield (client := BlobServiceClient.from_connection_string(az_blob_conn_string))

    client.close()


@pytest.fixture
def tmp_container(blob_service_client):
    """A temporary blob storage container."""
    container_name = f"test-{randomizer.random_id(include_alpha_upper=False)}"
    container_client = blob_service_client.create_container(
        container_name, metadata={"Category": "test"}
    )

    Container = namedtuple("Container", ["name", "client"])
    yield Container(name=container_name, client=container_client)

    container_client.delete_container()


@pytest.fixture
def client(blob_service_client, tmp_container) -> AzureBlobStorageClient:
    return AzureBlobStorageClient(blob_service_client, tmp_container.name)


class TestAzureBlobStorageClient:
    """Test suite for :class:`lumiere.persistence.clients.AzureBlobStorageClient`."""

    # -----------------------------------
    # ------- `SAVE` TESTS -------
    # -----------------------------------

    @pytest.mark.slow
    @pytest.mark.parametrize("path", ["jujutsu-high/records"])
    def test_save_uploads_data_as_blob_to_azure_blob_storage(
        self, client, tmp_container, path
    ):
        data = b"divergent-fist"
        blob_client = tmp_container.client.get_blob_client(path)

        # Verify the artifact doesn't already exist.
        assert not blob_client.exists()

        client.save(path, data)

        # Verify the artifact has been created.
        assert blob_client.exists()
        assert blob_client.get_blob_properties().size != 0

    @pytest.mark.slow
    def test_save_overwrites_existing_blob_if_overwrite_flag_is_true(
        self, client, tmp_container
    ):
        path = "shibuya/october-31/vessel.log"
        blob_client = tmp_container.client.get_blob_client(path)

        assert not blob_client.exists()

        client.save(path, b"yuji-in-control")
        assert blob_client.exists()
        assert blob_client.download_blob().readall() == b"yuji-in-control"

        client.save(path, b"sukuna-takes-over", overwrite=True)
        assert blob_client.download_blob().readall() == b"sukuna-takes-over"

    def test_save_prevents_overwriting_existing_blob_by_default(
        self, client, tmp_container
    ):
        path = "jujutsu-high/binding-vow"
        blob_client = tmp_container.client.get_blob_client(path)

        assert not blob_client.exists()

        client.save(path, b"yuji-itadori")
        assert blob_client.exists()
        assert blob_client.download_blob().readall() == b"yuji-itadori"

        with pytest.raises(StorageError):
            client.save(path, b"ryomen-sukuna")

        assert blob_client.download_blob().readall() == b"yuji-itadori"

    def test_save_raises_error_if_error_occurs_during_upload(self, mocker, client):
        mocker.patch(
            "azure.storage.blob.BlobClient.upload_blob", side_effect=Exception()
        )

        with pytest.raises(StorageError):
            client.save("yuki/cursed-techniques", b"antigravity")

    # -----------------------------------
    # ---------- `LOAD` TESTS -----------
    # -----------------------------------
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "path",
        [
            "sorcerers/grade-1",
            "jujutsu-high/sorcerers/grade-1",
            "jujutsu-high/tokyo/sorcerers/grade-1",
        ],
    )
    def test_load_retrieves_data_from_blob(self, client, tmp_container, path):
        expected_data = b"yuji-itadori"
        tmp_container.client.upload_blob(path, expected_data)

        actual_data = client.load(path)

        assert actual_data == expected_data

    @pytest.mark.slow
    def test_load_returns_none_if_blob_could_not_be_found(self, client):
        assert client.load("gojo/prison-realm/location") is None

    @pytest.mark.slow
    def test_load_returns_none_if_blob_container_does_not_exist(
        self, blob_service_client
    ):
        client = AzureBlobStorageClient(blob_service_client, "death-painting-womb")

        assert client.load("cursed-spirit-painting") is None

    def test_load_raises_error_if_error_occurs_while_retrieving_blob(
        self, mocker, client
    ):
        mocker.patch(
            "azure.storage.blob.BlobClient.download_blob",
            side_effect=AzureError("A random error occurred."),
        )

        with pytest.raises(StorageError):
            client.load("innate-domain/unlimited-void")


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
