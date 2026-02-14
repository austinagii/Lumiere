import pickle
import random
import string
from pathlib import Path

import pytest
import yaml

from lumiere.persistence.clients import FileSystemStorageClient
from lumiere.persistence.errors import StorageError
from lumiere.training.run import generate_run_id


# TODO: Add the following two fixtures to a test utils module.
@pytest.fixture
def run_id():
    """Generate a random run id."""
    return generate_run_id()


@pytest.fixture
def artifact_key():
    """Generate a random artifact key."""
    vocab = string.ascii_uppercase + string.ascii_lowercase
    return "".join([random.choice(vocab) for _ in range(6)])


@pytest.fixture
def fs_storage_client(tmp_path):
    """Returns a FileSystemStorageClient instance.

    The provided instance uses a temporary directory as its base storage path.
    """
    return FileSystemStorageClient(base_dir=tmp_path)


class TestFileSystemStorageClient:
    def test_can_be_initialized_with_str(self, tmp_path):
        pathstr = str(tmp_path)

        client = FileSystemStorageClient(pathstr)

        assert client.base_dir == tmp_path

    def test_can_be_initialized_with_path(self, tmp_path):
        client = FileSystemStorageClient(tmp_path)

        assert client.base_dir == tmp_path

    def test_init_run_creates_a_new_directory_containing_the_train_config(
        self, tmp_path, fs_storage_client, run_id
    ):
        train_config_yaml = """
        name: gpt2
        model:
            num_heads: 10
            num_blocks: 3
        """
        train_config = yaml.safe_load(train_config_yaml)

        fs_storage_client.init_run(run_id, train_config)

        expected_config_path = tmp_path / f"{run_id}/config.yaml"

        assert expected_config_path.exists()
        # Text written to disk should be the equivalent of a yaml dump of the config.
        assert expected_config_path.read_text() == yaml.dump(train_config)

    def test_save_checkpoint_creates_directory_if_it_does_not_exist(
        self, fs_storage_client, run_id
    ):
        expected_checkpoint_dir = fs_storage_client.base_dir / f"{run_id}/checkpoints/"

        assert not expected_checkpoint_dir.exists()

        fs_storage_client.save_checkpoint(
            run_id=run_id, checkpoint_tag="best", checkpoint=b"test"
        )

        assert expected_checkpoint_dir.exists()

    def test_save_checkpoint_correctly_save_checkpoints_the_artifact(
        self, fs_storage_client, run_id
    ):
        fs_storage_client.save_checkpoint(
            run_id=run_id, checkpoint_tag="best", checkpoint=b"test"
        )

        expected_checkpoint_path = (
            fs_storage_client.base_dir / f"{run_id}/checkpoints/best.pt"
        )

        assert expected_checkpoint_path.exists()
        assert expected_checkpoint_path.read_text() == "test"

    def test_save_checkpoint_overwrites_existing_artifacts(
        self, fs_storage_client, run_id
    ):
        expected_checkpoint_path = (
            fs_storage_client.base_dir / f"{run_id}/checkpoints/best.pt"
        )

        # Write some content to the checkpoint path. We expect this to be overridden.
        expected_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        expected_checkpoint_path.write_text("test")

        fs_storage_client.save_checkpoint(
            run_id=run_id, checkpoint_tag="best", checkpoint=b"test2"
        )

        assert expected_checkpoint_path.read_text() == "test2"

    def test_load_checkpoint_successfully_load_checkpoints_the_artifact(
        self, fs_storage_client
    ):
        expected_checkpoint_path = (
            fs_storage_client.base_dir / f"{run_id}/checkpoints/best.pt"
        )

        expected_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        expected_checkpoint_path.write_text("test")

        checkpoint_bytes = fs_storage_client.load_checkpoint(
            run_id=run_id, checkpoint_tag="best"
        )

        assert checkpoint_bytes == b"test"

    # ===============================
    # ===== SAVE_ARTIFACT TESTS =====
    # ===============================
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "artifact",
        [
            {"best_snack": "kit-kat"},
            ["San Francisco", "Tokyo", "Medellin", "Vancouver"],
        ],
    )
    def test_save_artifact_writes_artifact_to_file_system(
        self, fs_storage_client, run_id, artifact_key, artifact
    ):
        expected_artifact_path = Path(
            f"{fs_storage_client.base_dir}/{run_id}/artifacts/{artifact_key}"
        )

        fs_storage_client.save_artifact(run_id, artifact_key, artifact)

        assert pickle.loads(expected_artifact_path.read_bytes()) == artifact

    def test_save_artifact_overwrites_existing_if_overwrite_is_true(
        self, fs_storage_client, run_id, artifact_key
    ):
        artifact = ("Sam Altman", "Ilya Sutskever", "Dario Amodei", "Mira Murati")
        expected_artifact_path = Path(
            f"{fs_storage_client.base_dir}/{run_id}/artifacts/{artifact_key}"
        )

        assert not expected_artifact_path.exists()

        # Save the artifact.
        fs_storage_client.save_artifact(run_id, artifact_key, artifact)

        assert pickle.loads(expected_artifact_path.read_bytes()) == artifact

        # Attempt to overwrite the artifact.
        new_artifact = "Sam Altman"

        fs_storage_client.save_artifact(
            run_id, artifact_key, new_artifact, overwrite=True
        )

        # Verify that the overwrite was successful.
        assert pickle.loads(expected_artifact_path.read_bytes()) != artifact
        assert pickle.loads(expected_artifact_path.read_bytes()) == new_artifact

    def test_save_artifact_raises_error_if_existing_key_and_overwrite_is_false(
        self, fs_storage_client, run_id, artifact_key
    ):
        artifact = ("Sam Altman", "Ilya Sutskever", "Dario Amodei", "Mira Murati")
        expected_artifact_path = Path(
            f"{fs_storage_client.base_dir}/{run_id}/artifacts/{artifact_key}"
        )

        assert not expected_artifact_path.exists()

        # Save the artifact.
        fs_storage_client.save_artifact(run_id, artifact_key, artifact)

        # Verify that an error occurrs when attempting to overwrite.
        new_artifact = "Sam Altman"
        with pytest.raises(KeyError):
            fs_storage_client.save_artifact(
                run_id, artifact_key, new_artifact, overwrite=False
            )

    def test_save_artifact_does_not_overwrite_artifacts_by_default(
        self, fs_storage_client, run_id, artifact_key
    ):
        artifact = ("Sam Altman", "Ilya Sutskever", "Dario Amodei", "Mira Murati")
        expected_artifact_path = Path(
            f"{fs_storage_client.base_dir}/{run_id}/artifacts/{artifact_key}"
        )

        assert not expected_artifact_path.exists()

        # Save the artifact.
        fs_storage_client.save_artifact(run_id, artifact_key, artifact)

        # Verify that an error occurrs when writing with same key.
        new_artifact = "Sam Altman"
        with pytest.raises(KeyError):
            fs_storage_client.save_artifact(run_id, artifact_key, new_artifact)

    def test_save_artifact_raises_error_if_error_occurred_while_writing_to_file_system(
        self, mocker, fs_storage_client, run_id, artifact_key
    ):
        mocker.patch("pathlib.Path.write_bytes", side_effect=Exception())

        with pytest.raises(StorageError):
            fs_storage_client.save_artifact(run_id, artifact_key, set("B200"))

    # ===============================
    # ===== LOAD_ARTIFACT TESTS =====
    # ===============================
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "artifact",
        [
            {"best_snack": "kit-kat"},
            ["San Francisco", "Tokyo", "Medellin", "Vancouver"],
            ("Sam Altman", "Ilya Sutskever", "Dario Amodei", "Mira Murati"),
        ],
    )
    def test_load_artifact_reads_artifact_from_file_system(
        self, fs_storage_client, run_id, artifact_key, artifact
    ):
        expected_artifact_path = Path(
            f"{fs_storage_client.base_dir}/{run_id}/artifacts/{artifact_key}"
        )

        # Create the artifact file directly on the file system
        expected_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        expected_artifact_path.write_bytes(pickle.dumps(artifact))

        loaded_artifact = fs_storage_client.load_artifact(run_id, artifact_key)

        assert loaded_artifact == artifact

    def test_load_artifact_returns_none_if_artifact_not_found(
        self, fs_storage_client, run_id, artifact_key
    ):
        loaded_artifact = fs_storage_client.load_artifact(run_id, artifact_key)

        assert loaded_artifact is None

    def test_load_artifact_raises_error_if_error_occurred_while_reading_from_file_system(
        self, mocker, fs_storage_client, run_id, artifact_key
    ):
        artifact = {"test": "data"}
        expected_artifact_path = Path(
            f"{fs_storage_client.base_dir}/{run_id}/artifacts/{artifact_key}"
        )

        # Create the artifact file so it exists
        expected_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        expected_artifact_path.write_bytes(pickle.dumps(artifact))

        # Mock read_bytes to raise an exception
        mocker.patch("pathlib.Path.read_bytes", side_effect=Exception())

        with pytest.raises(StorageError):
            fs_storage_client.load_artifact(run_id, artifact_key)

    def test_load_artifact_successfully_reconstructs_saved_artifacts(
        self, fs_storage_client, run_id, artifact_key
    ):
        original_artifact = {
            "model_params": {"lr": 0.001, "batch_size": 32},
            "metrics": [0.95, 0.97, 0.98],
            "metadata": ("experiment", "version_1"),
        }

        # Save the artifact using save_artifact
        fs_storage_client.save_artifact(run_id, artifact_key, original_artifact)

        # Load it back using load_artifact
        loaded_artifact = fs_storage_client.load_artifact(run_id, artifact_key)

        # Verify complete reconstruction
        assert loaded_artifact == original_artifact
        assert type(loaded_artifact) == type(original_artifact)
        assert loaded_artifact["model_params"] == original_artifact["model_params"]
        assert loaded_artifact["metrics"] == original_artifact["metrics"]
        assert loaded_artifact["metadata"] == original_artifact["metadata"]
