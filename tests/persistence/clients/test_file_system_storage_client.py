import pytest

from lumiere.persistence.clients import FileSystemStorageClient
from lumiere.persistence.errors import StorageError


@pytest.fixture
def client(tmp_path):
    """Returns a FileSystemStorageClient instance.

    The provided instance uses a temporary directory as its base storage path.
    """
    return FileSystemStorageClient(base_dir=tmp_path)


@pytest.fixture
def data():
    return bytes("Test Data", "utf-8")


class TestFileSystemStorageClient:
    """Test suite for :class:`lumiere.persistence.clients.FileSystemStorageClient`."""

    def test_init_accepts_base_path_as_str(self, tmp_path):
        pathstr = str(tmp_path)

        client = FileSystemStorageClient(pathstr)

        assert client.base_dir == tmp_path

    def test_init_accepts_base_path_as_path_obj(self, tmp_path):
        client = FileSystemStorageClient(tmp_path)

        assert client.base_dir == tmp_path

    @pytest.mark.parametrize(
        "path",
        [
            "marathon.txt",  # creates directly in base directory.
            "marathon/runners/thief.mt",  # creates any needed subdirectories.
        ],
    )
    def test_save_writes_data_to_a_file_at_the_specified_path(self, client, path, data):
        expected_path = client.base_dir / path

        assert not expected_path.exists()

        client.save(path, data)

        assert expected_path.read_bytes() == data

    def test_save_prevents_overwriting_existing_files_by_default(self, client):
        original_data = b"This was first."
        new_data = b"Then this came."
        expected_file_path = client.base_dir / "userdata"

        assert not expected_file_path.exists()

        client.save("userdata", original_data)
        assert expected_file_path.read_bytes() == original_data

        with pytest.raises(StorageError):
            client.save("userdata", new_data)
        assert expected_file_path.read_bytes() == original_data

    def test_save_overwrites_existing_files_if_overwrite_flag_is_set(self, client):
        original_data = b"This was first."
        new_data = b"Then this came."
        expected_file_path = client.base_dir / "userdata"

        assert not expected_file_path.exists()

        client.save("userdata", original_data)
        assert expected_file_path.read_bytes() == original_data

        client.save("userdata", new_data, overwrite=True)
        assert expected_file_path.read_bytes() == new_data

    @pytest.mark.parametrize("path", [None, 1, 1.5])
    def test_save_raises_an_error_if_the_path_is_invalid(self, client, path, data):
        with pytest.raises(ValueError):
            client.save(path, data)

    def test_save_raises_an_error_if_data_could_not_be_written_to_the_filesystem(
        self, tmp_path, data
    ):
        base_dir = tmp_path / "ps6"
        base_dir.mkdir(0o111, parents=True, exist_ok=False)
        client = FileSystemStorageClient(base_dir)

        path = client.base_dir / "games/mgs3"
        try:
            with pytest.raises(StorageError):
                client.save(path, data)
        finally:
            base_dir.chmod(0o755)

    # ===============================
    # ===== LOAD_ARTIFACT TESTS =====
    # ===============================
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "path",
        [
            "marathon",
            "games/marathon",
            "games/genres/extraction/marathon",
        ],
    )
    def test_load_reads_data_from_file(self, client, path):
        expected_data = b"Destiny walked so I could run."
        path = client.base_dir / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected_data)

        actual_data = client.load(path)

        assert isinstance(actual_data, bytes)
        assert actual_data == expected_data

    def test_load_returns_none_if_no_file_was_found(self, client):
        data = client.load("a/fake/path")

        assert data is None

    def test_load_returns_empty_bytes_if_file_is_empty(self, client):
        path = "marathon"
        fullpath = client.base_dir / path
        fullpath.touch()

        data = client.load(path)

        assert len(data) == 0

    def test_load_raises_error_if_error_occurrs_while_reading_file(self, client):
        path = "restrictedfile.txt"
        fullpath = client.base_dir / path
        fullpath.touch(0o366, exist_ok=False)

        with pytest.raises(StorageError):
            client.load(path)
