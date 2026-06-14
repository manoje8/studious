import json
import pytest
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from src.storage.local_storage import LocalStorage
from src.storage.gcp_storage import GoogleCloudStorage
from src.storage.storage_factory import StorageFactory
from src.utils.constants import StorageType


# Test dataclass for list uploads
@dataclass
class TestData:
    id: int
    name: str
    value: float


class TestLocalStorage:
    """Tests for LocalStorage implementation"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def storage(self, temp_dir):
        """Create LocalStorage instance"""
        return LocalStorage(base_path=temp_dir)

    def test_upload_binary_data(self, storage, temp_dir):
        """Test uploading binary data"""
        key = "test"
        data = BytesIO(b"Hello, World!")

        uri = storage.upload(key, data)

        assert uri == str(temp_dir / key)
        assert (temp_dir / key).exists()
        assert (temp_dir / key).read_bytes() == b"Hello, World!"

    def test_upload_list_data(self, storage, temp_dir):
        """Test uploading list of dataclass objects"""
        key = "test"
        test_list = [
            TestData(id=1, name="item1", value=10.5),
            TestData(id=2, name="item2", value=20.5),
        ]

        uri = storage.upload(key, test_list)

        assert uri == str(temp_dir / key)
        assert (temp_dir / key).exists()

        # Verify content
        content = json.loads((temp_dir / key).read_bytes())
        assert len(content) == 2
        assert content[0]["id"] == 1
        assert content[0]["name"] == "item1"

    def test_upload_with_metadata(self, storage, temp_dir):
        """Test uploading with metadata"""
        key = "test"
        data = BytesIO(b"test data")
        metadata = {"author": "test", "version": "1.0"}

        uri = storage.upload(key, data, metadata)  # noqa

        meta_path = temp_dir / (key + ".meta")
        assert meta_path.exists()
        saved_metadata = json.loads(meta_path.read_text())
        assert saved_metadata == metadata

    def test_upload_unsupported_type(self, storage):
        """Test uploading unsupported data type"""
        with pytest.raises(ValueError, match="Unsupported data type"):
            storage.upload("test.txt", "string data")

    def test_download_existing_file(self, storage, temp_dir):
        """Test downloading existing file"""
        key = "test/download.txt"
        test_content = b"Download me!"
        (temp_dir / key).parent.mkdir(parents=True, exist_ok=True)
        (temp_dir / key).write_bytes(test_content)

        result = storage.download(key)

        assert isinstance(result, BytesIO) or hasattr(result, "read")
        content = result.read()
        assert content == test_content

    def test_download_nonexistent_file(self, storage):
        """Test downloading non-existent file"""
        with pytest.raises(FileNotFoundError, match="Key 'nonexistent.txt' not found"):
            storage.download("nonexistent.txt")

    def test_delete_existing_file(self, storage, temp_dir):
        """Test deleting existing file"""
        key = "test/delete.txt"
        (temp_dir / "test").mkdir(parents=True, exist_ok=True)
        (temp_dir / key).write_bytes(b"to delete")
        meta_path = temp_dir / (key + ".meta")
        meta_path.write_text("{}")

        assert (temp_dir / key).exists()
        storage.delete(key)

        assert not (temp_dir / key).exists()
        assert not meta_path.exists()

    def test_delete_nonexistent_file(self, storage):
        """Test deleting non-existent file (should not error)"""
        storage.delete("nonexistent.txt")  # Should not raise exception

    def test_exists(self, storage, temp_dir):
        """Test checking file existence"""
        key = "test/exists.txt"

        assert not storage.exists(key)

        (temp_dir / "test").mkdir(parents=True, exist_ok=True)
        (temp_dir / key).write_bytes(b"data")
        assert storage.exists(key)

    def test_list_without_prefix(self, storage, temp_dir):
        """Test listing all files"""
        files = ["a.txt", "b/c.txt", "b/d/e.txt"]
        for file in files:
            (temp_dir / file).parent.mkdir(parents=True, exist_ok=True)
            (temp_dir / file).write_bytes(b"data")

        listed = storage.list()
        listed_set = set(listed)

        assert listed_set == set(files)

    def test_list_with_prefix(self, storage, temp_dir):
        """Test listing files with prefix"""
        files = {
            "prefix/a.txt": b"data",
            "prefix/b/c.txt": b"data",
            "other/d.txt": b"data",
        }
        for path, content in files.items():
            (temp_dir / path).parent.mkdir(parents=True, exist_ok=True)
            (temp_dir / path).write_bytes(content)

        listed = storage.list(prefix="prefix")

        assert "prefix/a.txt" in listed
        assert "prefix/b/c.txt" in listed
        assert "other/d.txt" not in listed

    def test_list_ignores_meta_files(self, storage, temp_dir):
        """Test that list() doesn't return .meta files"""
        (temp_dir / "data.txt").write_bytes(b"data")
        (temp_dir / "data.txt.meta").write_text("{}")

        listed = storage.list()

        assert "data.txt" in listed
        assert "data.txt.meta" not in listed

    def test_path_traversal_prevention(self, storage):
        """Test that path traversal attacks are prevented"""
        with pytest.raises(ValueError, match="escapes the storage root"):
            storage._resolve("../../etc/passwd")


class TestGoogleCloudStorage:
    """Tests for GoogleCloudStorage implementation"""

    @pytest.fixture
    def mock_gcs(self):
        """Mock Google Cloud Storage client and bucket"""
        with (
            patch("google.cloud.storage.Client") as mock_client_cls,
            patch("src.storage.gcp_storage.config") as mock_config,
        ):
            mock_config.PROJECT_ID = "test-project"
            mock_client = MagicMock()
            mock_bucket = MagicMock()
            mock_bucket.name = "test-bucket"

            # Track blobs by the key passed to blob()
            blobs: dict = {}

            def make_blob(key):
                if key not in blobs:
                    b = MagicMock()
                    b.name = key
                    blobs[key] = b
                return blobs[key]

            mock_bucket.blob.side_effect = make_blob
            mock_client_cls.return_value = mock_client
            mock_client.bucket.return_value = mock_bucket

            yield {
                "client": mock_client,
                "bucket": mock_bucket,
                "blobs": blobs,  # key -> MagicMock blob
                "get_blob": make_blob,  # helper: get_blob(full_key)
            }

    @pytest.fixture
    def storage(self, mock_gcs):
        """Create GoogleCloudStorage instance with mocked dependencies"""
        return GoogleCloudStorage(bucket_name="test-bucket", prefix="test-prefix")

    def test_upload_binary_data(self, storage, mock_gcs):
        """Test uploading binary data to GCS"""
        key = "file.txt"
        data = BytesIO(b"test data")

        uri = storage.upload(key, data)

        mock_gcs["bucket"].blob.assert_called_with("test-prefix/file.txt")
        blob = mock_gcs["get_blob"]("test-prefix/file.txt")
        blob.upload_from_file.assert_called_with(data)
        assert uri == "gs://test-bucket/test-prefix/file.txt"

    def test_upload_list_data(self, storage, mock_gcs):
        """Test uploading list data to GCS"""
        key = "data.json"
        test_list = [TestData(id=1, name="test", value=99.9)]

        uri = storage.upload(key, test_list)

        mock_gcs["bucket"].blob.assert_called_with("test-prefix/data.json")
        blob = mock_gcs["get_blob"]("test-prefix/data.json")
        call_args = blob.upload_from_file.call_args

        # Check that upload_from_file was called with BytesIO and content-type
        assert call_args[1]["content_type"] == "application/json"
        assert uri == "gs://test-bucket/test-prefix/data.json"

    def test_upload_with_metadata(self, storage, mock_gcs):
        """Test uploading with metadata to GCS"""
        key = "file.txt"
        data = BytesIO(b"test")
        metadata = {"key": "value"}

        storage.upload(key, data, metadata)

        blob = mock_gcs["get_blob"]("test-prefix/file.txt")
        assert blob.metadata == metadata

    def test_upload_unsupported_type(self, storage):
        """Test uploading unsupported data type"""
        with pytest.raises(ValueError, match="Unsupported data type"):
            storage.upload("test.txt", "string data")

    def test_download_existing_file(self, storage, mock_gcs):
        """Test downloading existing file from GCS"""
        key = "file.txt"
        blob = mock_gcs["get_blob"]("test-prefix/file.txt")
        blob.exists.return_value = True

        result = storage.download(key)

        mock_gcs["bucket"].blob.assert_called_with("test-prefix/file.txt")
        blob.download_to_file.assert_called_once()
        assert isinstance(result, BytesIO)

    def test_download_nonexistent_file(self, storage, mock_gcs):
        """Test downloading non-existent file from GCS"""
        key = "nonexistent.txt"
        blob = mock_gcs["get_blob"]("test-prefix/nonexistent.txt")
        blob.exists.return_value = False

        with pytest.raises(FileNotFoundError, match="Key 'nonexistent.txt' not found"):
            storage.download(key)

    def test_delete_existing_file(self, storage, mock_gcs):
        """Test deleting existing file from GCS"""
        key = "file.txt"
        blob = mock_gcs["get_blob"]("test-prefix/file.txt")
        blob.exists.return_value = True

        storage.delete(key)

        blob.delete.assert_called_once()

    def test_delete_nonexistent_file(self, storage, mock_gcs):
        """Test deleting non-existent file from GCS"""
        key = "nonexistent.txt"
        blob = mock_gcs["get_blob"]("test-prefix/nonexistent.txt")
        blob.exists.return_value = False

        storage.delete(key)  # Should not call delete

        blob.delete.assert_not_called()

    def test_exists(self, storage, mock_gcs):
        """Test checking file existence in GCS"""
        key = "file.txt"
        blob = mock_gcs["get_blob"]("test-prefix/file.txt")
        blob.exists.return_value = True

        assert storage.exists(key) is True

        blob.exists.return_value = False
        assert storage.exists(key) is False

    def test_list_without_prefix(self, storage, mock_gcs):
        """Test listing all files in GCS"""
        blob1 = MagicMock()
        blob1.name = "test-prefix/file1.txt"
        blob2 = MagicMock()
        blob2.name = "test-prefix/subdir/file2.txt"
        blob3 = MagicMock()
        blob3.name = "other-prefix/file3.txt"
        mock_gcs["bucket"].list_blobs.return_value = [blob1, blob2, blob3]

        listed = storage.list()

        assert "file1.txt" in listed
        assert "subdir/file2.txt" in listed
        assert "other-prefix/file3.txt" not in listed

    def test_list_with_prefix(self, storage, mock_gcs):
        """Test listing files with prefix in GCS"""
        storage.list(prefix="subdir/")

        mock_gcs["bucket"].list_blobs.assert_called_with(prefix="test-prefix/subdir/")

    def test_full_key_generation(self, storage):
        """Test full key generation with prefix"""
        assert storage._full_key("file.txt") == "test-prefix/file.txt"

        storage_no_prefix = GoogleCloudStorage(bucket_name="test", prefix="")
        assert storage_no_prefix._full_key("file.txt") == "file.txt"


class TestStorageFactory:
    """Tests for StorageFactory"""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_create_local_storage(self, temp_dir):
        """Test creating LocalStorage instance"""
        config = {"type": StorageType.LOCAL.value, "base_dir": temp_dir}

        storage = StorageFactory.create(config)

        assert isinstance(storage, LocalStorage)
        assert storage.base == Path(temp_dir)

    # def test_create_gcs_storage(self):
    #     """Test creating GoogleCloudStorage instance"""
    #     config = {
    #         "type": StorageType.GCS.value,
    #         "bucket": "test-bucket",
    #         "prefix": "test-prefix",
    #     }
    #
    #     with patch("src.storage.gcp_storage.config") as mock_config:
    #         mock_config.PROJECT_ID = "test-project"
    #         storage = StorageFactory.create(config)
    # assert isinstance(storage, GoogleCloudStorage)

    # def test_create_gcs_storage_without_prefix(self):
    #     """Test creating GCS storage without prefix"""
    #     config = {"type": StorageType.GCS.value, "bucket": "test-bucket"}
    #
    #     with patch("src.storage.gcp_storage.config") as mock_config:
    #         mock_config.PROJECT_ID = "test-project"
    #         storage = StorageFactory.create(config)
    #
    #     assert storage.prefix == ""

    def test_create_unknown_storage_type(self):
        """Test creating storage with unknown type"""
        config = {"type": "unknown", "bucket": "test"}

        with pytest.raises(ValueError, match="Unknown storage type: unknown"):
            StorageFactory.create(config)

    def test_create_with_missing_config(self, temp_dir):
        """Test creating local storage with missing base_dir"""
        config = {"type": StorageType.LOCAL.value}

        with pytest.raises(KeyError):
            StorageFactory.create(config)


class TestBaseStorageInterface:
    """Test that implementations adhere to BaseStorage interface"""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_local_storage_implements_all_methods(self, temp_dir):
        """Test LocalStorage implements all abstract methods"""
        storage = LocalStorage(temp_dir)

        # These should not raise AttributeError
        assert hasattr(storage, "upload")
        assert hasattr(storage, "download")
        assert hasattr(storage, "delete")
        assert hasattr(storage, "exists")
        assert hasattr(storage, "list")

        # All should be callable (even if they might raise other exceptions)
        assert callable(storage.upload)
        assert callable(storage.download)
        assert callable(storage.delete)
        assert callable(storage.exists)
        assert callable(storage.list)

    # def test_gcs_storage_implements_all_methods(self):
    #     """Test GoogleCloudStorage implements all abstract methods"""
    #     with patch("src.storage.gcp_storage.config") as mock_config:
    #         mock_config.PROJECT_ID = "test"
    #         storage = GoogleCloudStorage("test-bucket")
    #
    #     assert hasattr(storage, "upload")
    #     assert hasattr(storage, "download")
    #     assert hasattr(storage, "delete")
    #     assert hasattr(storage, "exists")
    #     assert hasattr(storage, "list")
    #
    #     assert callable(storage.upload)
    #     assert callable(storage.download)
    #     assert callable(storage.delete)
    #     assert callable(storage.exists)
    #     assert callable(storage.list)
