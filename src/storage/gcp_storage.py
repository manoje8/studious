from io import BytesIO
from typing import BinaryIO

from src.storage.base_storage import BaseStorage
from src.utils.config import config


class GoogleCloudStorage(BaseStorage):
    def __init__(self, bucket_name: str, prefix: str = ""):
        from google.cloud import storage as gcs

        self.client = gcs.Client(project=config.PROJECT_ID)
        self.bucket = self.client.bucket(bucket_name)
        self.prefix = prefix.rstrip("/")

    def _full_key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def upload(self, key: str, data: BinaryIO, metadata: dict = None) -> str:
        blob = self.bucket.blob(self._full_key(key))

        if metadata:
            blob.metadata = metadata

        blob.upload_from_file(data)

        return f"gs://{self.bucket.name}/{blob.name}"

    def download(self, key: str) -> BinaryIO:
        blob = self.bucket.blob(self._full_key(key))

        if not blob.exists():
            raise FileNotFoundError(f"Key '{key}' not found")

        buffer = BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)

        return buffer

    def delete(self, key: str) -> None:
        blob = self.bucket.blob(self._full_key(key))

        if blob.exists():
            blob.delete()

    def exists(self, key: str) -> bool:
        return self.bucket.blob(self._full_key(key)).exists()

    def list(self, prefix: str = "") -> list[str]:
        full_prefix = (
            self._full_key(prefix)
            if prefix
            else (self.prefix + "/" if self.prefix else "")
        )
        blobs = self.bucket.list_blobs(prefix=full_prefix)
        strip = (self.prefix + "/") if self.prefix else ""

        return [
            b.name[len(strip) :] if b.name.startswith(strip) else b.name for b in blobs
        ]
