from src.common.storage.base_storage import BaseStorage
from src.common.utils.constants import StorageType


class StorageFactory:
    """
    Config examples:
        {"type": "local", "base_dir": "./rag_storage"}
        {"type": "gcs",   "bucket": "my-rag-bucket", "prefix": "embeddings"}
    """

    @staticmethod
    def create(config: dict) -> BaseStorage:
        storage_type = config.get("type", "local").lower()

        if storage_type == StorageType.LOCAL.value:
            from src.common.storage.local_storage import LocalStorage

            return LocalStorage(base_path=config["base_dir"])
        elif storage_type == StorageType.GCS.value:
            from src.common.storage.gcp_storage import GoogleCloudStorage

            return GoogleCloudStorage(bucket_name=config["bucket"], prefix=config.get("prefix", ""))
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")
