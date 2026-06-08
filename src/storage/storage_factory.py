from src.storage.base_storage import BaseStorage


class StorageFactory:
    """
    Config examples:
        {"type": "local", "base_dir": "./rag_storage"}
        {"type": "gcs",   "bucket": "my-rag-bucket", "prefix": "embeddings"}
    """

    @staticmethod
    def create(config: dict) -> BaseStorage:
        storage_type = config.get("type", "local").lower()

        if storage_type == "local":
            from src.storage.local_storage import LocalStorage

            return LocalStorage(base_path=config["base_dir"])
        elif storage_type == "gcs":
            from src.storage.gcp_storage import GoogleCloudStorage

            return GoogleCloudStorage(
                bucket_name=config["bucket"], prefix=config.get("prefix", "")
            )
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")
