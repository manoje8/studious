import os
import secrets

import logfire
from dotenv import load_dotenv

load_dotenv()

_WEAK_KEY_SENTINEL = "change-me-in-production"


class ConfigApi:
    MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", 10)) * 1024 * 1024
    INGESTION_ROOT: str = os.getenv("INGESTION_ROOT", "/var/home")
    SECRET_KEY: str = os.getenv("SECRET_KEY", secrets.token_hex(32))
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCOUNTS: list[dict] = os.getenv("ACCOUNTS", [])

    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"

    RATE_LIMIT_QUERY_MAX_REQUESTS: int = int(os.getenv("RATE_LIMIT_QUERY_MAX_REQUESTS", 30))
    RATE_LIMIT_QUERY_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_QUERY_WINDOW_SECONDS", 60))

    RATE_LIMIT_INGESTION_MAX_REQUESTS: int = int(os.getenv("RATE_LIMIT_INGESTION_MAX_REQUESTS", 20))
    RATE_LIMIT_INGESTION_WINDOW_SECONDS: int = int(
        os.getenv("RATE_LIMIT_INGESTION_WINDOW_SECONDS", 60)
    )

    RATE_LIMIT_LOGIN_MAX_REQUESTS: int = int(os.getenv("RATE_LIMIT_LOGIN_MAX_REQUESTS", 10))
    RATE_LIMIT_LOGIN_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_LOGIN_WINDOW_SECONDS", 60))

    def validate(self) -> None:
        if not self.SECRET_KEY or self.SECRET_KEY == _WEAK_KEY_SENTINEL:
            logfire.warn(
                "SECURITY WARNING: SECRET_KEY is {key_status} "
                "Set a strong random value in your environment before going to production.",
                key_status="empty"
                if not self.SECRET_KEY
                else f"the example sentinel '{_WEAK_KEY_SENTINEL}'",
            )


config_api = ConfigApi()
