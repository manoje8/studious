from abc import ABC, abstractmethod
from typing import BinaryIO


class BaseStorage(ABC):
    @abstractmethod
    def upload(self, key: str, data: BinaryIO, metadata: dict = None) -> str:
        """Upload a file, return a URI or path"""

    @abstractmethod
    def download(self, key: str) -> BinaryIO:
        """Download a file by key"""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a file by key"""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists"""

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]:
        """List all keys with a given prefix"""
