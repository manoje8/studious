import io
import shutil
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

from src.storage.base_storage import BaseStorage


class LocalStorage(BaseStorage):
    def __init__(self, base_path):
        self.base = Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        resolved = (self.base / key).resolve()

        if not str(resolved).startswith(str(self.base.resolve())):
            raise ValueError(f"Key '{key}' escapes the storage root")

        return resolved

    def upload(self, key: str, data, metadata: dict = None) -> str:
        import json

        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, io.IOBase):
            with open(target, "wb") as f:
                shutil.copyfileobj(data, f)
        elif isinstance(data, list):
            payload = json.dumps([asdict(c) for c in data]).encode("utf-8")
            with open(target, "wb") as f:
                shutil.copyfileobj(BytesIO(payload), f)
        else:
            raise ValueError(
                f"Unsupported data type: {type(data)}. " "Expected BinaryIO or list."
            )

        if metadata is not None:
            meta_path = target.with_suffix(target.suffix + ".meta")
            meta_path.write_text(json.dumps(metadata))

        return str(target)

    def download(self, key: str):
        target = self._resolve(key)
        if not target.exists():
            raise FileNotFoundError(f"Key '{key}' not found")

        return open(target, "rb")

    def delete(self, key: str):
        target = self._resolve(key)

        if target.exists():
            target.unlink()

            meta = target.with_suffix(target.suffix + ".meta")
            if meta.exists():
                meta.unlink()

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def list(self, prefix: str = "") -> list[str]:
        search_root = self._resolve(prefix) if prefix else self.base

        if not search_root.exists():
            return []

        return [
            str(p.relative_to(self.base))
            for p in search_root.rglob("*")
            if p.is_file() and not p.suffix == ".meta"
        ]
