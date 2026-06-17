import hashlib
from collections.abc import Callable
from typing import Any
import yaml

Hasher = Callable[[str], str]
"""Type alias for a hasher function (data: str) -> str."""


def sha256_hasher(data: str) -> str:
    """Generate a SHA-256 hash for the input data."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def make_yaml_serializable(data: Any) -> Any:
    if isinstance(data, (list, tuple)):
        return [make_yaml_serializable(item) for item in data]
    if isinstance(data, set):
        return tuple(sorted(make_yaml_serializable(item) for item in data))
    if isinstance(data, dict):
        return tuple(
            sorted((key, make_yaml_serializable(value)) for key, value in data.items())
        )

    return str(data)


def hash_data(data: Any, *, hasher: Hasher | None = None) -> str:
    hasher = hasher or sha256_hasher
    try:
        return hasher(yaml.dump(data, sort_keys=True))
    except TypeError:
        return hasher(yaml.dump(make_yaml_serializable(data), sort_keys=True))
