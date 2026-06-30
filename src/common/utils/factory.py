from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Literal, TypeVar

from src.common.utils.hasher import hash_data

T = TypeVar("T", covariant=True)

ServiceScope = Literal["singleton", "transient"]


@dataclass
class _ServiceDescriptor(Generic[T]):
    scope: ServiceScope
    initializer: Callable[..., T]


class Factory(ABC, Generic[T]):
    _instance: ClassVar["Factory | None"] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "Factory[T]":
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._service_initializers: dict[str, _ServiceDescriptor[T]] = {}
            self._initialized_services: dict[str, T] = {}
            self._initialized = True

    def __contains__(self, strategy: str) -> bool:
        return strategy in self._service_initializers

    def keys(self) -> list[str]:
        return list(self._service_initializers.keys())

    def register(
        self,
        strategy: str,
        initializer: Callable[..., T],
        scope: ServiceScope = "transient",
    ) -> None:
        self._service_initializers[strategy] = _ServiceDescriptor(
            scope=scope, initializer=initializer
        )

    def create(self, strategy: str, init_args: dict[str, Any] | None = None) -> T:
        if strategy not in self._service_initializers:
            msg = f"Strategy '{strategy}' is not registered. Registered strategies are: {', '.join(list(self._service_initializers.keys()))}"
            raise ValueError(msg)

        init_args = {k: v for k, v in (init_args or {}).items() if v is not None}

        service_descriptor = self._service_initializers[strategy]

        if service_descriptor.scope == "singleton":
            cache_key = hash_data(
                {
                    "strategy": strategy,
                    "init_args": init_args,
                }
            )

            if cache_key not in self._initialized_services:
                self._initialized_services[cache_key] = service_descriptor.initializer(**init_args)

            return self._initialized_services[cache_key]

        return service_descriptor.initializer(**(init_args or {}))
