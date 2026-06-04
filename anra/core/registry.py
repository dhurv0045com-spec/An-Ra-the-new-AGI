from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Generic, TypeVar, cast

from anra.core.protocols import (
    IdentityModuleProtocol,
    InferenceStrategyProtocol,
    MemoryTierProtocol,
    ModelProtocol,
    ObjectiveProtocol,
    TrainerProtocol,
)

T = TypeVar("T")


class Registry(Generic[T]):
    """Typed registry for swappable AN-RA components."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: dict[str, type[T]] = {}

    def register(
        self,
        name: str,
        *,
        aliases: Sequence[str] | None = None,
        replace: bool = False,
    ) -> Callable[[type[T]], type[T]]:
        """Register a class under a canonical name and optional aliases."""

        def decorator(cls: type[T]) -> type[T]:
            names = [name, *(aliases or [])]
            duplicates = [
                registered_name for registered_name in names if registered_name in self._registry
            ]
            if duplicates and not replace:
                joined = ", ".join(sorted(duplicates))
                raise KeyError(f"{self._name} registration already exists: {joined}")
            for registered_name in names:
                self._registry[registered_name] = cls
            return cls

        return decorator

    def build(self, name: str, **kwargs: object) -> T:
        """Instantiate a registered class, ignoring unknown keyword arguments."""

        if name not in self._registry:
            available = ", ".join(self.list()) or "<none>"
            raise KeyError(f"Unknown {self._name}: '{name}'. Available: {available}")

        cls = self._registry[name]
        signature = inspect.signature(cls.__init__)
        accepts_var_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_var_kwargs:
            valid_kwargs = kwargs
        else:
            valid_kwargs = {
                key: value for key, value in kwargs.items() if key in signature.parameters
            }
        return cast(T, cls(**valid_kwargs))

    def get(self, name: str) -> type[T]:
        """Return the registered class for a name."""

        if name not in self._registry:
            available = ", ".join(self.list()) or "<none>"
            raise KeyError(f"Unknown {self._name}: '{name}'. Available: {available}")
        return self._registry[name]

    def contains(self, name: str) -> bool:
        """Return whether a name is registered."""

        return name in self._registry

    def list(self) -> list[str]:
        """Return all registered names and aliases."""

        return sorted(self._registry)


MODEL_REGISTRY: Registry[ModelProtocol] = Registry("model")
MEMORY_REGISTRY: Registry[MemoryTierProtocol] = Registry("memory_tier")
TRAINING_REGISTRY: Registry[TrainerProtocol] = Registry("trainer")
OBJECTIVE_REGISTRY: Registry[ObjectiveProtocol] = Registry("objective")
INFERENCE_REGISTRY: Registry[InferenceStrategyProtocol] = Registry("inference_strategy")
IDENTITY_REGISTRY: Registry[IdentityModuleProtocol] = Registry("identity_module")
