"""Small generic building blocks for symbolic constraint sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Protocol

import torch


class Constraint(Protocol):
    name: str

    def __call__(self, state: Any, **kwargs: Any) -> torch.Tensor:
        ...


@dataclass
class FunctionConstraint:
    name: str
    fn: Callable[..., torch.Tensor]

    def __call__(self, state: Any, **kwargs: Any) -> torch.Tensor:
        return self.fn(state, **kwargs)


class ConstraintSet:
    """Evaluate a named collection of arbitrary constraints."""

    def __init__(self, constraints: Iterable[Constraint]):
        self.constraints = list(constraints)

    def __call__(
        self,
        state: Any,
        active: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> Mapping[str, torch.Tensor]:
        active_set = set(active) if active is not None else None
        energies: Dict[str, torch.Tensor] = {}
        for constraint in self.constraints:
            if active_set is not None and constraint.name not in active_set:
                continue
            energies[constraint.name] = constraint(state, **kwargs)
        return energies

