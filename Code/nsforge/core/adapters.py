"""Protocols for connecting arbitrary models to nsforge."""

from __future__ import annotations

from typing import Any, Protocol

import torch


class Decoder(Protocol):
    """Decode a model-internal variable into a constraint state."""

    def __call__(self, value: torch.Tensor, **kwargs: Any) -> Any:
        ...


class Postprocessor(Protocol):
    """Repair model-specific invariants after a guidance step."""

    def __call__(self, value: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        ...


class RefinementAdapter(Protocol):
    """Minimal adapter needed for inference-time refinement."""

    def decode(self, value: torch.Tensor, **kwargs: Any) -> Any:
        ...

    def postprocess(self, value: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        ...

