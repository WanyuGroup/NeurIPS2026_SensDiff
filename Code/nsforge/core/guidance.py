"""Generic gradient guidance for training-time losses and inference refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Union

import torch

from .objective import EnergyMap, WeightedObjective, as_float_dict

EnergyFn = Callable[[torch.Tensor], Union[torch.Tensor, EnergyMap]]
PostprocessFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class GuidanceResult:
    value: torch.Tensor
    energies: Mapping[str, torch.Tensor]
    total_energy: torch.Tensor
    steps: int
    finite: bool = True

    def log_dict(self) -> Mapping[str, float]:
        return as_float_dict(self.energies)


class GradientGuidance:
    """Run gradient descent on any differentiable symbolic energy.

    `energy_fn` owns the domain-specific decoding and constraint evaluation.
    The guidance loop only handles autograd, gradient normalization, masking,
    optional postprocessing, and diagnostics.
    """

    def __init__(
        self,
        objective: Optional[WeightedObjective] = None,
        step_size: float = 1.0,
        steps: int = 1,
        normalize_grad: bool = True,
        eps: float = 1e-8,
    ):
        self.objective = objective
        self.step_size = step_size
        self.steps = steps
        self.normalize_grad = normalize_grad
        self.eps = eps

    def _coerce_energies(self, energies: Union[torch.Tensor, EnergyMap]) -> EnergyMap:
        if torch.is_tensor(energies):
            return {"energy": energies}
        return energies

    def _total(self, energies: EnergyMap) -> torch.Tensor:
        if self.objective is not None:
            total = self.objective.total(energies)
        else:
            total = None
            for energy in energies.values():
                total = energy if total is None else total + energy
            if total is None:
                total = torch.tensor(0.0)
        return total.sum() if total.ndim > 0 else total

    def _normalize(self, grad: torch.Tensor) -> torch.Tensor:
        if not self.normalize_grad:
            return grad
        if grad.ndim <= 1:
            return grad / grad.norm().clamp(min=self.eps)
        view_shape = (grad.shape[0],) + (1,) * (grad.ndim - 1)
        norm = grad.reshape(grad.shape[0], -1).norm(dim=1).clamp(min=self.eps)
        return grad / norm.view(view_shape)

    def step(
        self,
        value: torch.Tensor,
        energy_fn: EnergyFn,
        postprocess: Optional[PostprocessFn] = None,
        mask: Optional[torch.Tensor] = None,
        step_size: Optional[float] = None,
        steps: Optional[int] = None,
    ) -> GuidanceResult:
        n_steps = self.steps if steps is None else steps
        eta = self.step_size if step_size is None else step_size

        if n_steps <= 0 or eta == 0.0:
            zero = value.sum() * 0.0
            return GuidanceResult(value=value, energies={"energy": zero}, total_energy=zero, steps=0)

        original = value
        dtype = value.dtype
        work = value.detach().float()
        last_energies: EnergyMap = {}
        last_total: torch.Tensor = work.sum() * 0.0

        for step_idx in range(n_steps):
            work.requires_grad_(True)
            energies = self._coerce_energies(energy_fn(work))
            total = self._total(energies)
            if not torch.isfinite(total):
                zero = original.sum() * 0.0
                return GuidanceResult(
                    value=original,
                    energies={name: energy.detach() for name, energy in energies.items()},
                    total_energy=zero,
                    steps=step_idx,
                    finite=False,
                )

            grad = torch.autograd.grad(total, work, only_inputs=True)[0]
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            if mask is not None:
                grad = grad * mask
            grad = self._normalize(grad)

            next_value = work - eta * grad
            if postprocess is not None:
                next_value = postprocess(next_value)

            last_energies = {name: energy.detach() for name, energy in energies.items()}
            last_total = total.detach()
            work = next_value.detach()

        return GuidanceResult(
            value=work.to(dtype),
            energies=last_energies,
            total_energy=last_total,
            steps=n_steps,
        )

