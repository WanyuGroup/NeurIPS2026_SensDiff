"""Objective helpers for combining neuro-symbolic energies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

import torch

EnergyMap = Mapping[str, torch.Tensor]


def as_float_dict(energies: EnergyMap) -> Dict[str, float]:
    """Detach an energy mapping for logging."""
    out: Dict[str, float] = {}
    for name, value in energies.items():
        if torch.is_tensor(value):
            out[name] = float(value.detach().mean().cpu())
        else:
            out[name] = float(value)
    return out


def zero_like_energy(energies: EnergyMap) -> torch.Tensor:
    """Return a scalar zero on the same device as the first energy."""
    for value in energies.values():
        if torch.is_tensor(value):
            return value.sum() * 0.0
    return torch.tensor(0.0)


@dataclass
class WeightedObjective:
    """Weighted sum over a dictionary of differentiable energy terms.

    The class is intentionally domain-agnostic: it only assumes each active
    energy is a PyTorch tensor and leaves naming conventions to domain packs.
    """

    weights: Mapping[str, float] = field(default_factory=dict)
    default_weight: float = 0.0
    normalize_by_active_weight: bool = False

    def weight_for(self, name: str) -> float:
        return float(self.weights.get(name, self.default_weight))

    def active_weights(self, energies: EnergyMap) -> Dict[str, float]:
        return {
            name: self.weight_for(name)
            for name in energies
            if self.weight_for(name) != 0.0
        }

    def total(self, energies: EnergyMap) -> torch.Tensor:
        total = None
        weight_sum = 0.0
        for name, energy in energies.items():
            weight = self.weight_for(name)
            if weight == 0.0:
                continue
            term = energy * weight
            total = term if total is None else total + term
            weight_sum += abs(weight)

        if total is None:
            return zero_like_energy(energies)

        if self.normalize_by_active_weight and weight_sum > 0.0:
            total = total / weight_sum
        return total

    __call__ = total

