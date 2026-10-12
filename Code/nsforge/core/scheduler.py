"""Domain-agnostic staged weight scheduling."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Mapping, Sequence


@dataclass
class StagedWeightScheduler:
    """Activate named constraints over stages with warmup and adaptive release.

    Implements the paper's Equation 9: before activation the weight is zero;
    after activation it linearly ramps to half of the final weight over
    `ramp_steps`, and the remaining half is released proportionally to the
    decrease of the rolling energy mean relative to its ramp-end snapshot,
    reaching the full weight once the rolling mean drops to
    `energy_threshold`. Constraints whose energies are never recorded via
    `record_energy` fall back to full release at ramp end.
    """

    boundaries: Sequence[int]
    final_weights: Mapping[str, float]
    activation_stage: Mapping[str, int]
    ramp_steps: int = 0
    frozen_until: Dict[str, int] = field(default_factory=dict)
    energy_window: int = 20
    energy_threshold: float = 0.03
    energy_means: Dict[str, float] = field(default_factory=dict)
    reference_energy: Dict[str, float] = field(default_factory=dict)
    _energy_history: Dict[str, Deque[float]] = field(
        default_factory=dict, init=False, repr=False
    )

    def stage_at(self, step: int) -> int:
        for idx, boundary in enumerate(self.boundaries, start=1):
            if step <= boundary:
                return idx
        return len(self.boundaries) + 1

    def _stage_start(self, stage: int) -> int:
        if stage <= 1:
            return 1
        return int(self.boundaries[stage - 2]) + 1

    def _ramp(self, step: int, stage: int) -> float:
        if self.ramp_steps <= 0:
            return 1.0
        elapsed = step - self._stage_start(stage)
        if elapsed < 0:
            return 0.0
        return min(1.0, elapsed / max(1, self.ramp_steps))

    def _ramp_end(self, name: str) -> int:
        stage = int(self.activation_stage.get(name, 1))
        return self._stage_start(stage) + max(1, self.ramp_steps)

    def record_energy(self, name: str, value: float, step: int) -> None:
        """Record a per-step energy for the rolling mean and ramp-end snapshot."""
        history = self._energy_history.setdefault(
            name, deque(maxlen=max(1, self.energy_window))
        )
        history.append(float(value))
        mean = sum(history) / len(history)
        self.energy_means[name] = mean
        if name not in self.reference_energy and step >= self._ramp_end(name):
            self.reference_energy[name] = mean

    def _adaptive_release(self, name: str) -> float:
        """The `g` factor of Equation 9; full release when untracked."""
        if name not in self.reference_energy or name not in self.energy_means:
            return 1.0
        reference = self.reference_energy[name]
        span = reference - self.energy_threshold
        if span <= 0.0:
            return 1.0
        decrease = max(0.0, reference - self.energy_means[name])
        return min(1.0, decrease / span)

    def weights_at(self, step: int) -> Dict[str, float]:
        stage = self.stage_at(step)
        weights: Dict[str, float] = {}
        for name, final in self.final_weights.items():
            active_stage = int(self.activation_stage.get(name, 1))
            if stage < active_stage:
                weights[name] = 0.0
                continue
            ramp = self._ramp(step, active_stage)
            if ramp < 1.0:
                weight = 0.5 * float(final) * ramp
            else:
                weight = 0.5 * float(final) * (1.0 + self._adaptive_release(name))
            if step < self.frozen_until.get(name, 0):
                weight *= 0.5
            weights[name] = weight
        return weights

    def freeze(self, name: str, until_step: int) -> None:
        self.frozen_until[name] = int(until_step)

