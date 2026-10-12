"""Molecular NS4 curriculum schedule."""

from __future__ import annotations

from collections import deque

from nsforge.core.scheduler import StagedWeightScheduler


class NS4Scheduler:
    """Backward-compatible scheduler for the four molecular NS4 energies."""

    def __init__(
        self,
        stage_epochs=(500, 1400, 2200),
        lambda_clash=4.0,
        lambda_bond=3.0,
        lambda_angle=2.0,
        lambda_val=1.0,
        alpha_max=1.0,
        ramp_epochs=100,
        energy_window=20,
        energy_threshold=0.03,
    ):
        self.stage_epochs = tuple(stage_epochs)
        self.lambda_final = {
            "clash": lambda_clash,
            "bond": lambda_bond,
            "angle": lambda_angle,
            "val": lambda_val,
        }
        self.energy_weights = {
            "E_clash_nonbonded": lambda_clash,
            "E_bond_len_posterior": lambda_bond,
            "E_angle_posterior": lambda_angle,
            "E_val_posterior": lambda_val,
        }
        self.alpha_max = alpha_max
        self.ramp_epochs = ramp_epochs
        self._loss_history = {"clash": [], "bond": [], "angle": [], "val": []}
        self._frozen_until = {"clash": 0, "bond": 0, "angle": 0, "val": 0}
        self._generic = StagedWeightScheduler(
            boundaries=self.stage_epochs,
            final_weights={
                "clash": lambda_clash,
                "bond": lambda_bond,
                "angle": lambda_angle,
                "val": lambda_val,
            },
            activation_stage={"clash": 1, "bond": 2, "angle": 3, "val": 4},
            ramp_steps=ramp_epochs,
            energy_window=energy_window,
            energy_threshold=energy_threshold,
        )

    def get_stage(self, epoch):
        return self._generic.stage_at(epoch)

    def get_lambdas(self, epoch):
        self._generic.frozen_until = self._frozen_until
        return self._generic.weights_at(epoch)

    def get_energy_weights(self, epoch):
        lambdas = self.get_lambdas(epoch)
        return {
            "E_clash_nonbonded": lambdas["clash"],
            "E_bond_len_posterior": lambdas["bond"],
            "E_angle_posterior": lambdas["angle"],
            "E_val_posterior": lambdas["val"],
        }

    def get_alpha_t(self, t, T):
        return self.alpha_max * (1.0 - t / T) ** 2

    def check_stability(self, epoch, constraint_losses):
        window = 20
        freeze_duration = 50
        spike_factor = 2.0

        for key in ["clash", "bond", "angle", "val"]:
            val = constraint_losses.get(key, 0.0)
            if val == 0.0:
                continue
            self._loss_history[key].append(val)
            self._generic.record_energy(key, val, epoch)
            history = self._loss_history[key]
            if len(history) >= window:
                moving_avg = sum(history[-window:]) / window
                if val > spike_factor * moving_avg and epoch > self._frozen_until[key]:
                    self._frozen_until[key] = epoch + freeze_duration
                    print(
                        f"[NS4 Scheduler] Freezing {key} weight until "
                        f"{self._frozen_until[key]} due to spike "
                        f"({val:.4f} vs avg {moving_avg:.4f})"
                    )

    def state_dict(self):
        return {
            "loss_history": self._loss_history,
            "frozen_until": self._frozen_until,
            "energy_means": dict(self._generic.energy_means),
            "reference_energy": dict(self._generic.reference_energy),
        }

    def load_state_dict(self, d):
        self._loss_history = d["loss_history"]
        self._frozen_until = d["frozen_until"]
        self._generic.energy_means = d.get("energy_means", {})
        self._generic.reference_energy = d.get("reference_energy", {})
        for name, mean in self._generic.energy_means.items():
            history = self._generic._energy_history.setdefault(
                name, deque(maxlen=max(1, self._generic.energy_window))
            )
            history.clear()
            history.append(mean)

