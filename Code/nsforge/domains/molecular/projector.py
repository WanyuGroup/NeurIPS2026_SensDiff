"""Molecular guidance adapters, including the original NS4 latent projector."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from nsforge.core.guidance import GradientGuidance
from nsforge.core.objective import WeightedObjective
from nsforge.core.tensors import remove_mean_with_mask
from nsforge.domains.molecular.constraints import four_constraint_energies
from nsforge.domains.molecular.state import MolecularState


class MolecularGradientGuidance:
    """Model-agnostic molecular refinement over any differentiable variable."""

    def __init__(
        self,
        decode: Callable[..., MolecularState],
        bond_provider: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        stats,
        objective: WeightedObjective,
        postprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        stage: int = 4,
        steps: int = 1,
        step_size: float = 1.0,
        normalize_grad: bool = True,
    ):
        self.decode = decode
        self.bond_provider = bond_provider
        self.stats = stats
        self.objective = objective
        self.postprocess = postprocess
        self.stage = stage
        self.guidance = GradientGuidance(
            objective=objective,
            step_size=step_size,
            steps=steps,
            normalize_grad=normalize_grad,
        )

    def project(self, value: torch.Tensor, **decode_kwargs):
        def energy_fn(current):
            state = self.decode(current, **decode_kwargs)
            if not isinstance(state, MolecularState):
                R, A_soft = state
                state = MolecularState(R, A_soft, decode_kwargs["node_mask"])
            B_soft = state.bond_probs
            if B_soft is None:
                B_soft = self.bond_provider(state.R, state.A_soft, state.node_mask)
            return four_constraint_energies(
                state.R,
                state.A_soft,
                B_soft,
                self.stats,
                state.node_mask,
                self.stage,
            )

        return self.guidance.step(value, energy_fn, postprocess=self.postprocess)


class FourConstraintProjector:
    """Backward-compatible GeoLDM-style latent projector.

    This adapter keeps the old `ns4.projector_four_constraints` contract while
    using the generic nsforge guidance loop internally.
    """

    def __init__(
        self,
        model,
        bond_posterior_head,
        stats,
        lambda_clash=4.0,
        lambda_bond=3.0,
        lambda_angle=2.0,
        lambda_val=1.0,
        alpha_max=1.0,
        proj_steps=3,
        stage=4,
        eps=1e-8,
    ):
        self.model = model
        self.bond_posterior_head = bond_posterior_head
        self.stats = stats
        self.lambdas = {
            "clash": lambda_clash,
            "bond": lambda_bond,
            "angle": lambda_angle,
            "val": lambda_val,
        }
        self.objective = WeightedObjective(
            {
                "E_clash_nonbonded": lambda_clash,
                "E_bond_len_posterior": lambda_bond,
                "E_angle_posterior": lambda_angle,
                "E_val_posterior": lambda_val,
            }
        )
        self.alpha_max = alpha_max
        self.proj_steps = proj_steps
        self.stage = stage
        self.eps = eps

    def _zero_stats(self, device):
        zero = torch.tensor(0.0, device=device)
        return {
            "E_clash_nonbonded": zero,
            "E_bond_len_posterior": zero,
            "E_angle_posterior": zero,
            "E_val_posterior": zero,
        }

    def alpha_t(self, step_idx):
        return self.alpha_max * (1.0 - step_idx / self.model.T) ** 2

    def project(self, z, node_mask, edge_mask, context, step_idx):
        if self.proj_steps <= 0:
            return z, self._zero_stats(z.device)

        step_alpha = self.alpha_t(step_idx)
        if step_alpha <= 0:
            return z, self._zero_stats(z.device)

        def energy_fn(z_work):
            R, A_soft = self.model.decode_soft(z_work, node_mask, edge_mask, context)
            B_soft = self.bond_posterior_head(R, A_soft, node_mask)
            return four_constraint_energies(R, A_soft, B_soft, self.stats, node_mask, self.stage)

        def postprocess(z_next):
            x_next = remove_mean_with_mask(z_next[:, :, : self.model.n_dims], node_mask)
            h_next = z_next[:, :, self.model.n_dims :] * node_mask
            return torch.cat([x_next, h_next], dim=2)

        guidance = GradientGuidance(
            objective=self.objective,
            step_size=step_alpha / self.proj_steps,
            steps=self.proj_steps,
            normalize_grad=True,
            eps=self.eps,
        )
        result = guidance.step(z, energy_fn, postprocess=postprocess, mask=node_mask)
        if not result.finite:
            return z, self._zero_stats(z.device)
        return result.value, result.energies

