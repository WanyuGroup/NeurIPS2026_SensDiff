"""Molecular NS4 constraint energies.

These functions are a reusable domain pack: they only require soft molecular
states (`R`, `A_soft`, `B_soft`, `node_mask`) and a statistics object. They do
not depend on diffusion, GeoLDM, or a specific training loop.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch

from nsforge.core.objective import WeightedObjective
from nsforge.domains.molecular.state import MolecularState


MOLECULAR_NS4_ENERGY_NAMES = (
    "E_clash_nonbonded",
    "E_bond_len_posterior",
    "E_angle_posterior",
    "E_val_posterior",
)


def _pseudo_huber(u: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return delta**2 * (torch.sqrt(1.0 + (u / delta) ** 2) - 1.0)


def _pairwise_distances(R: torch.Tensor, eps: float = 1e-8):
    diff = R.unsqueeze(2) - R.unsqueeze(1)
    d_ij = diff.norm(dim=-1).clamp(min=eps)
    return d_ij, diff


def _pair_mask_triu(node_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    mask_flat = node_mask.squeeze(-1)
    n_nodes = mask_flat.shape[1]
    pair = mask_flat.unsqueeze(2) * mask_flat.unsqueeze(1)
    triu = torch.triu(torch.ones(n_nodes, n_nodes, device=device), diagonal=1)
    return pair * triu


def compute_clash_energy(
    R: torch.Tensor,
    A_soft: torch.Tensor,
    B_soft: torch.Tensor,
    stats,
    node_mask: torch.Tensor,
    eps: float = 1e-8,
    tau_c: float = 0.05,
) -> torch.Tensor:
    """Nonbonded atoms should not overlap."""
    d_ij, _ = _pairwise_distances(R, eps)
    d_min_ij = torch.einsum("bic,cd,bjd->bij", A_soft, stats.d_min, A_soft)
    s_clash = torch.sigmoid((d_ij - d_min_ij) / tau_c)

    pair_mask = _pair_mask_triu(node_mask, R.device)
    weights = B_soft[:, :, :, 0] * pair_mask
    Z_c = weights.sum(dim=(1, 2)).clamp(min=eps)
    return (weights * (-torch.log(s_clash + eps))).sum(dim=(1, 2)).div(Z_c).mean()


def compute_bond_len_energy(
    R: torch.Tensor,
    A_soft: torch.Tensor,
    B_soft: torch.Tensor,
    stats,
    node_mask: torch.Tensor,
    eps: float = 1e-8,
    delta: float = 1.0,
    sigma_min_b: float = 0.03,
) -> torch.Tensor:
    """Bond lengths should match empirical atom-pair/bond-order statistics."""
    d_ij, _ = _pairwise_distances(R, eps)
    mu = torch.einsum("bic,cdm,bjd->bijm", A_soft, stats.bond_mu, A_soft)
    sigma = torch.einsum("bic,cdm,bjd->bijm", A_soft, stats.bond_sigma, A_soft)
    sigma = sigma.clamp(min=sigma_min_b)

    z = (d_ij.unsqueeze(-1) - mu) / sigma
    s_bond = torch.exp(-_pseudo_huber(z, delta=delta))

    pair_mask = _pair_mask_triu(node_mask, R.device)
    weights = B_soft[:, :, :, 1:] * pair_mask.unsqueeze(-1)
    Z_b = weights.sum().clamp(min=eps)
    return (weights * (-torch.log(s_bond[:, :, :, 1:] + eps))).sum() / Z_b


def compute_angle_energy(
    R: torch.Tensor,
    A_soft: torch.Tensor,
    B_soft: torch.Tensor,
    stats,
    node_mask: torch.Tensor,
    eps: float = 1e-8,
    delta: float = 1.0,
    top_k: int = 4,
    sigma_min_a: float = 0.03,
) -> torch.Tensor:
    """Bond angles should match empirical central-atom/bond-order statistics."""
    batch_size, n_nodes, _ = R.shape
    n_bond_classes = B_soft.shape[-1]
    n_bonded_classes = n_bond_classes - 1
    device = R.device
    mask_flat = node_mask.squeeze(-1)

    bond_prob = B_soft[:, :, :, 1:].sum(dim=-1)
    bond_prob = bond_prob * mask_flat.unsqueeze(1) * mask_flat.unsqueeze(2)
    bond_prob = bond_prob * (1.0 - torch.eye(n_nodes, device=device))
    k = min(top_k, n_nodes - 1)
    _, top_idx = torch.topk(bond_prob, k=k, dim=2)

    idx3 = top_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    R_nbr = torch.gather(R.unsqueeze(1).expand(-1, n_nodes, -1, -1), 2, idx3)

    v = R_nbr - R.unsqueeze(2)
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=eps)
    cos_all = torch.einsum("bnjd,bnkd->bnjk", v, v).clamp(-1.0 + eps, 1.0 - eps)
    triu_k = torch.triu(torch.ones(k, k, device=device), diagonal=1)

    idx_g = top_idx.unsqueeze(-1).expand(-1, -1, -1, n_bond_classes)
    B_jn = torch.gather(B_soft, 2, idx_g)
    B_jn_b = B_jn[:, :, :, 1:]

    mu_fb = stats.angle_cos_mu_fallback
    sig_fb = stats.angle_cos_sigma_fallback
    mu_interp = torch.einsum("bnc,cmp->bnmp", A_soft, mu_fb)
    sig_interp = torch.einsum("bnc,cmp->bnmp", A_soft, sig_fb).clamp(min=sigma_min_a)
    mu_b = mu_interp[:, :, 1:, 1:]
    sig_b = sig_interp[:, :, 1:, 1:]

    w = B_jn_b.unsqueeze(3).unsqueeze(5) * B_jn_b.unsqueeze(2).unsqueeze(4)
    w = w * triu_k.view(1, 1, k, k, 1, 1) * mask_flat.view(batch_size, n_nodes, 1, 1, 1, 1)

    cos_exp = cos_all.unsqueeze(-1).unsqueeze(-1)
    mu_exp = mu_b.unsqueeze(2).unsqueeze(3)
    sig_exp = sig_b.unsqueeze(2).unsqueeze(3)

    z_val = (cos_exp - mu_exp) / sig_exp
    neg_log_s = -torch.log(torch.exp(-_pseudo_huber(z_val, delta=delta)) + eps)
    return (w * neg_log_s).sum() / w.sum().clamp(min=eps)


def compute_valence_energy(
    A_soft: torch.Tensor,
    B_soft: torch.Tensor,
    stats,
    node_mask: torch.Tensor,
    eps: float = 1e-8,
    sigma_v: float = 0.15,
) -> torch.Tensor:
    """Soft valence should fall in the allowed set for each atom type."""
    _, n_nodes, n_atom_types = A_soft.shape
    device = A_soft.device
    mask_flat = node_mask.squeeze(-1)

    nu = stats.bond_order_weights.to(device)
    weighted_B = (B_soft * nu.view(1, 1, 1, -1)).sum(dim=-1)
    weighted_B = weighted_B * (1.0 - torch.eye(n_nodes, device=device))
    v_soft = weighted_B.sum(dim=2)

    max_nv = max(stats.valence_sets[a].shape[0] for a in range(n_atom_types) if a in stats.valence_sets)
    V_pad = torch.zeros(n_atom_types, max_nv, device=device)
    pi_pad = torch.zeros(n_atom_types, max_nv, device=device)
    for atom_type in range(n_atom_types):
        if atom_type in stats.valence_sets:
            nv = stats.valence_sets[atom_type].shape[0]
            V_pad[atom_type, :nv] = stats.valence_sets[atom_type]
            pi_pad[atom_type, :nv] = stats.valence_priors[atom_type]

    v_exp = v_soft.unsqueeze(-1).unsqueeze(-1)
    V_exp = V_pad.unsqueeze(0).unsqueeze(0)
    gauss = torch.exp(-((v_exp - V_exp) ** 2) / (2 * sigma_v**2))
    weighted = (pi_pad.unsqueeze(0).unsqueeze(0) * gauss).sum(dim=-1)
    s_val = (A_soft * weighted).sum(dim=-1)

    neg_log = (-torch.log(s_val + eps)) * mask_flat
    n_valid = mask_flat.sum(dim=1).clamp(min=1.0)
    return (neg_log.sum(dim=1) / n_valid).mean()


def four_constraint_energies(
    R: torch.Tensor,
    A_soft: torch.Tensor,
    B_soft: torch.Tensor,
    stats,
    node_mask: torch.Tensor,
    stage: int,
    tau_c: float = 0.05,
    delta: float = 1.0,
    top_k: int = 4,
    sigma_v: float = 0.15,
    sigma_min_b: float = 0.03,
    sigma_min_a: float = 0.03,
) -> Mapping[str, torch.Tensor]:
    """Compute active NS4 molecular energies for the given stage."""
    zero = R.sum() * 0.0
    energies = {
        "E_clash_nonbonded": zero,
        "E_bond_len_posterior": zero,
        "E_angle_posterior": zero,
        "E_val_posterior": zero,
    }
    if stage >= 1:
        energies["E_clash_nonbonded"] = compute_clash_energy(
            R, A_soft, B_soft, stats, node_mask, tau_c=tau_c
        )
    if stage >= 2:
        energies["E_bond_len_posterior"] = compute_bond_len_energy(
            R, A_soft, B_soft, stats, node_mask, delta=delta, sigma_min_b=sigma_min_b
        )
    if stage >= 3:
        energies["E_angle_posterior"] = compute_angle_energy(
            R, A_soft, B_soft, stats, node_mask, delta=delta, top_k=top_k, sigma_min_a=sigma_min_a
        )
    if stage >= 4:
        energies["E_val_posterior"] = compute_valence_energy(
            A_soft, B_soft, stats, node_mask, sigma_v=sigma_v
        )
    return energies


class MolecularConstraintSet:
    """Callable constraint set for the molecular NS4 domain pack."""

    def __init__(
        self,
        stats,
        stage: int = 4,
        objective: Optional[WeightedObjective] = None,
        **energy_kwargs,
    ):
        self.stats = stats
        self.stage = stage
        self.objective = objective
        self.energy_kwargs = energy_kwargs

    def energies(
        self,
        R: torch.Tensor,
        A_soft: torch.Tensor,
        B_soft: torch.Tensor,
        node_mask: torch.Tensor,
        stage: Optional[int] = None,
    ) -> Mapping[str, torch.Tensor]:
        return four_constraint_energies(
            R,
            A_soft,
            B_soft,
            self.stats,
            node_mask,
            self.stage if stage is None else stage,
            **self.energy_kwargs,
        )

    def __call__(self, state: MolecularState, stage: Optional[int] = None) -> Mapping[str, torch.Tensor]:
        if state.bond_probs is None:
            raise ValueError("MolecularConstraintSet requires state.bond_probs.")
        return self.energies(state.R, state.A_soft, state.bond_probs, state.node_mask, stage=stage)

    def loss(self, state: MolecularState, stage: Optional[int] = None) -> torch.Tensor:
        if self.objective is None:
            raise ValueError("No objective was configured for this MolecularConstraintSet.")
        return self.objective.total(self(state, stage=stage))

