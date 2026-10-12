"""Soft bond-order posterior head for molecular constraint energies."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BondPosteriorHead(nn.Module):
    """Predict bond-order posteriors from coordinates and soft atom types."""

    def __init__(self, n_atom_types: int = 5, hidden_nf: int = 128, n_bond_classes: int = 5):
        super().__init__()
        self.n_atom_types = n_atom_types
        self.n_bond_classes = n_bond_classes

        pair_in = 1 + 2 * n_atom_types
        self.mlp = nn.Sequential(
            nn.Linear(pair_in, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, n_bond_classes),
        )

        penalty = torch.zeros(n_bond_classes)
        penalty[1:] = -1e9
        self.register_buffer("_self_loop_penalty", penalty)

    def forward(self, R: torch.Tensor, A_soft: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """Return `B_soft` with shape `[B, N, N, M]`."""
        batch, n_nodes, _ = R.shape
        n_bond_classes = self.n_bond_classes

        diff = R.unsqueeze(2) - R.unsqueeze(1)
        dist = diff.norm(dim=-1, keepdim=True)
        pair_feat = torch.cat(
            [
                dist,
                A_soft.unsqueeze(2).expand(-1, -1, n_nodes, -1),
                A_soft.unsqueeze(1).expand(-1, n_nodes, -1, -1),
            ],
            dim=-1,
        )

        logits = self.mlp(pair_feat.reshape(-1, pair_feat.shape[-1]))
        logits = logits.view(batch, n_nodes, n_nodes, n_bond_classes)
        logits = 0.5 * (logits + logits.transpose(1, 2))

        mask_flat = node_mask.squeeze(-1)
        pair_mask = mask_flat.unsqueeze(2) * mask_flat.unsqueeze(1)
        diag = torch.eye(n_nodes, device=R.device)

        logits = logits + (
            diag.view(1, n_nodes, n_nodes, 1)
            * self._self_loop_penalty.view(1, 1, 1, n_bond_classes)
        )
        logits = logits + (1.0 - pair_mask).unsqueeze(-1) * (-1e9)

        return F.softmax(logits, dim=-1) * pair_mask.unsqueeze(-1)

