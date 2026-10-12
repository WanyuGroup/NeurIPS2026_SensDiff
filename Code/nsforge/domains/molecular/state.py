"""State containers for molecular constraint packs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class MolecularState:
    """Soft molecular state consumed by NS-style molecular constraints."""

    coordinates: torch.Tensor
    atom_probs: torch.Tensor
    node_mask: torch.Tensor
    bond_probs: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def R(self) -> torch.Tensor:
        return self.coordinates

    @property
    def A_soft(self) -> torch.Tensor:
        return self.atom_probs

    @property
    def B_soft(self) -> Optional[torch.Tensor]:
        return self.bond_probs

    def with_bonds(self, bond_probs: torch.Tensor) -> "MolecularState":
        return MolecularState(
            coordinates=self.coordinates,
            atom_probs=self.atom_probs,
            node_mask=self.node_mask,
            bond_probs=bond_probs,
            metadata=dict(self.metadata),
        )

