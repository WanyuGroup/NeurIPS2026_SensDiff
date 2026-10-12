"""Statistics used by the molecular NS4 domain pack."""

from __future__ import annotations

from collections import defaultdict
from typing import List, Sequence

import numpy as np
import torch

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - optional dependency
    Chem = None


BOND_ORDER_MAP = (
    {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        Chem.rdchem.BondType.AROMATIC: 4,
    }
    if Chem is not None
    else {}
)

BOND_ORDER_WEIGHTS = [0.0, 1.0, 2.0, 3.0, 1.5]
N_BOND_CLASSES = 5


class MolecularStats:
    """Container for molecular constraint statistics.

    The tensors are intentionally plain attributes so downstream projects can
    build them from QM9, other molecule corpora, hand-written priors, or their
    own non-molecular-to-molecular adapters.
    """

    def __init__(self, n_atom_types: int = 5):
        self.n_atom_types = n_atom_types
        self.n_bond_classes = N_BOND_CLASSES
        self.d_min = None
        self.bond_mu = None
        self.bond_sigma = None
        self.angle_cos_mu = None
        self.angle_cos_sigma = None
        self.angle_cos_mu_fallback = None
        self.angle_cos_sigma_fallback = None
        self.angle_use_fallback = None
        self.valence_sets = {}
        self.valence_priors = {}
        self.bond_order_weights = torch.tensor(BOND_ORDER_WEIGHTS, dtype=torch.float32)

    def to(self, device):
        for name in [
            "d_min",
            "bond_mu",
            "bond_sigma",
            "angle_cos_mu",
            "angle_cos_sigma",
            "angle_cos_mu_fallback",
            "angle_cos_sigma_fallback",
            "angle_use_fallback",
            "bond_order_weights",
        ]:
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(device))
        for key in self.valence_sets:
            self.valence_sets[key] = self.valence_sets[key].to(device)
            self.valence_priors[key] = self.valence_priors[key].to(device)
        return self

    def state_dict(self):
        return {
            "n_atom_types": self.n_atom_types,
            "d_min": self.d_min,
            "bond_mu": self.bond_mu,
            "bond_sigma": self.bond_sigma,
            "angle_cos_mu": self.angle_cos_mu,
            "angle_cos_sigma": self.angle_cos_sigma,
            "angle_cos_mu_fallback": self.angle_cos_mu_fallback,
            "angle_cos_sigma_fallback": self.angle_cos_sigma_fallback,
            "angle_use_fallback": self.angle_use_fallback,
            "valence_sets": self.valence_sets,
            "valence_priors": self.valence_priors,
        }

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, device="cpu") -> "MolecularStats":
        data = torch.load(path, map_location=device)
        stats = cls(n_atom_types=data["n_atom_types"])
        stats.d_min = data["d_min"]
        stats.bond_mu = data["bond_mu"]
        stats.bond_sigma = data["bond_sigma"]
        stats.angle_cos_mu = data["angle_cos_mu"]
        stats.angle_cos_sigma = data["angle_cos_sigma"]
        stats.angle_cos_mu_fallback = data["angle_cos_mu_fallback"]
        stats.angle_cos_sigma_fallback = data["angle_cos_sigma_fallback"]
        stats.angle_use_fallback = data["angle_use_fallback"]
        stats.valence_sets = data["valence_sets"]
        stats.valence_priors = data["valence_priors"]
        return stats.to(device)


NS4Stats = MolecularStats


def precompute_stats(
    smiles_list: Sequence[str],
    positions_list: Sequence[np.ndarray],
    atom_types_list: Sequence[np.ndarray],
    atom_decoder: List[str],
    n_atom_types: int = 5,
    min_angle_samples: int = 100,
    nonbond_quantile: float = 0.02,
) -> MolecularStats:
    """Precompute molecular statistics from aligned SMILES/position/type data."""
    if Chem is None:
        raise ImportError("RDKit is required for molecular statistics precomputation.")

    n_types = n_atom_types
    n_bond_classes = N_BOND_CLASSES

    nonbond_dists = defaultdict(list)
    bond_dists = defaultdict(list)
    angle_cos_vals = defaultdict(list)
    angle_cos_fallback = defaultdict(list)
    valence_counts = defaultdict(lambda: defaultdict(int))

    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)

        pos = positions_list[idx]
        atom_types = atom_types_list[idx]
        n_atoms = len(atom_types)

        bond_graph = {}
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i >= n_atoms or j >= n_atoms:
                continue
            m = BOND_ORDER_MAP.get(bond.GetBondType(), 1)
            bond_graph[(i, j)] = m
            bond_graph[(j, i)] = m

        for (i, j), m in bond_graph.items():
            if i >= j:
                continue
            a, b = atom_types[i], atom_types[j]
            d = np.linalg.norm(pos[i] - pos[j])
            a_s, b_s = min(a, b), max(a, b)
            bond_dists[(a_s, b_s, m)].append(d)

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if (i, j) not in bond_graph and (j, i) not in bond_graph:
                    a, b = atom_types[i], atom_types[j]
                    d = np.linalg.norm(pos[i] - pos[j])
                    a_s, b_s = min(a, b), max(a, b)
                    nonbond_dists[(a_s, b_s)].append(d)

        for j in range(n_atoms):
            neighbors = [(k, bond_graph[(j, k)]) for k in range(n_atoms) if (j, k) in bond_graph]
            for i, m in neighbors:
                for k, n in neighbors:
                    if i >= k:
                        continue
                    v1 = pos[i] - pos[j]
                    v2 = pos[k] - pos[j]
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)
                    if norm1 < 1e-6 or norm2 < 1e-6:
                        continue
                    cos_val = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)

                    a_i, a_j, a_k = atom_types[i], atom_types[j], atom_types[k]
                    a_left, a_right = min(a_i, a_k), max(a_i, a_k)
                    m_left, m_right = (m, n) if a_i <= a_k else (n, m)
                    angle_cos_vals[(a_left, a_j, a_right, m_left, m_right)].append(cos_val)
                    angle_cos_fallback[(a_j, m_left, m_right)].append(cos_val)

        for i in range(n_atoms):
            a = atom_types[i]
            valence = 0.0
            for j in range(n_atoms):
                if (i, j) in bond_graph:
                    valence += BOND_ORDER_WEIGHTS[bond_graph[(i, j)]]
            valence_counts[a][valence] += 1

    stats = MolecularStats(n_atom_types=n_types)

    d_min = torch.full((n_types, n_types), 1.0)
    for (a, b), dists in nonbond_dists.items():
        if len(dists) > 10:
            q = np.quantile(dists, nonbond_quantile)
            d_min[a, b] = q
            d_min[b, a] = q
    stats.d_min = d_min

    bond_mu = torch.zeros(n_types, n_types, n_bond_classes)
    bond_sigma = torch.full((n_types, n_types, n_bond_classes), 0.1)
    for (a, b, m), dists in bond_dists.items():
        if len(dists) >= 2:
            mu = np.mean(dists)
            sigma = max(np.std(dists), 0.01)
            bond_mu[a, b, m] = mu
            bond_mu[b, a, m] = mu
            bond_sigma[a, b, m] = sigma
            bond_sigma[b, a, m] = sigma
    stats.bond_mu = bond_mu
    stats.bond_sigma = bond_sigma

    angle_cos_mu = torch.zeros(n_types, n_types, n_types, n_bond_classes, n_bond_classes)
    angle_cos_sigma = torch.full((n_types, n_types, n_types, n_bond_classes, n_bond_classes), 0.3)
    angle_use_fallback = torch.ones(n_types, n_types, n_types, n_bond_classes, n_bond_classes, dtype=torch.bool)
    for (a, b, c, m, n), vals in angle_cos_vals.items():
        if len(vals) >= min_angle_samples:
            angle_cos_mu[a, b, c, m, n] = np.mean(vals)
            angle_cos_sigma[a, b, c, m, n] = max(np.std(vals), 0.01)
            angle_use_fallback[a, b, c, m, n] = False

    angle_cos_mu_fb = torch.zeros(n_types, n_bond_classes, n_bond_classes)
    angle_cos_sigma_fb = torch.full((n_types, n_bond_classes, n_bond_classes), 0.3)
    for (b, m, n), vals in angle_cos_fallback.items():
        if len(vals) >= 5:
            angle_cos_mu_fb[b, m, n] = np.mean(vals)
            angle_cos_sigma_fb[b, m, n] = max(np.std(vals), 0.01)

    stats.angle_cos_mu = angle_cos_mu
    stats.angle_cos_sigma = angle_cos_sigma
    stats.angle_cos_mu_fallback = angle_cos_mu_fb
    stats.angle_cos_sigma_fallback = angle_cos_sigma_fb
    stats.angle_use_fallback = angle_use_fallback

    for atom_type in range(n_types):
        if atom_type in valence_counts and len(valence_counts[atom_type]) > 0:
            vals = sorted(valence_counts[atom_type].keys())
            counts = [valence_counts[atom_type][v] for v in vals]
            total = sum(counts)
            stats.valence_sets[atom_type] = torch.tensor(vals, dtype=torch.float32)
            stats.valence_priors[atom_type] = torch.tensor([count / total for count in counts], dtype=torch.float32)
        else:
            stats.valence_sets[atom_type] = torch.tensor([0.0], dtype=torch.float32)
            stats.valence_priors[atom_type] = torch.tensor([1.0], dtype=torch.float32)

    return stats

