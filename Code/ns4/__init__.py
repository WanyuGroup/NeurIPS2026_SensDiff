"""Compatibility package for the original NS4 imports.

New code should import from `nsforge` directly.
"""

from nsforge.domains.molecular import (
    BondPosteriorHead,
    FourConstraintProjector,
    MolecularConstraintSet,
    MolecularGradientGuidance,
    MolecularState,
    MolecularStats,
    NS4Logger,
    NS4Scheduler,
    NS4Stats,
    four_constraint_energies,
    precompute_stats,
)

__all__ = [
    "BondPosteriorHead",
    "FourConstraintProjector",
    "MolecularConstraintSet",
    "MolecularGradientGuidance",
    "MolecularState",
    "MolecularStats",
    "NS4Logger",
    "NS4Scheduler",
    "NS4Stats",
    "four_constraint_energies",
    "precompute_stats",
]

