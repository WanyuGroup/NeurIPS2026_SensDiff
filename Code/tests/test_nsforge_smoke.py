import unittest

import torch

from nsforge import GradientGuidance, WeightedObjective
from nsforge.domains.molecular import (
    BondPosteriorHead,
    MolecularConstraintSet,
    MolecularState,
    MolecularStats,
    NS4Scheduler,
    four_constraint_energies,
)


def make_stats(n_atom_types=3):
    stats = MolecularStats(n_atom_types=n_atom_types)
    n_bond_classes = 5
    stats.d_min = torch.full((n_atom_types, n_atom_types), 0.7)
    stats.bond_mu = torch.ones(n_atom_types, n_atom_types, n_bond_classes)
    stats.bond_sigma = torch.full((n_atom_types, n_atom_types, n_bond_classes), 0.2)
    stats.angle_cos_mu = torch.zeros(n_atom_types, n_atom_types, n_atom_types, n_bond_classes, n_bond_classes)
    stats.angle_cos_sigma = torch.full(
        (n_atom_types, n_atom_types, n_atom_types, n_bond_classes, n_bond_classes),
        0.3,
    )
    stats.angle_cos_mu_fallback = torch.zeros(n_atom_types, n_bond_classes, n_bond_classes)
    stats.angle_cos_sigma_fallback = torch.full((n_atom_types, n_bond_classes, n_bond_classes), 0.3)
    stats.angle_use_fallback = torch.ones(
        n_atom_types,
        n_atom_types,
        n_atom_types,
        n_bond_classes,
        n_bond_classes,
        dtype=torch.bool,
    )
    stats.valence_sets = {idx: torch.tensor([0.0, 1.0, 2.0, 3.0]) for idx in range(n_atom_types)}
    stats.valence_priors = {idx: torch.full((4,), 0.25) for idx in range(n_atom_types)}
    return stats


class NsforgeSmokeTests(unittest.TestCase):
    def test_generic_guidance_reduces_quadratic_energy(self):
        x = torch.tensor([2.0])
        guidance = GradientGuidance(
            objective=WeightedObjective({"quad": 1.0}),
            step_size=0.2,
            steps=3,
            normalize_grad=False,
        )

        result = guidance.step(x, lambda value: {"quad": (value**2).sum()})

        self.assertTrue(result.finite)
        self.assertLess(abs(float(result.value.item())), abs(float(x.item())))

    def test_molecular_energy_and_constraint_set_are_finite(self):
        torch.manual_seed(7)
        batch, n_nodes, n_atom_types = 2, 5, 3
        node_mask = torch.ones(batch, n_nodes, 1)
        R = torch.randn(batch, n_nodes, 3) * node_mask
        A_soft = torch.softmax(torch.randn(batch, n_nodes, n_atom_types), dim=-1) * node_mask
        bond_head = BondPosteriorHead(n_atom_types=n_atom_types, hidden_nf=16)
        B_soft = bond_head(R, A_soft, node_mask)
        stats = make_stats(n_atom_types=n_atom_types)

        energies = four_constraint_energies(R, A_soft, B_soft, stats, node_mask, stage=4)
        for energy in energies.values():
            self.assertTrue(torch.isfinite(energy).all())

        objective = WeightedObjective(
            {
                "E_clash_nonbonded": 4.0,
                "E_bond_len_posterior": 3.0,
                "E_angle_posterior": 2.0,
                "E_val_posterior": 1.0,
            }
        )
        constraints = MolecularConstraintSet(stats, stage=4, objective=objective)
        state = MolecularState(R, A_soft, node_mask, B_soft)
        loss = constraints.loss(state)

        self.assertTrue(torch.isfinite(loss).all())

    def test_ns4_scheduler_equation9_adaptive_release(self):
        scheduler = NS4Scheduler()

        # Constraints that have not been activated yet stay at zero weight.
        self.assertEqual(scheduler.get_lambdas(500)["val"], 0.0)
        self.assertEqual(scheduler.get_lambdas(1400)["val"], 0.0)

        # Linear warmup reaches half of the final weight at ramp end, and the
        # ramp-end rolling energy snapshot keeps the adaptive factor at zero.
        for epoch in range(1, 151):
            scheduler.check_stability(epoch, {"clash": 1.0})
        self.assertAlmostEqual(scheduler.get_lambdas(101)["clash"], 2.0)

        # Once the rolling energy mean approaches the release threshold, the
        # remaining half of the weight is released up to the full final value.
        for epoch in range(151, 171):
            scheduler.check_stability(epoch, {"clash": 1e-4})
        self.assertAlmostEqual(scheduler.get_lambdas(170)["clash"], 4.0)

        # Scheduler state survives a save/load round trip.
        restored = NS4Scheduler()
        restored.load_state_dict(scheduler.state_dict())
        self.assertAlmostEqual(restored.get_lambdas(170)["clash"], 4.0)


if __name__ == "__main__":
    unittest.main()

