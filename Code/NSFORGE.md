# nsforge

`nsforge` is a reusable library for neuro-symbolic constraints and guidance. The
core package is not tied to molecules, diffusion, or any specific model:
it combines named symbolic energies, schedules their weights, and applies them
as either training losses or inference-time gradient guidance.

The current NS4 chemistry implementation is included as the first domain pack:
`nsforge.domains.molecular`.

## Install

```bash
cd NS4
pip install -e .
```

For the molecular statistics precompute path, install the optional dependency:

```bash
pip install -e '.[molecular]'
```

## Generic Training Use

```python
from nsforge import WeightedObjective

objective = WeightedObjective({"constraint_a": 1.0, "constraint_b": 0.25})
energies = constraint_set(decoded_state)
loss = task_loss + objective.total(energies)
```

## Generic Inference Guidance

```python
from nsforge import GradientGuidance, WeightedObjective

guidance = GradientGuidance(
    objective=WeightedObjective({"rule": 1.0}),
    step_size=0.1,
    steps=4,
)

result = guidance.step(latent, energy_fn=lambda z: {"rule": rule_energy(z)})
latent = result.value
```

## Molecular NS4 Domain Pack

```python
from nsforge import WeightedObjective
from nsforge.domains.molecular import (
    BondPosteriorHead,
    MolecularConstraintSet,
    MolecularState,
    NS4Stats,
)

stats = NS4Stats.load("ns4/ns4_stats_qm9.pt", device=device)
bond_head = BondPosteriorHead(n_atom_types=5).to(device)
objective = WeightedObjective({
    "E_clash_nonbonded": 4.0,
    "E_bond_len_posterior": 3.0,
    "E_angle_posterior": 2.0,
    "E_val_posterior": 1.0,
})

R, A_soft = decode_soft(model_state)
B_soft = bond_head(R, A_soft, node_mask)
state = MolecularState(R, A_soft, node_mask, B_soft)

constraints = MolecularConstraintSet(stats, stage=4, objective=objective)
loss_ns = constraints.loss(state)
```

## Compatibility

The old `ns4.*` modules now forward to `nsforge.domains.molecular.*`, so the
existing `train.py`, `sample.py`, and paper experiment scripts keep working.
