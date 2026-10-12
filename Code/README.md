# Learning Generative Dynamics for 3D Molecule Generation via Sequential Neuro-Symbolic Constraints

Official code for:

> Learning Generative Dynamics for 3D Molecule Generation via Sequential Neuro-Symbolic Constraints (NeurIPS 2026)

The implementation lives in the `nsforge` package and the legacy `ns4.*` import paths keep working and forward to it.

## Requirements

- Python >= 3.9
- PyTorch >= 2.1
- NumPy
- RDKit 

## Installation

```bash
pip install -e .                 # core (torch + numpy)
pip install -e '.[molecular]'    # + RDKit, for Step 1 below
pip install -e '.[dev]'          # + pytest, to run the tests
```

## Run the tests

```bash
python -m unittest discover -s tests -v
# or
pytest tests/
```

## Step 1 — Precompute molecular statistics (QM9)

One-time offline pass over the QM9 training split (standard GEOLDM
preprocessing). Produces the `.pt` stats file consumed by training and
sampling:

```python
import numpy as np
from nsforge.domains.molecular import precompute_stats

ATOM_DECODER = {0: "H", 1: "C", 2: "N", 3: "O", 4: "F"}

smiles_list, positions_list, atom_types_list = [], [], []
for mol in load_qm9_train_split(...):  # your GEOLDM-preprocessed QM9 loader
    smiles_list.append(mol["smiles"])
    positions_list.append(np.asarray(mol["positions"], dtype=np.float64))
    atom_types_list.append(np.asarray(mol["atom_types"], dtype=np.int64))

stats = precompute_stats(
    smiles_list,
    positions_list,
    atom_types_list,
    list(ATOM_DECODER.values()),
)
stats.save("ns4_stats_qm9.pt")
```

## Step 2 — Training-time constraints

```python
import torch
from nsforge import WeightedObjective
from nsforge.domains.molecular import (
    BondPosteriorHead,
    MolecularConstraintSet,
    MolecularState,
    NS4Logger,
    NS4Scheduler,
    NS4Stats,
)

device = "cuda"
stats = NS4Stats.load("ns4_stats_qm9.pt", device=device)
bond_head = BondPosteriorHead(n_atom_types=5).to(device)  # pretrained in Stage 0, then frozen
scheduler = NS4Scheduler()
logger = NS4Logger(log_dir="runs/sensdiff_qm9")

# per epoch:
stage = scheduler.get_stage(epoch)
objective = WeightedObjective(scheduler.get_energy_weights(epoch))
constraints = MolecularConstraintSet(stats, stage=stage, objective=objective)

# per iteration, on the one-step reverse surrogate z_s:
R, A_soft = model.decode_soft(z_s, node_mask, edge_mask, context)
B_soft = bond_head(R, A_soft, node_mask)
state = MolecularState(R, A_soft, node_mask, B_soft)
energies = constraints(state)
loss_ns = objective.total(energies)
loss_total = loss_diffusion + loss_ns

# per epoch, after computing the four energies:
scheduler.check_stability(epoch, {
    "clash": energies["E_clash_nonbonded"],
    "bond": energies["E_bond_len_posterior"],
    "angle": energies["E_angle_posterior"],
    "val": energies["E_val_posterior"],
})
logger.log_losses(epoch, iter_idx, stage, energies, L_diff=loss_diffusion,
                  L_logic=loss_ns, L_total=loss_total)
```

`NS4Logger` writes `constraint_loss.csv` and `stage_events.jsonl` into
`log_dir`. To resume training across restarts, save/restore the scheduler
state alongside the model checkpoint:

```python
torch.save({"model": model.state_dict(), "scheduler": scheduler.state_dict()}, ckpt_path)
# later
scheduler.load_state_dict(torch.load(ckpt_path, map_location=device)["scheduler"])
```

## Step 3 — Inference-time refinement

```python
from nsforge.domains.molecular import FourConstraintProjector

projector = FourConstraintProjector(
    model=model,                 # GeoLDM-style diffusion model exposing decode_soft
    bond_posterior_head=bond_head,
    stats=stats,
)

# inside the reverse chain, after each denoising step:
z, energy_stats = projector.project(z, node_mask, edge_mask, context, step_idx=t)
```

## Repository layout

```
nsforge/core/                  generic objective, guidance loop, staged weight scheduler
nsforge/domains/molecular/     bond posterior head, constraint energies, statistics
                               precompute, curriculum scheduler, projector, logger
ns4/                           legacy import paths (forward to nsforge.domains.molecular)
tests/                         smoke tests
```

## Citation

If you use SensDiff in your research, please cite the paper:

```bibtex
@inproceedings{sensdiff2026,
  title     = {Learning Generative Dynamics for 3D Molecule Generation
               via Sequential Neuro-Symbolic Constraints},
  author    = {Li, Zhuoran; Sun, Xu; Chen, Chang Wen; Lin, Wanyu},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
