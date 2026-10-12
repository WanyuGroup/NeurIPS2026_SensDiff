"""CSV/JSONL logger used by the molecular NS4 training adapter."""

from __future__ import annotations

import csv
import json
import os
import time


class NS4Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.loss_path = os.path.join(log_dir, "constraint_loss.csv")
        self.stage_path = os.path.join(log_dir, "stage_events.jsonl")
        self._loss_file = open(self.loss_path, "a", newline="")
        self._writer = csv.DictWriter(
            self._loss_file,
            fieldnames=[
                "time",
                "epoch",
                "iter",
                "stage",
                "E_clash_nonbonded",
                "E_bond_len_posterior",
                "E_angle_posterior",
                "E_val_posterior",
                "L_diff",
                "L_logic",
                "L_total",
            ],
        )
        if os.path.getsize(self.loss_path) == 0:
            self._writer.writeheader()

    def log_losses(self, epoch, iter_idx, stage, energies, L_diff=None, L_logic=None, L_total=None):
        row = {
            "time": time.time(),
            "epoch": epoch,
            "iter": iter_idx,
            "stage": stage,
            "E_clash_nonbonded": energies.get("E_clash_nonbonded", 0.0),
            "E_bond_len_posterior": energies.get("E_bond_len_posterior", 0.0),
            "E_angle_posterior": energies.get("E_angle_posterior", 0.0),
            "E_val_posterior": energies.get("E_val_posterior", 0.0),
            "L_diff": L_diff,
            "L_logic": L_logic,
            "L_total": L_total,
        }
        self._writer.writerow(row)
        self._loss_file.flush()

    def log_stage_event(self, epoch, stage, extra=None):
        event = {"time": time.time(), "epoch": epoch, "stage": stage}
        if extra:
            event.update(extra)
        with open(self.stage_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def close(self):
        self._loss_file.close()

