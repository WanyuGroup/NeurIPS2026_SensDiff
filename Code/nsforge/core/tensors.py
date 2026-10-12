"""Tensor utilities shared by domain packs."""

from __future__ import annotations

import torch


def remove_mean_with_mask(x: torch.Tensor, node_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Center valid nodes and keep invalid nodes at zero."""
    masked = x * node_mask
    denom = node_mask.sum(dim=1, keepdim=True).clamp(min=eps)
    mean = masked.sum(dim=1, keepdim=True) / denom
    return (masked - mean * node_mask) * node_mask

