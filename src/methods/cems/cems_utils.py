"""
CEMS neighbourhood utilities — port of reference cems_utils.py (simplified).

The neighbourhood is precomputed once before training (DINOv2 features are
frozen, Y_train doesn't change) and reused every epoch.
"""
from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# neigh_size formula
# ---------------------------------------------------------------------------

def _next_power_of_2(n: int) -> int:
    """Smallest power of 2 that is >= n.  Port of shift_bit_length from reference."""
    if n < 1:
        return 1
    return 1 << (n - 1).bit_length()


def compute_neigh_size(d: int) -> int:
    """neigh_size = next_power_of_2(d + d*(d+1)//2 + 1).

    Guarantees the quadratic regression system is over-determined (more
    equations than unknowns).  Reference: main.py estimate_intrinsic_dim.
    """
    base = d + d * (d + 1) // 2
    return _next_power_of_2(base + 1)


# ---------------------------------------------------------------------------
# kNN precomputation
# ---------------------------------------------------------------------------

def precompute_knn(
        X: np.ndarray,
        Y: np.ndarray,
        neigh_size: int,
        device: str = "cpu",
) -> np.ndarray:
    """Pre-sort training points by Euclidean distance in joint [X, Y] space.

    Port of get_probabilities (knn mode) from reference cems_utils.py.
    Distance is in the concatenated [X, Y] space so label similarity
    influences neighbourhood membership (matches the reference design).

    Args:
        X:          features (N, d_x), e.g. DINOv2 384-d or latent 32-d.
        Y:          MinMaxScaled targets (N, n_targets).
        neigh_size: number of nearest neighbours to keep per point (incl. self).
        device:     'cuda', 'mps', or 'cpu' for distance computation.

    Returns:
        indices: int64 array (N, neigh_size).
                 Row i: indices of the neigh_size nearest points to i, with
                 i itself at column 0 (self-index).  Columns 1: are the
                 sorted nearest neighbours (ascending distance).
    """
    XY = np.concatenate([X, Y], axis=1).astype(np.float32)
    t = torch.tensor(XY, dtype=torch.float32)
    if device in ("cuda", "mps"):
        try:
            t = t.to(device)
        except Exception:
            pass   # fall back to CPU silently

    # -inf on diagonal so self sorts to column 0 in argsort (matches reference)
    dists = torch.cdist(t, t, p=2, compute_mode="donot_use_mm_for_euclid_dist")
    dists.fill_diagonal_(-float('inf'))

    sorted_idx = torch.argsort(dists, dim=1).cpu().numpy()   # (N, N), col 0 = self

    # Keep only neigh_size columns; col 0 = self, cols 1: = nearest neighbours
    indices = sorted_idx[:, :neigh_size]
    return indices.astype(np.int64)
