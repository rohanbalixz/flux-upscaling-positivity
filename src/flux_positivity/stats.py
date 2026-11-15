"""Statistics helpers: partial rank correlation and bootstrap resampling."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def partial_spearman(x, y, z) -> float:
    """Spearman correlation of x and y after linearly removing the ranks of z."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(np.corrcoef(ex, ey)[0, 1])


def group_bootstrap_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row indices of one bootstrap replicate that resamples whole groups."""
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    pick = rng.integers(0, len(uniq), size=len(uniq))
    return np.concatenate([members[p] for p in pick])


def interval(values, level: float = 0.95) -> list[float]:
    """[lower, median, upper] percentile interval."""
    a = (1 - level) / 2
    v = np.asarray(values, float)
    return [float(np.nanquantile(v, a)), float(np.nanquantile(v, 0.5)), float(np.nanquantile(v, 1 - a))]
