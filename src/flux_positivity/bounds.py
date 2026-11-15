"""Partial identification of a transported total when positivity fails.

Notation (docs/THEORY.md): anchors (a_k, y_k) are tower points with known
conditional mean; a target point x at distance d_k = ||x - a_k|| from anchor k.
F_L is the class of L-Lipschitz functions on input space.

* `nearest_anchor_interval`: f(x) in [y_k* - L d_k*, y_k* + L d_k*], k* the
  nearest anchor. Valid for every f in F_L that matches anchor k*; it contains
  the McShane-Whitney interval and is always non-empty.
* `mcshane_whitney`: the sharp interval over F_L given all anchors,
  [max_k (y_k - L d_k), min_k (y_k + L d_k)]. Empty (lo > hi) when the anchors
  themselves violate L-Lipschitz near x.
* `manski_interval`: no smoothness, outcome range only.
"""
from __future__ import annotations

import numpy as np


def nearest_anchor_interval(y_anchor, d, L, y_min=0.0, y_max=np.inf, delta=0.0):
    lo = np.clip(np.asarray(y_anchor) - L * np.asarray(d) - delta, y_min, y_max)
    hi = np.clip(np.asarray(y_anchor) + L * np.asarray(d) + delta, y_min, y_max)
    return lo, hi


def mcshane_whitney(Zx: np.ndarray, Za: np.ndarray, ya: np.ndarray, L: float):
    """Lower and upper McShane-Whitney envelopes at points Zx (n, p) given
    anchors Za (m, p) with values ya (m,). Returns (lo, hi, feasible)."""
    D = np.linalg.norm(Zx[:, None, :] - Za[None, :, :], axis=2)
    hi = np.min(ya[None, :] + L * D, axis=1)
    lo = np.max(ya[None, :] - L * D, axis=1)
    return lo, hi, lo <= hi


def anchors_consistent(Za: np.ndarray, ya: np.ndarray, L: float) -> bool:
    """True if the anchor data admit an L-Lipschitz interpolant."""
    D = np.linalg.norm(Za[:, None, :] - Za[None, :, :], axis=2)
    return bool(np.all(np.abs(ya[:, None] - ya[None, :]) <= L * D + 1e-12))


def manski_interval(n, y_min=0.0, y_max=1.0):
    return np.full(n, float(y_min)), np.full(n, float(y_max))


def total_bounds(point, inside, weights, out_lo, out_hi, delta=0.0, y_min=0.0, y_max=np.inf):
    """Bounds on sum(weights * f): point-identified inside support (+/- delta
    if conditional invariance is relaxed), [out_lo, out_hi] outside."""
    lo_in = np.clip(point - delta, y_min, y_max)
    hi_in = np.clip(point + delta, y_min, y_max)
    lo = np.where(inside, lo_in, out_lo)
    hi = np.where(inside, hi_in, out_hi)
    return float((lo * weights).sum()), float((hi * weights).sum())


def empirical_slopes(Za, ya, groups, E, dmin, dmax):
    """|y_i - y_j| / ||a_i - a_j|| for anchor pairs from non-neighbouring sites
    at distances in [dmin, dmax]."""
    from scipy.spatial import cKDTree
    pairs = cKDTree(Za).query_pairs(dmax, output_type="ndarray")
    if len(pairs) == 0:
        return np.array([])
    dd = np.linalg.norm(Za[pairs[:, 0]] - Za[pairs[:, 1]], axis=1)
    keep = (dd >= dmin) & ~E[groups[pairs[:, 0]], groups[pairs[:, 1]]]
    return np.abs(ya[pairs[keep, 0]] - ya[pairs[keep, 1]]) / dd[keep]
