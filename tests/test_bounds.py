import numpy as np

from flux_positivity.bounds import (
    anchors_consistent,
    empirical_slopes,
    manski_interval,
    mcshane_whitney,
    nearest_anchor_interval,
    total_bounds,
)


def lipschitz_fn(Z, L=2.0):
    """An L-Lipschitz function on R^2 (|grad| = L along the diagonal)."""
    return L * np.abs(Z @ np.array([1.0, 1.0]) / np.sqrt(2)) + 1.0


def test_bounds_contain_truth_for_lipschitz_function():
    rng = np.random.default_rng(0)
    Za = rng.uniform(-1, 1, size=(60, 2))
    ya = lipschitz_fn(Za)
    Zx = rng.uniform(-3, 3, size=(500, 2))
    truth = lipschitz_fn(Zx)
    assert anchors_consistent(Za, ya, 2.0)
    lo, hi, feas = mcshane_whitney(Zx, Za, ya, 2.0)
    assert feas.all() and (lo <= truth + 1e-9).all() and (truth <= hi + 1e-9).all()
    d = np.linalg.norm(Zx[:, None] - Za[None], axis=2)
    k = d.argmin(1)
    nlo, nhi = nearest_anchor_interval(ya[k], d[np.arange(500), k], 2.0, y_min=-np.inf)
    assert (nlo <= lo + 1e-9).all() and (hi <= nhi + 1e-9).all()      # nearest-anchor contains MW


def test_mcshane_whitney_is_sharp():
    """The upper envelope is itself L-Lipschitz and matches the anchors, so the
    bound is attained: no tighter bound is valid for the class."""
    rng = np.random.default_rng(1)
    Za = rng.uniform(-1, 1, size=(30, 2))
    ya = lipschitz_fn(Za)
    _, up_at_anchors, _ = mcshane_whitney(Za, Za, ya, 2.0)
    assert np.allclose(up_at_anchors, ya)
    Zx = rng.uniform(-2, 2, size=(200, 2))
    _, up, _ = mcshane_whitney(Zx, Za, ya, 2.0)
    D = np.linalg.norm(Zx[:, None] - Zx[None], axis=2)
    assert (np.abs(up[:, None] - up[None]) <= 2.0 * D + 1e-9).all()


def test_infeasible_when_anchors_violate_L():
    Za = np.array([[0.0, 0.0], [0.1, 0.0]])
    ya = np.array([0.0, 5.0])                       # slope 50
    assert not anchors_consistent(Za, ya, 2.0)
    _, _, feas = mcshane_whitney(np.array([[0.05, 0.0]]), Za, ya, 2.0)
    assert not feas[0]


def test_total_bounds_and_manski():
    point = np.array([1.0, 2.0, 3.0])
    inside = np.array([True, True, False])
    w = np.ones(3)
    lo_out, hi_out = manski_interval(3, 0.0, 10.0)
    lo, hi = total_bounds(point, inside, w, lo_out, hi_out)
    assert (lo, hi) == (3.0, 13.0)
    lo, hi = total_bounds(point, inside, w, lo_out, hi_out, delta=0.5, y_max=10.0)
    assert (lo, hi) == (2.0, 14.0)


def test_empirical_slopes_respects_distance_window_and_sites():
    Za = np.array([[0.0], [1.0], [3.0]])
    ya = np.array([0.0, 2.0, 2.0])
    groups = np.array([0, 1, 2])
    E = np.eye(3, dtype=bool)
    s = empirical_slopes(Za, ya, groups, E, 0.5, 1.5)
    assert np.allclose(sorted(s), [2.0])
