import numpy as np
import pandas as pd

from flux_positivity.support import Space, hellinger, loo_neighbourhood_nn, support_distances
from flux_positivity.towers import exclusion_matrix, realign_lai


def test_loo_neighbourhood_ignores_own_and_neighbouring_sites():
    # sites 0 and 1 are neighbours and share a point; site 2 is far away
    Zs = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 4.0]])
    sids = np.array([0, 1, 2])
    E = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], bool)
    d = loo_neighbourhood_nn(Zs, sids, E, k=3, workers=1)
    assert np.allclose(d, [5.0, 5.0, 5.0])


def test_support_threshold_flags_far_points_only():
    rng = np.random.default_rng(0)
    Xs = rng.normal(size=(400, 3))
    sids = np.repeat(np.arange(40), 10)
    E = np.eye(40, dtype=bool)
    Xt = np.vstack([np.zeros((5, 3)), np.full((5, 3), 8.0)])
    d, tau, _ = support_distances(Space(Xs, "z"), Xs, sids, E, Xt, 95)
    assert (d[:5] <= tau).all() and (d[5:] > tau).all()


def test_space_variants():
    rng = np.random.default_rng(1)
    X = rng.gamma(2.0, size=(1000, 3))
    z = Space(X, "z")(X)
    assert np.allclose(z.mean(0), 0, atol=1e-9) and np.allclose(z.std(0), 1)
    w = Space(X, "maha")(X)
    assert np.allclose(np.cov(w, rowvar=False), np.eye(3), atol=0.02)
    r = Space(X, "rank")(X)
    assert r.min() >= 0 and r.max() <= 1 / np.sqrt(1 / 12) + 1e-9


def test_hellinger_bounds():
    rng = np.random.default_rng(2)
    a = rng.normal(size=5000)
    assert hellinger(a, np.ones(5000), a, np.ones(5000)) < 1e-6
    assert hellinger(a, np.ones(5000), a + 10, np.ones(5000)) > 0.99


def test_exclusion_matrix_radius():
    tw = pd.DataFrame({"Site": ["A", "B", "C"], "Lat": [0.0, 0.3, 10.0], "Lon": [0.0, 0.0, 0.0]})
    E, sid = exclusion_matrix(tw, km=55)
    assert E[sid["A"], sid["B"]] and not E[sid["A"], sid["C"]] and E.diagonal().all()


def test_realign_lai_shifts_within_site():
    tw = pd.DataFrame({"Site": ["A"] * 6 + ["B"] * 6, "year": [2005] * 12,
                       "month": list(range(1, 7)) * 2, "LAI": list(range(6)) + list(range(10, 16))})
    out = realign_lai(tw, months=3)
    assert out.LAI.tolist()[:3] == [3, 4, 5] and np.isnan(out.LAI.iloc[3])
    assert out.LAI.tolist()[6:9] == [13, 14, 15]
