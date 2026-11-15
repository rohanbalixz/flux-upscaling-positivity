"""Support of the tower sample in MetaFlux's input space.

A target input vector x is *outside support* if its distance to the nearest
tower site-month exceeds tau, where tau is the TAU_PERCENTILE of each tower
site-month's distance to the nearest site-month of a tower outside its
NEIGHBOURHOOD_KM neighbourhood (PREREGISTRATION s3).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class Space:
    """A feature representation fitted on the tower site-months.

    kind: "z" (pooled z-score, primary; MetaFlux paper Eq. 2), "logp" (z-score
    after log1p of precipitation), "rank" (tower ECDF), "maha" (whitened by
    the tower covariance).
    """

    def __init__(self, Xs: np.ndarray, kind: str = "z"):
        if kind not in ("z", "logp", "rank", "maha"):
            raise ValueError(kind)
        self.kind = kind
        Xs = self._pre(Xs)
        if kind in ("z", "logp"):
            self.mu, self.sd = Xs.mean(0), Xs.std(0)
        elif kind == "rank":
            self.q = np.sort(Xs, axis=0)
        else:
            self.mu = Xs.mean(0)
            self.W = np.linalg.cholesky(np.linalg.inv(np.cov(Xs, rowvar=False)))

    def _pre(self, X):
        X = np.asarray(X, np.float64).copy()
        if self.kind == "logp":
            X[..., 0] = np.log1p(np.maximum(X[..., 0], 0))
        return X

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = self._pre(X)
        if self.kind in ("z", "logp"):
            return (X - self.mu) / self.sd
        if self.kind == "rank":
            out = np.empty_like(X)
            for j in range(X.shape[-1]):
                out[..., j] = np.searchsorted(self.q[:, j], X[..., j]) / len(self.q[:, j])
            return out / np.sqrt(1 / 12)       # unit SD of a uniform variable
        return (X - self.mu) @ self.W


def loo_neighbourhood_nn(Zs: np.ndarray, site_ids: np.ndarray, E: np.ndarray,
                         k: int = 256, workers: int = -1) -> np.ndarray:
    """For each tower site-month, the distance to the nearest site-month of a
    site outside its neighbourhood (E[i, j] True = sites i, j are neighbours)."""
    tree = cKDTree(Zs)
    k = min(k, len(Zs))
    d, j = tree.query(Zs, k=k, workers=workers)
    site_ids = np.asarray(site_ids)
    ok = ~E[site_ids[:, None], site_ids[j]]
    out = d[np.arange(len(Zs)), ok.argmax(1)]
    for i in np.flatnonzero(~ok.any(1)):      # rare: all k neighbours are local
        m = ~E[site_ids[i], site_ids]
        out[i] = np.sqrt(((Zs[m] - Zs[i]) ** 2).sum(1)).min()
    return out


def support_distances(space: Space, Xs: np.ndarray, site_ids: np.ndarray, E: np.ndarray,
                      Xt: np.ndarray, q: float = 95):
    """(d_target, tau, d_towers_loo): distances of target vectors to the tower
    sample, the support threshold, and the tower leave-neighbourhood-out distances."""
    Zs = space(Xs)
    dl = loo_neighbourhood_nn(Zs, site_ids, E)
    tau = float(np.percentile(dl, q))
    d = cKDTree(Zs).query(space(Xt), k=1, workers=-1)[0]
    return d, tau, dl


def hellinger(a, wa, b, wb, bins: int = 100) -> float:
    """Hellinger distance between two weighted 1-D samples on shared quantile bins."""
    edges = np.unique(np.quantile(np.concatenate([a, b]), np.linspace(0, 1, bins + 1)))
    pa = np.histogram(a, edges, weights=wa)[0]
    pb = np.histogram(b, edges, weights=wb)[0]
    pa, pb = pa / pa.sum(), pb / pb.sum()
    return float(np.sqrt(max(0.0, 1 - np.sum(np.sqrt(pa * pb)))))
