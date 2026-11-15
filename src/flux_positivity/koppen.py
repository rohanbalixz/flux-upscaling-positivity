"""Koppen main climate groups from monthly climatologies (Beck et al. 2018 criteria)."""
from __future__ import annotations

import numpy as np

GROUP_NAMES = {"A": "Tropical", "B": "Arid", "C": "Temperate", "D": "Continental", "E": "Polar/alpine"}


def koppen_main(t_c: np.ndarray, p_mm: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Main Koppen group per location.

    t_c: (12, n) monthly mean temperature (C); p_mm: (12, n) monthly totals (mm);
    lat: (n,) latitude, used to define the summer half-year.
    Precedence: E (warmest month < 10 C), then B (arid), then A, C, D.
    """
    t_c, p_mm, lat = np.asarray(t_c, float), np.asarray(p_mm, float), np.asarray(lat, float)
    mat, map_ = t_c.mean(0), p_mm.sum(0)
    tcold, thot = t_c.min(0), t_c.max(0)
    nh_summer = np.isin(np.arange(12), [3, 4, 5, 6, 7, 8])[:, None]
    sh_summer = np.isin(np.arange(12), [9, 10, 11, 0, 1, 2])[:, None]
    summer = np.where((lat >= 0)[None, :], nh_summer, sh_summer)
    psum = (p_mm * summer).sum(0)
    pth = np.where(psum >= 0.7 * map_, 2 * mat + 28,
                   np.where(psum <= 0.3 * map_, 2 * mat, 2 * mat + 14))
    g = np.full(t_c.shape[1], "D", dtype="<U1")
    g[(thot > 10) & (tcold > 0)] = "C"
    g[tcold >= 18] = "A"
    g[map_ < 10 * pth] = "B"
    g[thot < 10] = "E"
    return g
