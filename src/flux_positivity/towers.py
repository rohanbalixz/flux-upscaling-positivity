"""The MetaFlux training towers (public sample files of leap-stc/metaflux)."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

from . import config
from .grid import cell_index


def load_towers(root: Path | None = None) -> pd.DataFrame:
    """All site-month records, with year, month and grid cell added."""
    root = Path(root or config.TOWERS)
    files = sorted(glob.glob(str(root / "*" / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No tower CSVs under {root}; run pipeline/00_fetch.sh")
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        d = d.loc[:, ~d.columns.str.match("Unnamed")]
        d["task"] = Path(f).parent.name
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df["year"] = df.TIMESTAMP // 100
    df["month"] = df.TIMESTAMP % 100
    df["cell"] = cell_index(df.Lat.values, df.Lon.values)
    return df


def towers_complete(root: Path | None = None) -> pd.DataFrame:
    """Site-months with all five inputs and GPP (the source sample)."""
    tw = load_towers(root)
    return tw.dropna(subset=config.FEATURES + [config.TARGET]).reset_index(drop=True)


def great_circle_km(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    a = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def exclusion_matrix(tw: pd.DataFrame, km: float = config.NEIGHBOURHOOD_KM):
    """Boolean (n_sites, n_sites), True where two sites are within `km`
    (including a site with itself), and a site -> integer id mapping."""
    s = tw.groupby("Site").agg(lat=("Lat", "first"), lon=("Lon", "first")).sort_index()
    d = great_circle_km(s.lat.values[:, None], s.lon.values[:, None],
                        s.lat.values[None, :], s.lon.values[None, :])
    sid = {name: i for i, name in enumerate(s.index)}
    return d <= km, sid


def site_clusters(tw: pd.DataFrame, km: float = config.NEIGHBOURHOOD_KM) -> np.ndarray:
    """Cluster label per row: connected components of the within-`km` graph.
    Whole clusters are held out together in cross-validation."""
    E, sid = exclusion_matrix(tw, km)
    _, lab = connected_components(E, directed=False)
    return tw.Site.map(sid).map(lambda i: lab[i]).values


def realign_lai(tw: pd.DataFrame, months: int = config.LAI_REALIGN_MONTHS) -> pd.DataFrame:
    """DEVIATIONS.md D1: LAI_aligned(t) = LAI_csv(t + months), within site."""
    tw = tw.copy()
    key = tw.year * 12 + tw.month - 1
    lut = dict(zip(zip(tw.Site, key), tw.LAI))
    tw["LAI"] = [lut.get((s, k + months), np.nan) for s, k in zip(tw.Site, key)]
    return tw
