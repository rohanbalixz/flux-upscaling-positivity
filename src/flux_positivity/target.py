"""Target (pixel) features in MetaFlux's input space, vegetated mask, Koppen
groups and the MetaFlux GPP / GPP_std climatologies. Cached in
data/processed/target.npz."""
from __future__ import annotations

import numpy as np

from . import config
from .grid import DAYS_IN_MONTH, NLON, cell_latlon
from .koppen import koppen_main


def build_target(force: bool = False) -> dict:
    f = config.PROC / "target.npz"
    if f.exists() and not force:
        return dict(np.load(f, allow_pickle=True))
    import xarray as xr

    e5 = np.load(config.PROC / "era5_monthly.npz")
    L = np.load(config.PROC / "lai_monthly.npz", allow_pickle=True)
    cells = e5["cells"]
    if not np.array_equal(cells, L["cells"]):
        raise ValueError("ERA5 and LAI caches were built on different masks")
    yrs = e5["years"]
    ysel = (yrs >= config.CLIM_YEARS[0]) & (yrs <= config.CLIM_YEARS[1])
    clim = {v: e5[v][ysel].mean(0) for v in ["P_ERA", "VPD_ERA", "TA_ERA", "SW_IN_ERA"]}
    clim["LAI"] = L["lai_clim"]
    Xc = np.stack([clim[v] for v in config.FEATURES], axis=-1).astype(np.float32)   # (12, n, 5)
    peryear = np.stack([e5[v] for v in ["P_ERA", "VPD_ERA", "TA_ERA", "SW_IN_ERA"]], -1)

    lat, lon = cell_latlon(cells)
    nonveg = L["frac_barren"] + L["frac_snow"] + L["frac_water"]
    laimax = np.nanmax(L["lai_clim"], axis=0)
    mf_valid = np.load(config.PROC / "mask.npz")["metaflux_valid"].ravel()[cells]
    veg = (mf_valid & (nonveg < config.VEG_MAX_NONVEG_SHARE) & (laimax >= config.VEG_MIN_LAIMAX)
           & np.isfinite(Xc).all(axis=(0, 2)))
    kg = koppen_main(clim["TA_ERA"], clim["P_ERA"] * DAYS_IN_MONTH[:, None], lat)

    gpp = np.zeros((12, cells.size))
    gstd = np.zeros((12, cells.size))
    r, c = cells // NLON, cells % NLON
    years = range(config.CLIM_YEARS[0], config.CLIM_YEARS[1] + 1)
    for y in years:
        with xr.open_dataset(config.METAFLUX_NC / f"METAFLUX_GPP_RECO_monthly_{y}.nc") as ds:
            gpp += ds["GPP"].values[:, r, c]
            gstd += ds["GPP_std"].values[:, r, c]
    gpp /= len(years)
    gstd /= len(years)
    out = dict(cells=cells, lat=lat, lon=lon, Xc=Xc, peryear=peryear.astype(np.float32), years=yrs,
               veg=veg, kg=kg, gpp=gpp.astype(np.float32), gpp_std=gstd.astype(np.float32),
               area=np.cos(np.deg2rad(lat)).astype(np.float32), nonveg=nonveg, laimax=laimax)
    np.savez_compressed(f, **out)
    return out
