"""Monthly MODIS LAI climatology on the MetaFlux grid from full-resolution
MOD15A2H (Terra, C6.1, 500 m, 8-day) tiles on Microsoft Planetary Computer.

Full resolution is required: the COG overviews on Planetary Computer blend fill
codes (249-255) into valid LAI, so they cannot be used for area means.

  * two composites per month (nearest the 1st and the 17th), max taken
    (limits cloud-contaminated retrievals), then averaged over LAI_YEARS
  * cell value = mean of valid 500 m pixels; fill-code shares are kept
  * towers: mean of valid pixels in the 3x3 (1.5 km) window at the tower
"""
from __future__ import annotations

import datetime as dt
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .. import config
from ..grid import NLAT, NLON

R_SIN = 6371007.181          # MODIS sinusoidal sphere radius (m)
TILE_M = 1111950.5197665     # MODIS tile width (m)
NCELL = NLAT * NLON


def composite_dates(years=config.LAI_YEARS, days=config.LAI_COMPOSITE_DAYS):
    """[(year, month, slot, start_date)] of the 8-day composites used."""
    out = []
    for y in years:
        starts = [dt.date(y, 1, 1) + dt.timedelta(days=8 * k) for k in range(46)]
        for m in range(1, 13):
            for slot, day in enumerate(days):
                target = dt.date(y, m, day)
                out.append((y, m, slot, min(starts, key=lambda s, t=target: abs((s - t).days))))
    return out


def tile_cell_index(transform, in_mask: np.ndarray) -> np.ndarray:
    """Flat MetaFlux cell index (or -1) of every 500 m pixel of a 2400x2400 tile."""
    c = np.arange(2400) + 0.5
    x = transform.c + c * transform.a
    y = transform.f + c * transform.e
    X, LATR = np.meshgrid(x, y / R_SIN)
    with np.errstate(divide="ignore", invalid="ignore"):
        lonr = X / (R_SIN * np.cos(LATR))
    ok = np.abs(lonr) <= np.pi
    r = np.clip(np.round((np.rad2deg(LATR) + 90) / 0.25).astype(np.int64), 0, NLAT - 1)
    cc = np.round((np.rad2deg(lonr) + 180) / 0.25).astype(np.int64) % NLON
    idx = np.where(ok, r * NLON + cc, -1)
    idx[ok] = np.where(in_mask[idx[ok]], idx[ok], -1)
    return idx.ravel()


def tower_pixels(towers, h: int, v: int, transform):
    """[(tower_row, row, col)] for towers whose 500 m pixel lies in tile (h, v)."""
    out = []
    for i, (la, lo) in enumerate(zip(np.deg2rad(towers.lat.values), np.deg2rad(towers.lon.values))):
        x, y = R_SIN * lo * np.cos(la), R_SIN * la
        if int(np.floor((x + 18 * TILE_M) / TILE_M)) != h or int(np.floor((9 * TILE_M - y) / TILE_M)) != v:
            continue
        col = int((x - transform.c) / transform.a)
        row = int((y - transform.f) / transform.e)
        out.append((i, min(max(row, 1), 2398), min(max(col, 1), 2398)))
    return out


def build_lai_monthly(mask_cells: np.ndarray, towers, out=None, max_tiles=None, workers=24, log=print):
    """towers: DataFrame with columns Site, lat, lon (one row per site)."""
    import planetary_computer as pc
    import pystac_client
    import rasterio
    import requests

    in_mask = np.zeros(NCELL, bool)
    in_mask[mask_cells] = True
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32))

    def read_tile(href, tries=5):
        for k in range(tries):
            try:
                r = session.get(pc.sign(href), timeout=120)
                r.raise_for_status()
                with rasterio.MemoryFile(io.BytesIO(r.content)) as mf, mf.open() as src:
                    return src.read(1), src.transform
            except Exception:  # noqa: BLE001
                if k == tries - 1:
                    raise
                time.sleep(3 * (k + 1))

    def tile_task(tile_id, hrefs):
        h, v = int(tile_id[2:5]), int(tile_id[5:8])          # ids look like "51" + hhh + vvv
        ncomp, res = len(hrefs), None
        for k, href in enumerate(hrefs):
            if href is None:
                continue
            a, transform = read_tile(href)
            if res is None:
                idx = tile_cell_index(transform, in_mask)
                keep = idx >= 0
                ucells, inv = np.unique(idx[keep], return_inverse=True)
                nloc = ucells.size
                tw = tower_pixels(towers, h, v, transform)
                res = dict(cells=ucells, s=np.zeros((ncomp, nloc), np.float32), n=np.zeros((ncomp, nloc), np.int32),
                           tot=np.zeros(nloc, np.int64), barren=np.zeros(nloc, np.int64), snow=np.zeros(nloc, np.int64),
                           water=np.zeros(nloc, np.int64), tw=[t[0] for t in tw],
                           twv=np.full((len(tw), ncomp), np.nan, np.float32))
            flat = a.ravel()[keep]
            valid = flat <= 100
            res["s"][k] = np.bincount(inv[valid], weights=flat[valid] * 0.1, minlength=nloc)
            res["n"][k] = np.bincount(inv[valid], minlength=nloc)
            res["tot"] += np.bincount(inv, minlength=nloc)
            res["barren"] += np.bincount(inv[flat == 253], minlength=nloc)
            res["snow"] += np.bincount(inv[flat == 252], minlength=nloc)
            res["water"] += np.bincount(inv[flat == 254], minlength=nloc)
            for j, (_, row, col) in enumerate(tw):
                w = a[row - 1:row + 2, col - 1:col + 2].astype(float)
                w = w[w <= 100]
                if w.size:
                    res["twv"][j, k] = w.mean() * 0.1
        return res

    comps = composite_dates()
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    hrefs: dict[str, list] = {}
    for k, (_, _, _, d) in enumerate(comps):
        items = cat.search(collections=["modis-15A2H-061"], datetime=f"{d.isoformat()}/{d.isoformat()}",
                           query={"platform": {"eq": "terra"}}, limit=1000).item_collection()
        for it in items:
            if it.properties.get("start_datetime", "")[:10] == d.isoformat():
                hrefs.setdefault(it.properties["modis:tile-id"], [None] * len(comps))[k] = it.assets["Lai_500m"].href
    tiles = list(hrefs.items())[:max_tiles] if max_tiles else list(hrefs.items())
    log(f"{len(comps)} composites, {len(tiles)} tiles")

    S = np.zeros((len(comps), NCELL), np.float32)
    N = np.zeros((len(comps), NCELL), np.int32)
    TOT, BAR, SNO, WAT = (np.zeros(NCELL, np.int64) for _ in range(4))
    TW = np.full((len(towers), len(comps)), np.nan, np.float32)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(tile_task, tid, hl) for tid, hl in tiles]
        for n_done, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r is not None:
                c = r["cells"]
                S[:, c] += r["s"]
                N[:, c] += r["n"]
                TOT[c] += r["tot"]
                BAR[c] += r["barren"]
                SNO[c] += r["snow"]
                WAT[c] += r["water"]
                for j, si in enumerate(r["tw"]):
                    TW[si] = r["twv"][j]
            if n_done % 10 == 0:
                el = time.time() - t0
                log(f"{n_done}/{len(futs)} tiles, {el/60:.1f} min, ETA {el/n_done*(len(futs)-n_done)/60:.1f} min")

    ny, nd = len(config.LAI_YEARS), len(config.LAI_COMPOSITE_DAYS)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm = (S / N).reshape(ny, 12, nd, NCELL)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            clim = np.nanmean(np.nanmax(cm, axis=2), axis=0)
            twm = np.nanmean(np.nanmax(TW.reshape(len(towers), ny, 12, nd), axis=3), axis=1)
        frac = lambda a: (a / np.maximum(TOT, 1)).astype(np.float32)  # noqa: E731
    res = dict(lai_clim=clim[:, mask_cells].astype(np.float32), cells=mask_cells,
               frac_valid=(N.sum(0) / np.maximum(TOT, 1))[mask_cells].astype(np.float32),
               frac_barren=frac(BAR)[mask_cells], frac_snow=frac(SNO)[mask_cells],
               frac_water=frac(WAT)[mask_cells], tower_sites=towers.Site.values,
               tower_lai_3x3=twm.astype(np.float32))
    if out is not None:
        np.savez_compressed(out, **res)
    return res
