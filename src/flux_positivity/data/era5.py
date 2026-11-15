"""Monthly ERA5 inputs on the MetaFlux grid.

Source: the WeatherBench2 copy of ERA5 daily means at 0.25 deg and the ARCO-ERA5
hourly archive (both public on Google Cloud Storage, no credentials). Chunks are
read directly over HTTPS and decoded with numcodecs.

  TA_ERA    daily-mean 2 m temperature (C)
  VPD_ERA   es(T) - es(Td) from daily means (hPa), as in the MetaFlux paper
  P_ERA     daily total precipitation (mm/d)
  SW_IN_ERA net surface shortwave / (1 - forecast albedo) (W/m2); WeatherBench2
            stores net, not downward, shortwave. Validated against the towers'
            SW_IN_ERA in pipeline/04_validate_inputs.py (r = 0.97).
"""
from __future__ import annotations

import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .. import config
from ..grid import era5_to_mf
from ..physics import vpd_hpa

WB2 = ("https://storage.googleapis.com/weatherbench2/datasets/era5_daily/"
       "1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr")
ARCO = ("https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
        "full_37-1h-0p25deg-chunk-1.zarr-v3")


def _session():
    import requests
    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64))
    return s


def build_era5_monthly(cells: np.ndarray, years=config.ERA5_YEARS, every=config.ERA5_SAMPLE_EVERY,
                       albedo_days=config.ALBEDO_DAYS, out=None, max_days=None, workers=12, log=print):
    """Accumulate monthly means for `cells` (flat MetaFlux indices).

    max_days limits the number of sampled days (smoke tests)."""
    import numcodecs

    codec = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=1)
    session = _session()
    yrs = list(range(years[0], years[1] + 1))

    def fetch(url, tries=5):
        for k in range(tries):
            try:
                r = session.get(url, timeout=60)
                r.raise_for_status()
                a = np.frombuffer(codec.decode(r.content), dtype="<f4").reshape(721, 1440)
                return era5_to_mf(a).ravel()[cells]
            except Exception:  # noqa: BLE001 - retry any transport error
                if k == tries - 1:
                    raise
                time.sleep(2 * (k + 1))

    def day_task(d):
        i = (d - dt.date(1959, 1, 1)).days
        t = fetch(f"{WB2}/2m_temperature/{i}.0.0") - 273.15
        td = fetch(f"{WB2}/2m_dewpoint_temperature/{i}.0.0") - 273.15
        p = fetch(f"{WB2}/total_precipitation_24hr/{i}.0.0") * 1000.0
        swn = fetch(f"{WB2}/mean_surface_net_short_wave_radiation_flux/{i}.0.0")
        return d, np.stack([t, vpd_hpa(t, td), p, swn]).astype(np.float32)

    def albedo_task(d):
        h = int((dt.datetime(d.year, d.month, d.day, 12) - dt.datetime(1900, 1, 1)).total_seconds() // 3600)
        return d, fetch(f"{ARCO}/forecast_albedo/{h}.0.0").astype(np.float32)

    days, d = [], dt.date(yrs[0], 1, 1)
    while d.year <= yrs[-1]:
        days.append(d)
        d += dt.timedelta(days=every)
    adays = [dt.date(y, m, k) for y in yrs for m in range(1, 13) for k in albedo_days]
    if max_days:
        days, adays = days[:max_days], adays[:max_days]
    log(f"{len(days)} days x 4 variables + {len(adays)} albedo fields")

    ny, nc = len(yrs), cells.size
    sums = np.zeros((ny, 12, 4, nc))
    cnt = np.zeros((ny, 12), np.int32)
    alb = np.zeros((ny, 12, nc))
    acnt = np.zeros((ny, 12), np.int32)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(day_task, x) for x in days] + [ex.submit(albedo_task, x) for x in adays]
        for n, f in enumerate(as_completed(futs), 1):
            d, a = f.result()
            yi, mi = d.year - yrs[0], d.month - 1
            if a.ndim == 2:
                sums[yi, mi] += a
                cnt[yi, mi] += 1
            else:
                alb[yi, mi] += a
                acnt[yi, mi] += 1
            if n % 200 == 0:
                el = time.time() - t0
                log(f"{n}/{len(futs)} fields, {el/60:.1f} min, ETA {el/n*(len(futs)-n)/60:.1f} min")

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (sums / cnt[:, :, None, None]).astype(np.float32)
        albm = (alb / acnt[:, :, None]).astype(np.float32)
    ta, vpd, p, swn = (mean[:, :, k] for k in range(4))
    sw_in = swn / np.clip(1.0 - albm, 0.15, 1.0)   # cap the inversion over permanent snow
    res = dict(years=np.array(yrs), cells=cells, TA_ERA=ta, VPD_ERA=vpd, P_ERA=p,
               SW_IN_ERA=sw_in.astype(np.float32), SW_NET=swn, ALBEDO=albm, n_days=cnt, n_albedo=acnt)
    if out is not None:
        np.savez_compressed(out, **res)
    return res
