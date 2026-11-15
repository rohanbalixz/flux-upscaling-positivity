#!/usr/bin/env python3
"""Verify the LAI phase offset in the MetaFlux sample training files, in ~1 minute.

Uses only the public CSVs in github.com/leap-stc/metaflux (metaflux/data/sample)
and numpy + pandas. No other data, no MODIS, no ERA5: the test compares columns
of the same rows with each other.

For each lag k, it prints the pooled within-site correlation between a column at
month t and GPP at month t + k. A column in phase with GPP peaks near k = 0; a
column that runs late peaks at negative k.

  python scripts/check_lai_phase.py                 # download (if needed) and check
  python scripts/check_lai_phase.py --download-only # just fetch the CSVs
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "leap-stc/metaflux"
DEST = Path(__file__).resolve().parents[1] / "data" / "towers"
GPP = "GPP_NT_VUT_REF"


def download(dest: Path = DEST) -> list[Path]:
    tree = json.load(urllib.request.urlopen(
        f"https://api.github.com/repos/{REPO}/git/trees/main?recursive=1", timeout=60))
    paths = [e["path"] for e in tree["tree"]
             if e["path"].startswith("metaflux/data/sample/") and e["path"].endswith(".csv")]

    def get(p):
        out = dest / p.removeprefix("metaflux/data/sample/")
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"https://raw.githubusercontent.com/{REPO}/main/{p}", out)
        return out

    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(get, paths))


def lag_table(df: pd.DataFrame, col: str, lags=range(-4, 5)) -> dict[int, float]:
    d = df[["Site", "k", col, GPP]]
    out = {}
    for k in lags:
        x = d[["Site", "k", col]].dropna()
        y = d[["Site", "k", GPP]].dropna().assign(k=lambda t, k=k: t.k - k)
        m = x.merge(y, on=["Site", "k"])
        for c in (col, GPP):
            m[c] = m[c] - m.groupby("Site")[c].transform("mean")
        out[k] = float(np.corrcoef(m[col], m[GPP])[0, 1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()
    files = sorted(DEST.glob("*/*.csv")) or download()
    if len(files) < 200:
        files = download()
    print(f"{len(files)} tower files in {DEST}")
    if args.download_only:
        return 0

    df = pd.concat([pd.read_csv(f).assign(Site=f.stem.split("_")[0]) for f in files], ignore_index=True)
    df = df[df.TIMESTAMP // 100 >= 2000]                      # MODIS era
    df["k"] = (df.TIMESTAMP // 100) * 12 + df.TIMESTAMP % 100

    print("\nPooled within-site correlation of X(t) with GPP(t + k)")
    print("k:          " + " ".join(f"{k:>6d}" for k in range(-4, 5)))
    res = {}
    for col in ("SW_IN_ERA", "TA_ERA", "LAI", "EVI"):
        res[col] = lag_table(df, col)
        print(f"{col:11s} " + " ".join(f"{v:6.2f}" for v in res[col].values()))
    peak = {c: max(v, key=v.get) for c, v in res.items()}
    print("\nlag of peak correlation:", peak)

    c = df.assign(month=df.TIMESTAMP % 100).groupby(["Site", "month"])[["LAI", GPP]].mean().unstack()
    best = []
    for s in c.index:
        a, g = c.loc[s, "LAI"].values, c.loc[s, GPP].values
        if np.isnan(a).any() or np.isnan(g).any() or np.std(g) < 0.5:
            continue
        r = {k: np.corrcoef(a, np.roll(g, k))[0, 1] for k in range(-5, 7)}
        best.append(max(r, key=r.get))
    vc = pd.Series(best).value_counts().sort_index()
    print("per-site best lag of LAI against the site's own GPP cycle (months LAI is late):",
          {int(k): int(v) for k, v in vc.items()})
    print(f"share of sites at +3: {(np.array(best) == 3).mean():.0%} (n = {len(best)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
