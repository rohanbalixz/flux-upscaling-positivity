"""Step 3: monthly MODIS LAI climatology for every analysis cell (~10 min).

  python pipeline/03_lai.py             # full build -> data/processed/lai_monthly.npz
  python pipeline/03_lai.py --smoke     # 2 tiles, writes nothing
"""
import argparse

import numpy as np

from flux_positivity import config
from flux_positivity.data.modis_lai import build_lai_monthly
from flux_positivity.towers import load_towers

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

cells = np.load(config.PROC / "mask.npz")["cells"]
towers = (load_towers().groupby("Site").agg(lat=("Lat", "first"), lon=("Lon", "first")).reset_index())
if args.smoke:
    r = build_lai_monthly(cells, towers, max_tiles=2)
    print("cells with LAI:", int(np.isfinite(r["lai_clim"]).any(0).sum()))
else:
    build_lai_monthly(cells, towers, out=config.PROC / "lai_monthly.npz")
    print("saved", config.PROC / "lai_monthly.npz")
