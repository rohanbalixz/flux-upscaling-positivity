"""Step 2: monthly ERA5 inputs for every analysis cell, 2001-2022 (~10 min).

  python pipeline/02_era5.py            # full build -> data/processed/era5_monthly.npz
  python pipeline/02_era5.py --smoke    # 4 days, writes nothing, checks the code path
"""
import argparse

import numpy as np

from flux_positivity import config
from flux_positivity.data.era5 import build_era5_monthly

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

cells = np.load(config.PROC / "mask.npz")["cells"]
if args.smoke:
    r = build_era5_monthly(cells, years=(2005, 2005), max_days=4)
    print({k: float(np.nanmean(r[k][0, 0])) for k in ("TA_ERA", "VPD_ERA", "P_ERA", "SW_IN_ERA")})
else:
    build_era5_monthly(cells, out=config.PROC / "era5_monthly.npz")
    print("saved", config.PROC / "era5_monthly.npz")
