"""Every analysis parameter in one place.

Values were fixed in docs/PREREGISTRATION.md before any outcome was computed;
the section is given in brackets. Changing a value here changes a
pre-registered choice and must be logged in docs/DEVIATIONS.md.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---- paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TOWERS = DATA / "towers"                       # MetaFlux training CSVs (fetched)
PROC = DATA / "processed"                      # derived arrays (generated)
RAW = Path(os.environ.get("FLUXPOS_RAW", str(DATA / "raw")))   # large downloads
METAFLUX_NC = RAW / "metaflux"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
LOGS = ROOT / "logs"

# ---- model inputs [s2] ---------------------------------------------------------
FEATURES = ["P_ERA", "LAI", "VPD_ERA", "TA_ERA", "SW_IN_ERA"]   # order of the public notebooks
TARGET = "GPP_NT_VUT_REF"

# ---- target inputs [s2] ---------------------------------------------------------
CLIM_YEARS = (2001, 2014)          # climatology: tower record x product overlap
ERA5_YEARS = (2001, 2022)          # WeatherBench2 daily copy ends 2023-01-10
ERA5_SAMPLE_EVERY = 3              # read every 3rd day
ALBEDO_DAYS = (5, 15, 25)          # ARCO-ERA5 forecast albedo, 12 UTC
LAI_YEARS = (2008, 2012)           # MOD15A2H years
LAI_COMPOSITE_DAYS = (1, 17)       # two composites per month, max taken
VEG_MAX_NONVEG_SHARE = 0.5         # barren + snow + water share of 500 m pixels
VEG_MIN_LAIMAX = 0.3               # m2/m2, max monthly LAI

# ---- support [s3] -------------------------------------------------------------
NEIGHBOURHOOD_KM = 55.0            # towers closer than this are not independent evidence
TAU_PERCENTILE = 95
TAU_ROBUSTNESS = (90, 99)
N_BOOT = 200
SEED = 20261115
CLASSIFIER_TARGET_SAMPLE = 100_000

# ---- surrogate [s4, H2-H3] ---------------------------------------------------------
CV_FOLDS = 10
SURROGATE_SEEDS = (0, 1, 2, 3, 4)
MLP_HIDDEN = (350, 350)            # MetaFlux MLP hidden size

# ---- deviations ------------------------------------------------------------------
LAI_REALIGN_MONTHS = 3             # DEVIATIONS.md D1

# ---- identification layer [s5] -----------------------------------------------------
SLOPE_DISTANCE_RANGE = (0.5, 2.0)  # x tau, pairs used for empirical Lipschitz slopes
BREAKDOWN_HALFWIDTH = 0.10         # L* = largest L with half-width <= 10% of theta
DELTA_SITE_MIN_MONTHS = 12         # DEVIATIONS D5: sites need >= 12 in-support held-out months

# ---- network design [s6] -------------------------------------------------------------
NETWORK_K = 10
NETWORK_RANDOM_DRAWS = 20
NETWORK_CANDIDATES = 6000

for _p in (PROC, RESULTS, FIGS, LOGS):
    _p.mkdir(parents=True, exist_ok=True)
