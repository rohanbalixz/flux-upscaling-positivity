"""Step 1: analysis cells = every cell where MetaFlux reports GPP, plus every
tower cell (some towers sit where MetaFlux masks the Arctic)."""
import numpy as np
import xarray as xr

from flux_positivity import config
from flux_positivity.grid import NLAT, NLON
from flux_positivity.towers import load_towers

valid = np.zeros((NLAT, NLON), bool)
for y in (2005, 2012, 2020):
    with xr.open_dataset(config.METAFLUX_NC / f"METAFLUX_GPP_RECO_monthly_{y}.nc") as ds:
        valid |= np.isfinite(ds["GPP"].values).any(axis=0)

tower_cells = np.unique(load_towers()["cell"].values)
mask = valid.copy().ravel()
mask[tower_cells] = True
np.savez_compressed(config.PROC / "mask.npz", cells=np.flatnonzero(mask), metaflux_valid=valid,
                    tower_cells=tower_cells)
print(f"MetaFlux valid cells: {valid.sum()} | analysis cells: {mask.sum()} | tower cells: "
      f"{tower_cells.size} ({(~valid.ravel()[tower_cells]).sum()} outside the MetaFlux mask)")
