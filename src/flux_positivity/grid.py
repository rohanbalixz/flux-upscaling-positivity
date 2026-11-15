"""The MetaFlux 0.25 degree grid and mappings onto it.

Convention (MetaFlux product): 720 x 1440 cells, latitude ascending from -90.0
to 89.75, longitude from -180.0 to 179.75, cell centres at those coordinates.
This is the ERA5 0.25 degree lattice, so ERA5 fields map onto it by an index
permutation with no interpolation.
"""
from __future__ import annotations

import numpy as np

NLAT, NLON = 720, 1440
LAT = -90.0 + 0.25 * np.arange(NLAT)
LON = -180.0 + 0.25 * np.arange(NLON)
DAYS_IN_MONTH = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
EARTH_RADIUS_M = 6371e3


def era5_to_mf(a: np.ndarray) -> np.ndarray:
    """Map an ERA5 field (..., 721 lat 90 -> -90, 1440 lon 0 -> 359.75) onto
    the MetaFlux grid (..., 720 lat -90 -> 89.75, 1440 lon -180 -> 179.75)."""
    rows = 720 - np.arange(NLAT)              # lat -90 is ERA5 row 720
    cols = (np.arange(NLON) + 720) % 1440     # lon -180 is ERA5 column 720
    return a[..., rows[:, None], cols[None, :]]


def cell_index(lat, lon) -> np.ndarray:
    """Flat index of the cell whose centre is nearest (lat, lon)."""
    r = np.clip(np.round((np.asarray(lat) + 90.0) / 0.25).astype(int), 0, NLAT - 1)
    c = np.round((np.asarray(lon) + 180.0) / 0.25).astype(int) % NLON
    return r * NLON + c


def cell_latlon(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cells = np.asarray(cells)
    return LAT[cells // NLON], LON[cells % NLON]


def cell_area_m2(lat: np.ndarray) -> np.ndarray:
    """Exact area of a 0.25 x 0.25 degree cell centred at `lat` (m2)."""
    lat = np.asarray(lat, float)
    return (EARTH_RADIUS_M ** 2 * np.deg2rad(0.25)
            * (np.sin(np.deg2rad(lat + 0.125)) - np.sin(np.deg2rad(lat - 0.125))))
