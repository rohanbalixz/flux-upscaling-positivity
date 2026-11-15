import numpy as np

from flux_positivity.grid import LAT, LON, NLON, cell_area_m2, cell_index, cell_latlon, era5_to_mf
from flux_positivity.physics import es_hpa, vpd_hpa


def test_era5_to_mf_maps_coordinates_exactly():
    e5_lat = 90.0 - 0.25 * np.arange(721)
    e5_lon = 0.25 * np.arange(1440)
    lat_field = np.repeat(e5_lat[:, None], 1440, 1)
    lon_field = np.repeat(e5_lon[None, :], 721, 0)
    assert np.allclose(era5_to_mf(lat_field)[:, 0], LAT)
    mapped_lon = era5_to_mf(lon_field)[0]
    assert np.allclose(((mapped_lon + 180) % 360) - 180, LON)


def test_cell_index_roundtrip():
    lat, lon = np.array([-33.4648, 42.5378, 0.0]), np.array([-66.4598, -72.1715, 179.9])
    la, lo = cell_latlon(cell_index(lat, lon))
    assert np.all(np.abs(la - lat) <= 0.125)
    assert np.all(np.abs(((lo - lon + 180) % 360) - 180) <= 0.125)
    assert cell_index(0.0, 180.0) == cell_index(0.0, -180.0)        # dateline wraps
    assert cell_index(0, 0) == 360 * NLON + 720


def test_cell_area_sums_to_earth_surface():
    total = (cell_area_m2(LAT + 0.0) * NLON).sum()
    # cells are centred on LAT, so the grid covers -90.125..89.875 and the sum is ~4 pi R^2
    assert abs(total / (4 * np.pi * 6371e3 ** 2) - 1) < 1e-3


def test_saturation_vapour_pressure_reference_values():
    assert abs(es_hpa(0.0) - 6.108) < 0.01
    assert abs(es_hpa(20.0) - 23.37) < 0.05
    assert abs(es_hpa(30.0) - 42.43) < 0.1


def test_vpd_zero_at_saturation_and_never_negative():
    assert vpd_hpa(15.0, 15.0) == 0.0
    assert vpd_hpa(10.0, 12.0) == 0.0
    assert vpd_hpa(25.0, 10.0) > 15
