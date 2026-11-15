import numpy as np

from flux_positivity.koppen import koppen_main

DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def classify(t_monthly, p_mm_per_day, lat):
    t = np.asarray(t_monthly, float)[:, None]
    p = (np.asarray(p_mm_per_day, float) * DAYS)[:, None]
    return koppen_main(t, p, np.array([lat]))[0]


def test_reference_climates():
    assert classify([27] * 12, [8] * 12, -3) == "A"                                   # Manaus-like
    assert classify([20, 23, 27, 31, 33, 32, 30, 29, 30, 30, 25, 21], [0.05] * 12, 13) == "B"  # Sahel
    assert classify([5, 6, 8, 11, 14, 17, 19, 19, 16, 12, 8, 5], [2] * 12, 49) == "C"       # Paris
    assert classify([-8, -7, -2, 6, 13, 17, 19, 17, 11, 5, -1, -6], [1.7] * 12, 56) == "D"  # Moscow
    assert classify([-20, -18, -15, -8, 0, 5, 8, 7, 2, -6, -14, -18], [0.8] * 12, 70) == "E"  # tundra


def test_summer_half_is_hemisphere_aware():
    t = [15] * 12
    nh_summer_rain = [0.1] * 3 + [3] * 6 + [0.1] * 3
    # identical series, opposite hemispheres -> different arid thresholds are allowed but both valid groups
    assert classify(t, nh_summer_rain, 40) in "BC"
    assert classify(t, nh_summer_rain, -40) in "BC"
