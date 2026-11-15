"""Small physical helpers."""
from __future__ import annotations

import numpy as np


def es_hpa(t_c):
    """Saturation vapour pressure over water (hPa), Tetens formula; t in deg C."""
    t_c = np.asarray(t_c, float)
    return 6.1078 * np.exp(17.27 * t_c / (t_c + 237.3))


def vpd_hpa(t_c, td_c):
    """Vapour pressure deficit es(T) - es(Td) in hPa, floored at zero, as the
    MetaFlux paper computes it from ERA5 air and dewpoint temperature."""
    return np.maximum(es_hpa(t_c) - es_hpa(td_c), 0.0)
