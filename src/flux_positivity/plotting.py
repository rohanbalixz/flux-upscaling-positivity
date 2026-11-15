"""Figure style: reference categorical slots 1-3 (blue, orange, aqua; validated
all-pairs for colour-vision deficiency) and a single-hue blue ramp for magnitude."""
from __future__ import annotations

import numpy as np

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
RAMP = ["#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]


def apply_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 200,
        "savefig.bbox": "tight", "lines.linewidth": 2})
    return plt


def base_map(ax, land_cells, values=None, cmap=None, norm=None):
    """Equirectangular land map on the MetaFlux grid; `values` is (720, 1440)."""
    from matplotlib.colors import ListedColormap

    from .grid import LAT, LON, NLAT, NLON
    land = np.zeros(NLAT * NLON)
    land[land_cells] = 1
    land = land.reshape(NLAT, NLON)
    ext = [-180, 180, -90, 90]
    ax.imshow(np.where(land > 0, 1.0, np.nan), origin="lower", extent=ext,
              cmap=ListedColormap(["#e6e5df"]), interpolation="nearest")
    if values is not None:
        ax.imshow(values, origin="lower", extent=ext, cmap=cmap, norm=norm, interpolation="nearest")
    ax.contour(LON, LAT, land, levels=[0.5], colors=[AXIS], linewidths=0.4)
    ax.set_xlim(-170, 180)
    ax.set_ylim(-58, 72)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
