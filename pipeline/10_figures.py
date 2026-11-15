"""Step 10: all figures, from results/*.json and data/processed caches."""
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

from flux_positivity import config
from flux_positivity.grid import NLAT, NLON
from flux_positivity.io import load_json
from flux_positivity.koppen import GROUP_NAMES
from flux_positivity.plotting import AQUA, BLUE, INK, INK2, MUTED, ORANGE, RAMP, SURF, apply_style, base_map
from flux_positivity.target import build_target
from flux_positivity.towers import load_towers

plt = apply_style()
FIGS = config.FIGS
T = build_target()
S = np.load(config.PROC / "support.npz")
veg = T["veg"]
sup = load_json("support_results.json")
p, b = sup["primary"], sup["bootstrap"]
towers = load_towers()
sites = towers.groupby("Site").agg(lat=("Lat", "first"), lon=("Lon", "first")).reset_index()

grid = np.full(NLAT * NLON, np.nan)
grid[T["cells"][veg]] = S["outside"].sum(0)
grid = grid.reshape(NLAT, NLON)
cmap = ListedColormap(["#f6f5f1"] + RAMP[1:])
norm = BoundaryNorm([-0.5, 0.5, 2.5, 5.5, 8.5, 12.5], cmap.N)


def support_map(title, subtitle, fname, extra=None):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    base_map(ax, T["cells"], grid, cmap, norm)
    if extra:
        extra(ax)
    else:
        ax.scatter(sites.lon, sites.lat, s=6, c=INK, linewidths=0, zorder=5)
    ax.set_title(title, pad=22)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, color=INK2, fontsize=8.5, va="bottom")
    cax = fig.add_axes([0.30, 0.10, 0.42, 0.025])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                      ticks=[0, 1.5, 4, 7, 10.5])
    cb.ax.set_xticklabels(["0", "1–2", "3–5", "6–8", "9–12"])
    cb.outline.set_visible(False)
    cb.set_label("Calendar months per year outside support (2001–2014 climatology)", color=INK2, fontsize=8)
    fig.savefig(FIGS / fname)
    plt.close(fig)


# ---- Fig 1 --------------------------------------------------------------------------
support_map("Where MetaFlux's inputs leave the support of its 206 training towers",
            f"{100*p['O1_area']:.1f}% of vegetated land-months (95% CI {100*b['O1_area'][0]:.1f}–"
            f"{100*b['O1_area'][2]:.1f}%), carrying {100*p['O2_gpp']:.1f}% of MetaFlux GPP, fall outside "
            "support in MetaFlux's 5-input space. Black dots: towers.", "fig1_support_map.png")

# ---- Fig 2 --------------------------------------------------------------------------
groups = list("ABCDE")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [2.2, 1]})
x, wbar = np.arange(5), 0.36
for j, (key, col, lab) in enumerate((("O1", BLUE, "Share of land area"), ("O2", ORANGE, "Share of MetaFlux GPP"))):
    v = np.array([p[f"{key}_{g}"] for g in groups]) * 100
    lo = np.array([b[f"{key}_{g}"][0] for g in groups]) * 100
    hi = np.array([b[f"{key}_{g}"][2] for g in groups]) * 100
    xs = x + (j - 0.5) * (wbar + 0.03)
    a1.bar(xs, v, wbar, color=col, label=lab, zorder=3)
    a1.errorbar(xs, v, yerr=[v - lo, hi - v], fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2, zorder=4)
a1.set_xticks(x)
a1.set_xticklabels([f"{GROUP_NAMES[g]} ({g})" for g in groups])
a1.set_ylabel("% of vegetated land-months outside support")
a1.legend(loc="upper center")
a1.set_title("Extrapolation is concentrated in the tropics and polar/alpine climates")
a1.grid(axis="x", visible=False)
share = p["P1c_share_outside_but_inside_all_marginals"] * 100
a2.barh([0], [share], color=BLUE, height=0.5, zorder=3)
a2.barh([0], [100 - share], left=[share + 0.6], color=ORANGE, height=0.5, zorder=3)
a2.text(share / 2, 0, f"{share:.0f}%\ninside every\nsingle-variable range", ha="center", va="center",
        color="#ffffff", fontsize=8)
a2.text(share + (100 - share) / 2, 0, f"{100-share:.0f}%\noutside\n≥1 range", ha="center", va="center",
        color="#ffffff", fontsize=8)
a2.set_xlim(0, 101)
a2.set_yticks([])
a2.set_xlabel("% of outside-support land-months")
a2.set_title("A one-variable check misses most of it")
a2.grid(False)
fig.tight_layout()
fig.savefig(FIGS / "fig2_by_climate_and_joint_vs_marginal.png")
plt.close(fig)

# ---- Fig 3 --------------------------------------------------------------------------
h4, sg = load_json("H4_product_uncertainty.json"), load_json("H2_H3_surrogate.json")
tau = float(S["tau"])
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11, 3.3))
dec = pd.DataFrame(h4["by_distance_decile"])
a1.plot(dec.d_median, dec.std_median, color=BLUE, marker="o", ms=4)
a1.set_xlabel("Distance to tower support (decile medians)")
a1.set_ylabel("GPP_std (gC m⁻² d⁻¹)")
a1.set_title("MetaFlux spread rises with distance…")
a2.plot(dec.d_median, 100 * dec.cv_median, color=BLUE, marker="o", ms=4)
a2.set_xlabel("Distance to tower support (decile medians)")
a2.set_ylabel("GPP_std / GPP (%)")
a2.set_title("…but only because GPP does")
a2.text(0.98, 0.95, f"partial Spearman given GPP = {h4['P4b_partial_spearman_std_d_given_gpp']:.2f}",
        transform=a2.transAxes, ha="right", va="top", color=INK2, fontsize=8)
sd = pd.DataFrame(sg["csv_by_distance_decile"])
a3.plot(sd.d_median, sd.rmse, color=ORANGE, marker="o", ms=4)
a3.plot(sd.d_median, sd.rms_spread, color=BLUE, marker="o", ms=4)
a3.text(sd.d_median.iloc[-1], sd.rmse.iloc[-1], "  held-out error", color=INK2, va="center", fontsize=8)
a3.text(sd.d_median.iloc[-1], sd.rms_spread.iloc[-1], "  seed spread", color=INK2, va="center", fontsize=8)
a3.set_ylim(0, sd.rmse.max() * 1.12)
a3.axvline(tau, color=MUTED, lw=1, ls=(0, (3, 3)))
a3.text(tau, sd.rmse.max() * 0.6, " τ (support edge)", color=INK2, va="center", fontsize=8)
a3.set_xlabel("Distance of held-out tower-month to training folds")
a3.set_ylabel("gC m⁻² d⁻¹")
a3.set_title("Surrogate: seed spread vs real error")
a3.set_xlim(right=sd.d_median.max() * 1.45)
fig.tight_layout()
fig.savefig(FIGS / "fig3_uncertainty_does_not_notice.png")
plt.close(fig)

# ---- Fig 4 --------------------------------------------------------------------------
dbf = towers[(towers.PFT == "DBF") & (towers.Lat > 30) & (towers.year >= 2000)]
lc = pd.read_csv(config.RESULTS / "lai_tower_comparison.csv")
lc = lc[lc.Site.isin(dbf.Site.unique())].groupby("month").lai_3x3.mean()
m = np.arange(1, 13)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for lab, s, col in [("GPP (training file)", dbf.groupby("month")[config.TARGET].mean(), BLUE),
                    ("LAI, MODIS at the tower", lc, AQUA),
                    ("LAI (training file)", dbf.groupby("month").LAI.mean(), ORANGE)]:
    v = s.reindex(m).values
    ax.plot(m, v / np.nanmax(v), color=col, marker="o", ms=4, label=lab)
ax.set_xticks(m)
ax.set_xticklabels(list("JFMAMJJASOND"))
ax.set_ylabel("Fraction of annual maximum")
ax.set_ylim(0, 1.08)
ax.legend(loc="upper left", fontsize=8)
ax.set_title(f"Deciduous broadleaf towers (n={dbf.Site.nunique()}): training LAI runs ~3 months late")
fig.tight_layout()
fig.savefig(FIGS / "fig4_lai_phase_offset.png")
plt.close(fig)

# ---- Fig 5 --------------------------------------------------------------------------
bd = load_json("bounds.json")
c = pd.DataFrame(bd["curve"])
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(c.L, c.halfwidth_pct, color=BLUE)
ax.set_ylim(0, c.halfwidth_pct.max() * 1.12)
labs = {"L50": f"median slope\n(rejected: {100*bd['MW_L50_share_infeasible']:.0f}% infeasible)",
        "L90": f"90th-pct slope\n({100*bd['MW_L90_share_infeasible']:.0f}% infeasible)"}
for key, yfrac in (("L50", 0.80), ("L90", 1.08)):
    ax.axvline(bd[key], color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(bd[key] + 0.3, c.halfwidth_pct.max() * yfrac, labs[key], color=INK2, va="top", fontsize=7.5)
ax.axhline(10, color=ORANGE, lw=1)
ax.text(c.L.max(), 10, "±10% of MetaFlux total  ", color=INK2, ha="right", va="bottom", fontsize=8)
ax.scatter([bd["L_star_10pct"]], [10], s=40, color=ORANGE, edgecolors=SURF, linewidths=1.5, zorder=5)
ax.text(bd["L_star_10pct"] + 0.4, 8.6, f"L* = {bd['L_star_10pct']:.1f}", color=INK, fontsize=8, va="top")
ax.set_xlabel("Lipschitz constant L (gC m⁻² d⁻¹ per unit input distance)")
ax.set_ylabel("Half-width of bound on global GPP (%)")
ax.set_title("Towers + smoothness: how tightly is global GPP pinned?")
fig.tight_layout()
fig.savefig(FIGS / "fig5_bound_vs_L.png")
plt.close(fig)

# ---- Fig 6 --------------------------------------------------------------------------
y = pd.read_csv(config.RESULTS / "support_by_year.csv")
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.axvspan(2014.5, 2022.5, color="#f0efec", zorder=0)
ax.text(2018.5, 100 * y.O1_area.max() * 1.02, "after FLUXNET2015 ends", ha="center", color=INK2, fontsize=8)
ax.plot(y.year, 100 * y.O1_area, color=BLUE, marker="o", ms=4)
ax.set_ylabel("% vegetated land-months outside")
ax.set_ylim(0, 100 * y.O1_area.max() * 1.15)
ax.set_title("Year-by-year inputs: no material drift out of support after 2014")
fig.tight_layout()
fig.savefig(FIGS / "fig6_by_year.png")
plt.close(fig)

# ---- Fig 7 --------------------------------------------------------------------------
nd = load_json("network_design.json")


def picks(ax):
    ax.scatter(sites.lon, sites.lat, s=5, c=MUTED, linewidths=0, zorder=5)
    for s in nd["sites"]:
        ax.scatter(s["lon"], s["lat"], s=60, marker="D", c=ORANGE, edgecolors=SURF, linewidths=1.2, zorder=6)
        ax.text(s["lon"] + 2.5, s["lat"] + 1.5, str(s["rank"]), color=INK, fontsize=8, weight="bold", zorder=7)


support_map(f"Ten new towers, placed greedily, bring {100*nd['greedy_share_recovered']:.0f}% of "
            "out-of-support GPP back inside",
            f"Orange: greedy picks in order. Grey: existing towers. Random placement of ten towers recovers "
            f"{100*nd['random_share_recovered_mean']:.0f}% on average.", "fig7_network_design.png", extra=picks)
print("figures written to", FIGS)
