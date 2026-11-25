"""Step 8: identification layer (PREREGISTRATION s5; theory in docs/THEORY.md).

theta = mean annual GPP (PgC/yr) over vegetated MetaFlux cells, 2001-2014
climatology. In support: point-identified under A1 (MetaFlux value plugged in).
Outside support: nearest-anchor L-Lipschitz interval, or [0, y_max] with no
smoothness. Relaxing A1 by delta widens every cell-month by +/- delta, with
two calibrations: the pre-registered in-support CV RMSE (includes month-level
noise, an upper envelope) and the systematic site-level part (DEVIATIONS D5).

Writes results/bounds.json.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.spatial import cKDTree

from flux_positivity import config
from flux_positivity.bounds import empirical_slopes, mcshane_whitney, nearest_anchor_interval, total_bounds
from flux_positivity.grid import DAYS_IN_MONTH, cell_area_m2
from flux_positivity.io import load_json, save_json
from flux_positivity.support import Space
from flux_positivity.target import build_target
from flux_positivity.towers import exclusion_matrix, towers_complete

F = config.FEATURES
T = build_target()
S = np.load(config.PROC / "support.npz")
tau = float(S["tau"])
veg = T["veg"]
nv = int(veg.sum())
W = DAYS_IN_MONTH[:, None] * cell_area_m2(T["lat"][veg])[None, :] / 1e15     # gC m-2 d-1 -> PgC
gpp = T["gpp"][:, veg].astype(float)
inside = ~S["outside"]
kg = np.broadcast_to(T["kg"][veg][None, :], (12, nv))

tw = towers_complete()
SP = Space(tw[F].values, "z")
anc = tw.groupby(["Site", "month"]).agg(**{f: (f, "mean") for f in F}, y=(config.TARGET, "mean")).reset_index()
Za, ya = SP(anc[F].values), anc.y.values
y_max = float(tw[config.TARGET].max())
Zt = SP(T["Xc"][:, veg, :].reshape(-1, 5))
dA, iA = (a.reshape(12, nv) for a in cKDTree(Za).query(Zt, k=1, workers=-1))

E, SID = exclusion_matrix(tw)
sid = anc.Site.map(SID).values
lo_f, hi_f = config.SLOPE_DISTANCE_RANGE
slopes = empirical_slopes(Za, ya, sid, E, lo_f * tau, hi_f * tau)
L50, L90 = (float(v) for v in np.percentile(slopes, [50, 90]))
# A1 among the towers themselves: GPP gap of near-identical inputs at different sites
pairs = cKDTree(Za).query_pairs(lo_f * tau, output_type="ndarray")
other_site = ~E[sid[pairs[:, 0]], sid[pairs[:, 1]]]
near_diff = np.abs(ya[pairs[other_site, 0]] - ya[pairs[other_site, 1]])

theta_mf = float((gpp * W).sum())
try:
    delta = float(load_json("H2_H3_surrogate.json")["csv"]["RMSE_inside"])
except FileNotFoundError:
    delta = float("nan")


def bound(L, delta=0.0, mask=None):
    m = np.ones_like(inside, dtype=float) if mask is None else mask.astype(float)
    if np.isinf(L):
        out_lo, out_hi = np.zeros_like(gpp), np.full_like(gpp, y_max)
    else:
        out_lo, out_hi = nearest_anchor_interval(ya[iA], dA, L, 0.0, y_max, delta)
    return total_bounds(gpp, inside, W * m, out_lo, out_hi, delta, 0.0, y_max)


def halfwidth_frac(L):
    lo, hi = bound(L)
    return (hi - lo) / 2 / theta_mf


# DEVIATIONS D5: split the in-support held-out error into a systematic site-level part
# (a site's mean residual: the kind of discrepancy A1 excludes) and month-to-month
# scatter (measurement noise + model misfit). delta_site = RMS of site-mean residuals
# over sites with >= DELTA_SITE_MIN_MONTHS in-support held-out months.
pred = np.load(config.PROC / "surrogate_preds.npz", allow_pickle=True)
resid = np.nanmean(pred["csv"], 0) - pred["y"]
ins = pred["d_train"] <= tau
rdf = pd.DataFrame({"site": pred["site"][ins], "r": resid[ins]})
site_mean = rdf.groupby("site").r.agg(["mean", "size"])
site_mean = site_mean[site_mean["size"] >= config.DELTA_SITE_MIN_MONTHS]
delta_site = float(np.sqrt((site_mean["mean"] ** 2).mean()))
within = rdf[rdf.site.isin(site_mean.index)].groupby("site").r.transform(lambda x: x - x.mean())

res = {"theta_metaflux_veg_PgC": theta_mf, "tau": tau, "y_max": y_max, "n_anchors": int(len(ya)),
       "L50": L50, "L90": L90, "n_slope_pairs": int(len(slopes)),
       "near_pairs_n": int(len(near_diff)), "near_pairs_share_disagree_gt2": float((near_diff > 2).mean()),
       "near_pairs_median_absdiff": float(np.median(near_diff)), "delta_inside_RMSE": delta,
       "delta_site": delta_site, "delta_site_n_sites": int(len(site_mean)),
       "within_site_residual_sd": float(np.sqrt((within ** 2).mean())),
       "mean_inside_residual": float(rdf.r.mean()),
       "share_theta_from_outside": float((gpp * W * ~inside).sum() / theta_mf)}
res["table"] = {}
for a1, dl in (("A1_exact", 0.0), ("A1_plus_minus_delta", delta), ("A1_plus_minus_delta_site", delta_site)):
    for name, L in (("L50", L50), ("L90", L90), ("Linf", np.inf)):
        lo, hi = bound(L, dl)
        res["table"][f"{a1}|{name}"] = {"lo": lo, "hi": hi, "halfwidth_pct": 100 * (hi - lo) / 2 / theta_mf}
res["L_star_10pct"] = float(brentq(lambda L: halfwidth_frac(L) - config.BREAKDOWN_HALFWIDTH, 1e-6, 1e4))
res["L_star_over_L50"] = res["L_star_10pct"] / L50
res["curve"] = [{"L": float(L), "halfwidth_pct": 100 * halfwidth_frac(L)} for L in np.linspace(0, 3 * L90, 61)]
res["regional_L50"] = {}
for g in "ABCDE":
    m = kg == g
    lo, hi = bound(L50, 0.0, m)
    th = float((gpp * W * m).sum())
    res["regional_L50"][g] = {"theta_mf": th, "lo": lo, "hi": hi, "halfwidth_pct": 100 * (hi - lo) / 2 / th,
                              "share_outside_gpp": float((gpp * W * m * ~inside).sum() / th)}
rng = np.random.default_rng(0)
oi = np.flatnonzero(~inside.ravel())
sub = rng.choice(oi, size=min(20000, oi.size), replace=False)
for name, L in (("L50", L50), ("L90", L90)):
    lo, hi, feas = mcshane_whitney(Zt[sub], Za, ya, L)
    res[f"MW_{name}_share_infeasible"] = float((~feas).mean())
    res[f"MW_{name}_median_width_feasible"] = (float(np.median((np.minimum(hi, y_max) - np.maximum(lo, 0))[feas]))
                                               if feas.any() else None)
save_json("bounds.json", res)
print({k: v for k, v in res.items() if k not in ("curve", "table", "regional_L50")})
for k, v in res["table"].items():
    print(f"{k:28s} [{v['lo']:.1f}, {v['hi']:.1f}]  +/-{v['halfwidth_pct']:.1f}%")
