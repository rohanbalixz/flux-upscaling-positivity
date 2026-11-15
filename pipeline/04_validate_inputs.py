"""Step 4 (PREREGISTRATION s7): do the gridded inputs at tower cells reproduce the
inputs the network was trained on, and is the training LAI in phase?

Writes results/inputs_validation.json, results/lai_tower_comparison.csv,
results/lai_phase_lag_by_site.csv.
"""
import numpy as np
import pandas as pd

from flux_positivity import config
from flux_positivity.io import save_json
from flux_positivity.towers import load_towers

tw = load_towers()
e5 = np.load(config.PROC / "era5_monthly.npz")
pos = {c: i for i, c in enumerate(e5["cells"])}
y0 = int(e5["years"][0])
out = {}

# ---- climate: matched site-year-month, 2001-2014 --------------------------------
sub = tw[(tw.year >= y0) & (tw.year <= 2014)]
ci, yi, mi = sub.cell.map(pos).values, sub.year.values - y0, sub.month.values - 1
rows = []
for v in ["TA_ERA", "VPD_ERA", "P_ERA", "SW_IN_ERA"]:
    g, c = e5[v][yi, mi, ci], sub[v].values
    ok = np.isfinite(g) & np.isfinite(c)
    rows.append(dict(var=v, n=int(ok.sum()), r=float(np.corrcoef(g[ok], c[ok])[0, 1]),
                     bias=float(np.mean(g[ok] - c[ok])), rel_bias=float(np.mean(g[ok] - c[ok]) / np.mean(c[ok])),
                     slope_grid_on_csv=float(np.polyfit(c[ok], g[ok], 1)[0])))
out["climate"] = rows
print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---- LAI: CSV climatology vs MODIS (0.25 deg cell and 3x3 tower window) ---------
L = np.load(config.PROC / "lai_monthly.npz", allow_pickle=True)
lpos = {c: i for i, c in enumerate(L["cells"])}
site_idx = {s: i for i, s in enumerate(L["tower_sites"])}
cc = tw.groupby(["Site", "month"]).agg(LAI=("LAI", "mean"), cell=("cell", "first")).reset_index()
cc["lai_cell"] = [L["lai_clim"][m - 1, lpos[c]] for m, c in zip(cc.month, cc.cell)]
cc["lai_3x3"] = [L["tower_lai_3x3"][site_idx[s], m - 1] for s, m in zip(cc.Site, cc.month)]
cc.to_csv(config.RESULTS / "lai_tower_comparison.csv", index=False)
lrows = []
for col in ["lai_cell", "lai_3x3"]:
    ok = cc[["LAI", col]].notna().all(1)
    lrows.append(dict(compare=f"monthly: CSV vs MODIS {col}", n=int(ok.sum()),
                      r=float(np.corrcoef(cc.LAI[ok], cc[col][ok])[0, 1])))
site = cc.groupby("Site")[["LAI", "lai_cell", "lai_3x3"]].mean().dropna()
for col in ["lai_cell", "lai_3x3"]:
    lrows.append(dict(compare=f"site mean: CSV vs MODIS {col}", n=len(site), r=float(site.LAI.corr(site[col]))))
out["lai_level"] = lrows
print(pd.DataFrame(lrows).round(3).to_string(index=False))

# ---- LAI phase: within-file lags (independent of MODIS) and per-site best lag ----
t2 = tw[tw.year >= 2000].copy()
t2["k"] = t2.year * 12 + t2.month


def pooled_lag_corr(a, b, k):
    """Pooled within-site anomaly correlation of a(t) with b(t + k)."""
    x = t2[["Site", "k", a]].dropna()
    y = t2[["Site", "k", b]].dropna().assign(k=lambda d: d.k - k)
    m = x.merge(y, on=["Site", "k"], suffixes=("_a", "_b"))
    col_a, col_b = (f"{a}_a", f"{b}_b") if a == b else (a, b)
    m[col_a] -= m.groupby("Site")[col_a].transform("mean")
    m[col_b] -= m.groupby("Site")[col_b].transform("mean")
    return float(np.corrcoef(m[col_a], m[col_b])[0, 1])


lags = range(-4, 5)
out["phase_within_file"] = {f"{a}~{b}": {int(k): pooled_lag_corr(a, b, k) for k in lags}
                            for a, b in [("LAI", config.TARGET), ("EVI", config.TARGET),
                                         ("SW_IN_ERA", config.TARGET), ("LAI", "EVI")]}
gclim = tw.groupby(["Site", "month"])[config.TARGET].mean().unstack()
best = []
for s, g in cc.groupby("Site"):
    g = g.set_index("month").reindex(range(1, 13))
    a, b = g.LAI.values, g.lai_3x3.values
    row = {"Site": s}
    if not (np.isnan(a).any() or np.isnan(b).any() or np.std(b) < 0.3):
        rs = {k: np.corrcoef(a, np.roll(b, k))[0, 1] for k in range(-5, 7)}
        row["lag_vs_modis"] = max(rs, key=rs.get)
    if s in gclim.index and not np.isnan(a).any() and np.std(gclim.loc[s]) > 0.5:
        gg = gclim.loc[s].values
        rg = {k: np.corrcoef(a, np.roll(gg, k))[0, 1] for k in range(-5, 7)}
        row["lag_vs_gpp"] = max(rg, key=rg.get)
    best.append(row)
bl = pd.DataFrame(best).set_index("Site")
bl.to_csv(config.RESULTS / "lai_phase_lag_by_site.csv")
out["best_lag_counts"] = {c: {int(k): int(v) for k, v in bl[c].value_counts().sort_index().items()}
                          for c in ("lag_vs_modis", "lag_vs_gpp")}
for k, v in out["phase_within_file"].items():
    print(k, {kk: round(vv, 2) for kk, vv in v.items()})
print(out["best_lag_counts"])
save_json("inputs_validation.json", out)
