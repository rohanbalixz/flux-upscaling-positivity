"""Step 6, H4: does MetaFlux's own ensemble spread (GPP_std) rise where its
inputs leave tower support? Unit: vegetated cell x calendar month (2001-2014
climatology). Intervals: 5-degree spatial block bootstrap (DEVIATIONS D2).

Writes results/H4_product_uncertainty.json.
"""
import numpy as np
from scipy.stats import rankdata, spearmanr

from flux_positivity import config
from flux_positivity.io import save_json
from flux_positivity.stats import interval, partial_spearman
from flux_positivity.target import build_target
from flux_positivity.towers import towers_complete

T = build_target()
S = np.load(config.PROC / "support.npz")
veg = T["veg"]
nv = int(veg.sum())
d = S["d"].ravel().astype(float)
out = S["outside"].ravel()
g = T["gpp"][:, veg].ravel().astype(float)
s = T["gpp_std"][:, veg].ravel().astype(float)
kg = np.broadcast_to(T["kg"][veg][None, :], (12, nv)).ravel()
block = np.broadcast_to(((np.floor((T["lat"][veg] + 90) / 5)) * 100
                         + np.floor((T["lon"][veg] + 180) / 5))[None, :], (12, nv)).ravel()
ok = np.isfinite(g) & np.isfinite(s) & np.isfinite(d)
d, out, g, s, kg, block = d[ok], out[ok], g[ok], s[ok], kg[ok], block[ok]
cv = s / g

res = {"n_cell_months": int(ok.sum()),
       "P4_spearman_std_d": float(spearmanr(s, d)[0]),
       "P4b_partial_spearman_std_d_given_gpp": partial_spearman(s, d, g),
       "P4c_ratio_median_cv_out_over_in": float(np.median(cv[out]) / np.median(cv[~out])),
       "median_cv_inside": float(np.median(cv[~out])), "median_cv_outside": float(np.median(cv[out])),
       "spearman_cv_d": float(spearmanr(cv, d)[0])}

# 5-degree block bootstrap of the three statistics (ranks computed once)
rs, rd, rg = rankdata(s), rankdata(d), rankdata(g)
ub, inv = np.unique(block, return_inverse=True)
order = np.argsort(inv)
starts = np.r_[0, np.cumsum(np.bincount(inv))]
rng = np.random.default_rng(config.SEED)
bs = {"P4": [], "P4b": [], "P4c": []}
for _ in range(config.N_BOOT):
    pick = rng.integers(0, len(ub), len(ub))
    idx = np.concatenate([order[starts[b]:starts[b + 1]] for b in pick])
    a, b_, z = rs[idx], rd[idx], rg[idx]
    bs["P4"].append(np.corrcoef(a, b_)[0, 1])
    ea = a - np.polyval(np.polyfit(z, a, 1), z)
    eb = b_ - np.polyval(np.polyfit(z, b_, 1), z)
    bs["P4b"].append(np.corrcoef(ea, eb)[0, 1])
    o = out[idx]
    bs["P4c"].append(np.median(cv[idx][o]) / np.median(cv[idx][~o]))
res["block_bootstrap_5deg"] = {k: interval(v) for k, v in bs.items()}

res["by_koppen"] = {k: {"spearman_std_d": float(spearmanr(s[kg == k], d[kg == k])[0]),
                        "partial_given_gpp": partial_spearman(s[kg == k], d[kg == k], g[kg == k])}
                    for k in "ABCDE"}
tw = towers_complete()
gmax, g99 = float(tw[config.TARGET].max()), float(np.percentile(tw[config.TARGET], 99))
res.update(tower_max_monthly_gpp=gmax, tower_p99_monthly_gpp=g99,
           n_cell_months_above_tower_max=int((g > gmax).sum()),
           share_of_above_max_outside_support=float(out[g > gmax].mean()),
           n_cell_months_above_tower_p99=int((g > g99).sum()),
           share_of_above_p99_outside_support=float(out[g > g99].mean()),
           share_outside_overall=float(out.mean()))
q = np.quantile(d, np.linspace(0, 1, 11))
dec = np.clip(np.searchsorted(q, d, side="right") - 1, 0, 9)
res["by_distance_decile"] = [{"decile": int(k), "d_median": float(np.median(d[dec == k])),
                              "std_median": float(np.median(s[dec == k])),
                              "gpp_median": float(np.median(g[dec == k])),
                              "cv_median": float(np.median(cv[dec == k]))} for k in range(10)]
save_json("H4_product_uncertainty.json", res)
print({k: v for k, v in res.items() if k not in ("by_distance_decile", "by_koppen")})
