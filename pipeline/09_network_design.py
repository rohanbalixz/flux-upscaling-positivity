"""Step 9 (PREREGISTRATION s6): where would 10 new towers shrink extrapolation most?

A hypothetical tower in cell c adds c's 12 climatological input vectors to the
support. Objective: GPP mass of currently-outside cell-months brought within
tau. Lazy greedy max-coverage; baseline = random vegetated cells.

Writes results/network_design.json.
"""
import numpy as np
from scipy.spatial import cKDTree

from flux_positivity import config
from flux_positivity.grid import DAYS_IN_MONTH, cell_latlon
from flux_positivity.io import save_json
from flux_positivity.support import Space
from flux_positivity.target import build_target
from flux_positivity.towers import towers_complete

rng = np.random.default_rng(7)
T = build_target()
S = np.load(config.PROC / "support.npz")
tau = float(S["tau"])
veg = T["veg"]
nv = int(veg.sum())
cells = T["cells"][veg]
Z = Space(towers_complete()[config.FEATURES].values, "z")(T["Xc"][:, veg, :])     # (12, nv, 5)
w = (T["gpp"][:, veg] * DAYS_IN_MONTH[:, None] * T["area"][veg][None, :]).astype(float)
out = S["outside"]
oi = np.flatnonzero(out.ravel())
wo = w.ravel()[oi]
tree = cKDTree(Z.reshape(-1, 5)[oi])
total_out = float(wo.sum())


def covered(ci):
    lists = tree.query_ball_point(Z[:, ci, :], r=tau)
    return np.unique(np.concatenate([np.asarray(x, int) for x in lists]))


cand_cells = np.flatnonzero(out.any(0))
cand = rng.choice(cand_cells, size=min(config.NETWORK_CANDIDATES, cand_cells.size), replace=False)
cov = {int(c): covered(c) for c in cand}
got = np.zeros(len(oi), bool)
gains = {c: wo[v].sum() for c, v in cov.items()}
chosen = []
for k in range(config.NETWORK_K):
    best, bestg = None, -1.0
    for c in sorted(gains, key=gains.get, reverse=True):          # lazy greedy
        g = wo[cov[c][~got[cov[c]]]].sum()
        gains[c] = g
        if g >= bestg:
            best, bestg = c, g
        if bestg >= max((gains[x] for x in gains if x != c), default=0):
            break
    got[cov[best]] = True
    la, lo = cell_latlon(cells[best])
    chosen.append({"rank": k + 1, "lat": float(la), "lon": float(lo), "koppen": str(T["kg"][veg][best]),
                   "cum_share_of_outside_gpp_recovered": float(wo[got].sum() / total_out)})
    del gains[best]

rand = []
for _ in range(config.NETWORK_RANDOM_DRAWS):
    g = np.zeros(len(oi), bool)
    for c in rng.choice(nv, size=config.NETWORK_K, replace=False):
        g[covered(c)] = True
    rand.append(float(wo[g].sum() / total_out))

res = {"k": config.NETWORK_K, "greedy_share_recovered": chosen[-1]["cum_share_of_outside_gpp_recovered"],
       "random_share_recovered_mean": float(np.mean(rand)),
       "random_share_recovered_range": [float(min(rand)), float(max(rand))],
       "O2_before": total_out / float(w.sum()), "O2_after_greedy": (total_out - float(wo[got].sum())) / float(w.sum()),
       "sites": chosen, "n_candidates": int(len(cand))}
save_json("network_design.json", res)
print({k: v for k, v in res.items() if k != "sites"})
