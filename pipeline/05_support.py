"""Step 5: H1 (where), H5 (after the tower record ends), H6 (normalisation),
robustness variants, domain classifier, drivers, and a site bootstrap.
PREREGISTRATION s3-s4; variant S-LAI per DEVIATIONS D1.

Writes results/support_results.json, support_bootstrap.csv, support_by_year.csv,
outside_support_drivers.json and data/processed/support.npz.
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, ttest_ind

from flux_positivity import config
from flux_positivity.grid import DAYS_IN_MONTH
from flux_positivity.io import save_json
from flux_positivity.stats import interval
from flux_positivity.support import Space, hellinger, loo_neighbourhood_nn, support_distances
from flux_positivity.target import build_target
from flux_positivity.towers import exclusion_matrix, realign_lai, towers_complete

F = config.FEATURES
tw = towers_complete()
E, SID = exclusion_matrix(tw)
site_ids = tw.Site.map(SID).values
Xs = tw[F].values

T = build_target()
veg = T["veg"]
nv = int(veg.sum())
Xt = T["Xc"][:, veg, :]
flatX = Xt.reshape(-1, 5)
area = np.broadcast_to(T["area"][veg][None, :], (12, nv))
gppw = T["gpp"][:, veg] * DAYS_IN_MONTH[:, None] * area        # GPP mass weight
kg = np.broadcast_to(T["kg"][veg][None, :], (12, nv))
SPACE = Space(Xs, "z")

lo, hi = np.percentile(Xs, 1, axis=0), np.percentile(Xs, 99, axis=0)
inside_marg = ((flatX >= lo) & (flatX <= hi)).all(1)


def summarise(outside):
    o = np.asarray(outside).reshape(12, nv)
    r = {"O1_area": float((o * area).sum() / area.sum()), "O2_gpp": float((o * gppw).sum() / gppw.sum())}
    for g in "ABCDE":
        m = kg == g
        r[f"O1_{g}"] = float((o * area * m).sum() / (area * m).sum())
        r[f"O2_{g}"] = float((o * gppw * m).sum() / (gppw * m).sum())
    return r


def boot_rep(seed):
    rng = np.random.default_rng(seed)
    names = tw.Site.unique()
    pick = rng.choice(names, size=len(names), replace=True)
    idx = np.concatenate([np.flatnonzero(tw.Site.values == s) for s in pick])
    d, tau, _ = support_distances(SPACE, Xs[idx], site_ids[idx], E, flatX, config.TAU_PERCENTILE)
    out = d > tau
    r = summarise(out)
    r["P1c"] = float((inside_marg & out).sum() / max(out.sum(), 1))
    r["tau"] = tau
    return r


if __name__ == "__main__":
    rng = np.random.default_rng(config.SEED)
    res = {}

    # ---------------- primary ------------------------------------------------------
    d, tau, dl = support_distances(SPACE, Xs, site_ids, E, flatX, config.TAU_PERCENTILE)
    out = d > tau
    prim = summarise(out)
    prim.update(tau=tau, P1c_share_outside_but_inside_all_marginals=float((inside_marg & out).sum() / out.sum()),
                share_inside_all_marginals_overall=float(inside_marg.mean()),
                hellinger={f: hellinger(Xs[:, j], np.ones(len(Xs)), flatX[:, j], area.ravel())
                           for j, f in enumerate(F)},
                n_tower_site_months=int(len(Xs)), n_sites=int(tw.Site.nunique()), n_veg_cells=nv)
    res["primary"] = prim
    print("primary:", {k: round(v, 4) for k, v in prim.items() if isinstance(v, float)})

    # ---------------- site bootstrap ----------------------------------------------
    with Pool(10) as p:
        boots = p.map(boot_rep, [int(s) for s in rng.integers(0, 2**31, config.N_BOOT)])
    bdf = pd.DataFrame(boots)
    bdf.to_csv(config.RESULTS / "support_bootstrap.csv", index=False)
    res["bootstrap"] = {c: interval(bdf[c]) for c in bdf.columns}
    res["bootstrap"]["P_B_greater_than_A"] = float((bdf.O1_B > bdf.O1_A).mean())

    # ---------------- robustness ------------------------------------------------------
    rob = {f"tau_q{q}": summarise(d > np.percentile(dl, q)) for q in config.TAU_ROBUSTNESS}
    for kind in ("logp", "rank", "maha"):
        dk, tk, _ = support_distances(Space(Xs, kind), Xs, site_ids, E, flatX, config.TAU_PERCENTILE)
        rob[kind] = summarise(dk > tk)
        rob[kind]["spearman_d_with_primary"] = float(spearmanr(dk[::50], d[::50])[0])
    al = realign_lai(tw).dropna(subset=["LAI"]).reset_index(drop=True)
    da, ta, _ = support_distances(Space(al[F].values, "z"), al[F].values, al.Site.map(SID).values, E, flatX,
                                  config.TAU_PERCENTILE)
    rob["S-LAI_realigned"] = summarise(da > ta)
    res["robustness"] = rob

    # ---------------- domain classifier (practical overlap) --------------------------
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    samp = rng.choice(flatX.shape[0], size=config.CLASSIFIER_TARGET_SAMPLE, replace=False,
                      p=area.ravel() / area.sum())
    Zs, Zt = SPACE(Xs), SPACE(flatX)
    Xc_ = np.vstack([Zs, Zt[samp]])
    yc = np.r_[np.ones(len(Zs)), np.zeros(len(samp))]
    oof, score = np.zeros(len(yc)), np.zeros(len(Zt))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xc_, yc):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0).fit(Xc_[tr], yc[tr])
        oof[te] = clf.predict_proba(Xc_[te])[:, 1]
        score += clf.predict_proba(Zt)[:, 1] / 5
    flag = score < np.percentile(oof[yc == 1], 100 - config.TAU_PERCENTILE)
    res["domain_classifier"] = {"auc_towers_vs_pixels": float(roc_auc_score(yc, oof)),
                                "O1_area_classifier": float((flag.reshape(12, nv) * area).sum() / area.sum()),
                                "agreement_with_distance_rule": float((flag == out).mean()),
                                "spearman_d_vs_score": float(spearmanr(d[::50], -score[::50])[0])}

    # ---------------- drivers: which input separates outside points from towers ------
    dd, j = cKDTree(Zs).query(Zt[out], k=1, workers=-1)
    diff = Zt[out] - Zs[j]
    dom = np.abs(diff).argmax(1)
    kgo = kg.ravel()[out]
    drivers = {}
    for g in ["A", "C", "E", "all"]:
        m = (kgo == g) if g != "all" else np.ones(out.sum(), bool)
        drivers[g] = {"dominant_dim_share": {F[k]: float((dom[m] == k).mean()) for k in range(5)},
                      "median_raw_inputs_outside": {F[k]: float(np.median(flatX[out][m][:, k])) for k in range(5)}}
    drivers["tower_p1"] = {f: float(v) for f, v in zip(F, lo)}
    drivers["tower_p99"] = {f: float(v) for f, v in zip(F, hi)}
    save_json("outside_support_drivers.json", drivers)

    # ---------------- H5: per-year inputs, LAI at climatology -----------------------
    py = T["peryear"][:, :, veg, :]                    # (Y, 12, nv, 4): P, VPD, T, SW
    lai = T["Xc"][:, veg, 1]
    tree = cKDTree(Zs)
    yearly = []
    for yi, y in enumerate(T["years"]):
        Xy = np.stack([py[yi, ..., 0], lai, py[yi, ..., 1], py[yi, ..., 2], py[yi, ..., 3]], -1).reshape(-1, 5)
        o = (tree.query(SPACE(Xy), k=1, workers=-1)[0] > tau).reshape(12, nv)
        yearly.append({"year": int(y), "O1_area": float((o * area).sum() / area.sum()),
                       "O2_gpp": float((o * gppw).sum() / gppw.sum()),
                       "mean_T_veg": float((Xy[:, 3].reshape(12, nv) * area).sum() / area.sum())})
    ydf = pd.DataFrame(yearly)
    ydf.to_csv(config.RESULTS / "support_by_year.csv", index=False)
    pre, post = ydf[ydf.year <= 2014], ydf[ydf.year >= 2015]
    res["H5"] = {"O1_mean_2001_2014": float(pre.O1_area.mean()), "O1_mean_2015_2022": float(post.O1_area.mean()),
                 "O2_mean_2001_2014": float(pre.O2_gpp.mean()), "O2_mean_2015_2022": float(post.O2_gpp.mean()),
                 "welch_p": float(ttest_ind(post.O1_area, pre.O1_area, equal_var=False).pvalue),
                 "trend_O1_per_decade": float(np.polyfit(ydf.year, ydf.O1_area, 1)[0] * 10)}

    # ---------------- H6: per-entity standardisation (exploratory) ------------------
    g = tw.groupby("Site")[F]
    Zs_site = ((tw[F] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).values
    okr = np.isfinite(Zs_site).all(1)
    ys = (T["years"] >= config.CLIM_YEARS[0]) & (T["years"] <= config.CLIM_YEARS[1])
    stats = py[ys].reshape(-1, nv, 4)
    mu, sd = np.empty((nv, 5)), np.empty((nv, 5))
    mu[:, [0, 2, 3, 4]], sd[:, [0, 2, 3, 4]] = stats.mean(0), stats.std(0)
    mu[:, 1], sd[:, 1] = lai.mean(0), lai.std(0)
    Zt_cell = ((Xt - mu[None]) / np.where(sd > 0, sd, np.nan)[None]).reshape(-1, 5)
    okt = np.isfinite(Zt_cell).all(-1)
    tau6 = float(np.percentile(loo_neighbourhood_nn(Zs_site[okr], site_ids[okr], E), config.TAU_PERCENTILE))
    d6 = np.full(okt.size, np.inf)
    d6[okt] = cKDTree(Zs_site[okr]).query(Zt_cell[okt], k=1, workers=-1)[0]
    res["H6_per_entity"] = summarise(d6 > tau6)

    np.savez_compressed(config.PROC / "support.npz", d=d.reshape(12, nv).astype(np.float32), tau=tau,
                        outside=out.reshape(12, nv), veg_cells=T["cells"][veg], dl_towers=dl,
                        inside_marg=inside_marg.reshape(12, nv),
                        classifier_score=score.reshape(12, nv).astype(np.float32))
    save_json("support_results.json", res)
    print("H5:", res["H5"])
    print("saved results/support_results.json")
