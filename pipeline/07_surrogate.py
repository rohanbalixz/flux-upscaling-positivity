"""Step 7, H2 (does leaving support cost accuracy?) and H3 (does a seed ensemble
notice?) with a surrogate of MetaFlux's MLP under spatially blocked 10-fold CV
(towers within 55 km held out together). ~12 min on 10 cores.

Configurations (all evaluated on the same rows):
  csv      pooled z, CSV inputs as trained                       (primary)
  aligned  pooled z, LAI realigned by 3 months                   (DEVIATIONS D1)
  nolai    pooled z, LAI dropped                                 (interpretation)
  persite  per-site z, the public dataloader convention          (H6)
  gbm      gradient-boosted trees on csv inputs                  (second model class, H2)
Cluster-bootstrap intervals for P2 and P3 per DEVIATIONS D2.

Writes results/H2_H3_surrogate.json and data/processed/surrogate_preds.npz.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402

from flux_positivity import config  # noqa: E402
from flux_positivity.io import save_json  # noqa: E402
from flux_positivity.stats import group_bootstrap_indices, interval, partial_spearman  # noqa: E402
from flux_positivity.support import Space  # noqa: E402
from flux_positivity.towers import realign_lai, site_clusters, towers_complete  # noqa: E402

F, SEEDS = config.FEATURES, list(config.SURROGATE_SEEDS)
CFGS = ["csv", "aligned", "nolai", "persite", "gbm"]
tw = towers_complete()
al = realign_lai(tw)
rows = al.LAI.notna().values
tw, al = tw[rows].reset_index(drop=True), al[rows].reset_index(drop=True)
cluster = site_clusters(tw)
y = tw[config.TARGET].values
tau = float(np.load(config.PROC / "support.npz")["tau"])
folds = list(GroupKFold(config.CV_FOLDS).split(tw, groups=cluster))


def design(cfg):
    if cfg in ("csv", "gbm"):
        return tw[F].values
    if cfg == "aligned":
        return al[F].values
    if cfg == "nolai":
        return tw[[f for f in F if f != "LAI"]].values
    g = tw.groupby("Site")[F]
    return np.nan_to_num(((tw[F] - g.transform("mean")) / g.transform("std")).values)


def fit_predict(cfg, k, seed):
    tr, te = folds[k]
    X = design(cfg)
    if cfg != "persite":
        sp = Space(X[tr], "z")
        Xtr, Xte = sp(X[tr]), sp(X[te])
    else:
        Xtr, Xte = X[tr], X[te]
    if cfg == "gbm":
        m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=seed)
    else:
        m = MLPRegressor(hidden_layer_sizes=config.MLP_HIDDEN, activation="relu", alpha=1e-4, batch_size=256,
                         learning_rate_init=1e-3, max_iter=300, early_stopping=True, validation_fraction=0.1,
                         n_iter_no_change=15, random_state=seed)
    m.fit(Xtr, y[tr])
    return cfg, k, seed, te, m.predict(Xte)


def metrics(err, sd, o, site_mean, yv):
    r = {"RMSE": float(np.sqrt(np.mean(err ** 2))), "R2": float(1 - np.mean(err ** 2) / np.var(yv)),
         "RMSE_inside": float(np.sqrt(np.mean(err[~o] ** 2))), "RMSE_outside": float(np.sqrt(np.mean(err[o] ** 2))),
         "bias_inside": float(err[~o].mean()), "bias_outside": float(err[o].mean())}
    r["P2_ratio_out_in"] = r["RMSE_outside"] / r["RMSE_inside"]
    rel = err / np.maximum(site_mean, 0.5)
    r["rel_RMSE_ratio_out_in"] = float(np.sqrt(np.mean(rel[o] ** 2)) / np.sqrt(np.mean(rel[~o] ** 2)))
    if sd is not None:
        r["P3a_spearman_spread_abs_err"] = float(spearmanr(sd, np.abs(err))[0])
        r["P3b_calibration_ratio_all"] = float(r["RMSE"] / np.sqrt(np.mean(sd ** 2)))
        r["P3b_calibration_ratio_inside"] = float(r["RMSE_inside"] / np.sqrt(np.mean(sd[~o] ** 2)))
        r["P3b_calibration_ratio_outside"] = float(r["RMSE_outside"] / np.sqrt(np.mean(sd[o] ** 2)))
        r["coverage_95_gaussian"] = float(np.mean(np.abs(err) <= 1.96 * sd))
    return r


if __name__ == "__main__":
    Xp = tw[F].values
    SP = Space(Xp, "z")
    d_train = np.empty(len(tw))
    for tr, te in folds:
        d_train[te] = cKDTree(SP(Xp[tr])).query(SP(Xp[te]), k=1)[0]
    o = d_train > tau
    site_mean = tw.groupby("Site")[config.TARGET].transform("mean").values

    jobs = [(c, k, s) for c in CFGS for k in range(config.CV_FOLDS) for s in (SEEDS if c != "gbm" else [0])]
    out = Parallel(n_jobs=10, verbose=2)(delayed(fit_predict)(*j) for j in jobs)
    P = {c: np.full((len(SEEDS), len(tw)), np.nan) for c in CFGS}
    for c, _fold, s, te, p in out:
        P[c][SEEDS.index(s) if c != "gbm" else 0, te] = p

    res = {"n_rows": int(len(tw)), "n_sites": int(tw.Site.nunique()), "n_clusters": int(len(np.unique(cluster))),
           "tau": tau, "share_heldout_outside": float(o.mean())}
    rng = np.random.default_rng(config.SEED)
    for c in CFGS:
        pm = P[c][0] if c == "gbm" else np.nanmean(P[c], 0)
        sd = None if c == "gbm" else np.nanstd(P[c], 0)
        err = pm - y
        r = metrics(err, sd, o, site_mean, y)
        st = pd.DataFrame({"Site": tw.Site, "d": d_train, "e2": err ** 2, "g": y}).groupby("Site").agg(
            d=("d", "median"), rmse=("e2", lambda v: np.sqrt(v.mean())), g=("g", "mean"))
        r["P2b_site_partial_spearman_d_rmse_given_gpp"] = partial_spearman(st.d, st.rmse, st.g)
        if c in ("csv", "gbm"):                       # D2: cluster bootstrap intervals
            bs = []
            for _ in range(config.N_BOOT):
                idx = group_bootstrap_indices(cluster, rng)
                if o[idx].sum() < 10:
                    continue
                rb = metrics(err[idx], None if sd is None else sd[idx], o[idx], site_mean[idx], y[idx])
                bs.append(rb)
            bdf = pd.DataFrame(bs)
            r["cluster_bootstrap"] = {k: interval(bdf[k]) for k in bdf.columns
                                      if k.startswith(("P2", "P3", "RMSE", "coverage"))}
        res[c] = r
        print(c, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)})

    pm, sd = np.nanmean(P["csv"], 0), np.nanstd(P["csv"], 0)
    err = pm - y
    q = np.quantile(d_train, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(q, d_train, side="right") - 1, 0, 9)
    res["csv_by_distance_decile"] = [{"decile": int(k), "d_median": float(np.median(d_train[dec == k])),
                                      "rmse": float(np.sqrt(np.mean(err[dec == k] ** 2))),
                                      "rms_spread": float(np.sqrt(np.mean(sd[dec == k] ** 2)))} for k in range(10)]
    save_json("H2_H3_surrogate.json", res)
    np.savez_compressed(config.PROC / "surrogate_preds.npz", **P, y=y, d_train=d_train,
                        site=tw.Site.values, month=tw.month.values)
