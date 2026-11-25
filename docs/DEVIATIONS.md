# Deviations from the pre-registration

Every change made after `PREREGISTRATION.md` was hashed is logged here with its date and the reason for it.

## D1. Added a realigned-LAI variant (15 Nov 2026, before any outcome analysis)

**Finding (from §7 input checks).** In the public MetaFlux training CSVs, the remote-sensing columns `LAI` and `EVI` run about **three months late** relative to the same rows' `GPP_NT_VUT_REF`, `SW_IN_ERA` and `TA_ERA`.

- **Within-file evidence.** The pooled within-site anomaly correlation between LAI(t) and GPP(t + k) peaks at k = −3 (r = 0.71). It is 0.18 at k = 0.
  - EVI shows the same pattern (0.72 at k = −3).
  - SW and GPP peak at k = 0 to +1, as expected.
  - LAI and EVI agree with each other at k = 0 (r = 0.79).
- **Per-site evidence.** The best lag of the CSV LAI is +3 months at:
  - 112 of 155 sites against the site's own GPP seasonal cycle;
  - 120 of 160 sites against MODIS MOD15A2H LAI in the 3×3 window at the tower.
- **Example.** Northern-hemisphere deciduous broadleaf sites show GPP peaking in June–July while CSV LAI peaks in September–November. There is also an isolated February spike in both LAI and EVI.
- **Site means still agree.** At the site-mean level the CSV LAI matches MODIS in the 3×3 window (r = 0.94). The *level* is right; the *phase* is shifted.
- **Files.** `results/lai_tower_comparison.csv`, `results/lai_phase_lag_by_site.csv`.

**What it does and doesn't show.** This concerns the public sample files in `leap-stc/metaflux`. Whether the production MetaFlux models were trained on the same files is unknown, so it is raised as a question, not a claim about the product.

**Change.**

- The pre-registered primary analysis keeps the CSV columns as they are, because that is what the network was trained on.
- Added variant **S-LAI**: within each site, LAI is realigned by three months, so that LAI_aligned(t) = LAI_csv(t + 3 months). It is used as a robustness check on H1 and as an extra surrogate configuration in H2 and H3, which measures the cross-validated error cost of the offset.
- No hypothesis, threshold or falsifier was changed.

## D2. Added uncertainty intervals for P2–P4 (15 Nov 2026, after the first run; reporting only)

The pre-registration gave intervals only for H1. Two sets were added after the first run:

- **P2 and P3.** A cluster bootstrap over held-out towers, resampling 55 km site clusters; 200 replicates.
- **P4.** A 5° spatial block bootstrap; 200 replicates. With 1.9 M spatially correlated cell-months, naive tests would be meaningless.

Point estimates, thresholds and the verdicts in the outcome table are unchanged. The intervals sit alongside them.

## D3. Code reorganised into a package (15 Nov 2026, after the first run)

**What changed.** The analysis was first run as standalone scripts. It was then reorganised into the `flux_positivity` package with `pipeline/` entry points, every parameter moved to `config.py`, and unit tests added. All results were then regenerated from the package.

**Check.** All 35 headline quantities compared against the first run were reproduced exactly. They cover every H1 share and τ, the bootstrap intervals, robustness variants, classifier AUC, H4–H6, every surrogate metric, all bounds and L values, and network design.

**One count redefined.** The per-site count of the LAI lag against each site's own GPP cycle is now computed over every site with a clear GPP cycle: 132 of 189 sites at +3 months. The first run counted only sites that also passed the MODIS seasonality filter, giving 112 of 155. Both counts are recorded; D1 keeps the original figure.

## D4. Editorial edit to the pre-registration (15 Nov 2026, after the analysis)

Two references to material outside this repository were removed from `PREREGISTRATION.md`: one sentence in §0 and the qualifier of the P4 label in §4. No hypothesis, prediction, threshold, falsifier or analysis choice changed. The file was re-hashed; `PREREGISTRATION.sha256` keeps the original hash and the time it was recorded.

## D5. A second calibration of δ, and precision fixes (24 Nov 2026, reporting only)

**Why.** The pre-registered δ, the in-support cross-validated RMSE of 2.64 gC m⁻² d⁻¹, mixes systematic site-level discrepancy with month-to-month measurement noise and model misfit. Only the systematic part corresponds to a violation of conditional invariance (A1).

**Added.**

- The held-out in-support residuals are split into a site-level part and within-site scatter:
  - δ_site = 1.40, the RMS over 191 sites (each with at least 12 in-support held-out months) of each site's mean residual;
  - within-site scatter = 2.32;
  - mean residual across towers = −0.02.
- The bounds are reported for δ_site as an extra row (±44%, ±53% and ±65% at L50, L90 and no smoothness), computed from the saved surrogate predictions in `pipeline/08_bounds.py`.
- The pre-registered δ row is unchanged and still reported.

**Precision fixes made at the same time.**

- **Vegetated-land GPP.** The bounded quantity is vegetated-land GPP within MetaFlux's mask, with in-support values taken from MetaFlux, not "global GPP from the towers alone".
- **Nearest-anchor interval.** The reported bound is this interval, not the sharp McShane–Whitney envelope.
- **Infeasibility.** An infeasible L is described as inconsistent with noisy anchors, not "rejected".
- **Support threshold.** "Outside support" is defined by the threshold τ.
- **Marginal ranges.** Ranges are the towers' 1st–99th percentiles.
- **Counts.** The count (67 cell-months) is given for the above-tower-maximum result.
- **P2 interval.** P2's interval spans its falsifier threshold.
- **Temperature bias.** Temperature bias is given in °C.

Only figure 5's labels changed; no other figure or result number changed.
