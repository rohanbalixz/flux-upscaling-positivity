<div align="center">

# Where does MetaFlux extrapolate, and does its uncertainty notice?

**A pre-registered positivity audit of tower-based carbon-flux upscaling**

[![ci: passed](https://img.shields.io/badge/ci-passed-brightgreen.svg)](https://github.com/rohanbalixz/flux-upscaling-positivity/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Pre-registered](https://img.shields.io/badge/hypotheses-pre--registered-success.svg)](docs/PREREGISTRATION.md)

</div>

MetaFlux ([Nathaniel, Liu & Gentine, *Sci. Data* 2023](https://doi.org/10.1038/s41597-023-02349-y)) learns GPP from five inputs at about 200 eddy-covariance towers and applies the learned map to every 0.25° land cell. That step is valid only where two conditions hold:

- **Positivity:** each land cell's inputs resemble inputs some tower has seen.
- **Conditional invariance:** at the same inputs, towers and land share the same GPP response.

This repository measures where the first condition fails, in MetaFlux's own input space. It then tests with tower labels whether failing it matters, asks whether the product's ensemble spread notices, and states what the towers alone can and cannot say about global GPP.

<p align="center"><img src="figures/fig1_support_map.png" width="900" alt="Map of calendar months per year in which each vegetated cell's inputs fall outside the support of MetaFlux's 206 training towers"></p>

## Key findings

**1. About 9% of vegetated land-months fall outside the support of the 206 training towers, carrying about 13% of MetaFlux's GPP.**
- Land-months outside: 9.0% (95% CI 4.9–18.0%). GPP outside: 13.1% (7.0–27.1%).
- Most of the out-of-support GPP is tropical: 17.4% of tropical land-months and 20.7% of tropical GPP fall outside. Polar/alpine climates have the largest outside share (31% of land-months) but little GPP.
- Arid (3.1%) and continental (2.2%) climates are well covered.
- FLUXCOM's richer predictor space showed the opposite tropical pattern (Jung et al. 2020).

**2. 63% of that extrapolation is invisible to a one-variable check.** Every input sits inside the tower range; only the combination is new. The typical case is a wet (8 mm/d), warm (25 °C), humid, bright and leafy tropical month. Climate-invariant ML (Beucler et al. 2024) matches input distributions one variable at a time; these months suggest checking joint overlap as well.

**3. MetaFlux's ensemble spread does not notice.**
- GPP_std rises with distance to support (Spearman 0.32), but only because GPP does. Controlling for GPP, the correlation is −0.01.
- Relative spread is *lower* outside support (2.3%) than inside (2.8%).
- On held-out towers, a surrogate seed ensemble understates error 8.6× and covers 21% at a nominal 95%.

**4. The product's most extreme values are extrapolations.** 98.5% of cell-months above the highest monthly GPP any tower recorded lie outside support, against a 7.7% base rate.

**5. From the towers alone, global GPP (MetaFlux: 125.8 PgC yr⁻¹) is bounded to ±15% under conditional invariance plus smoothness.**
- The bound is ±30% with no smoothness at all.
- Relax invariance by the observed cross-tower error and it becomes about ±84%. That sets δ to the in-support cross-validated RMSE, which includes measurement noise, so ±84% is an upper envelope.
- See [What the towers alone can say](#what-the-towers-alone-can-say-about-global-gpp).

**6. Ten well-placed towers would recover 39% of the out-of-support GPP; ten random ones recover 8%.** All ten greedy picks are tropical: Borneo, Sumatra, the Congo basin, Indochina and the Amazon.

**7. A data note on the public training files.** In the sample training CSVs, `LAI` and `EVI` run about three months behind the same rows' GPP: 132 of 189 sites are best aligned at +3 months.
- In a surrogate, realigning LAI raises held-out R² from 0.53 to 0.63.
- Whether the production model used these files is unknown. See [`docs/DATA_NOTE_LAI.md`](docs/DATA_NOTE_LAI.md).

## Check the data note in one minute

```bash
python scripts/check_lai_phase.py     # numpy + pandas; reads only the public MetaFlux CSVs
```
```
Pooled within-site correlation of X(t) with GPP(t + k)
k:              -4     -3     -2     -1      0      1      2      3      4
SW_IN_ERA    -0.50  -0.23   0.11   0.42   0.61   0.63   0.48   0.20  -0.14
TA_ERA       -0.23   0.10   0.40   0.60   0.63   0.49   0.22  -0.11  -0.41
LAI           0.61   0.71   0.52   0.18  -0.15  -0.39  -0.52  -0.53  -0.45
EVI           0.60   0.72   0.53   0.17  -0.16  -0.38  -0.49  -0.49  -0.42
```

This check also runs in CI against the live MetaFlux repository.

## Pre-registered predictions and outcomes

Hypotheses, thresholds and falsifiers were fixed in [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) before any outcome was computed.

- **A test guards the file.** It is hash-locked, and [`tests/test_preregistration.py`](tests/test_preregistration.py) fails if it changes.
- **The ordering is self-reported.** The hash was recorded in the same working session as the analysis, not by a third-party registry.
- **Every departure is logged** in [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md).
- **Failed predictions stay in the table.**

| | Prediction | Result (95% interval) | Outcome |
|---|---|---|---|
| P1a | ≥ 10% of vegetated land-months outside support | 9.0% (4.9–18.0%) | not met (narrowly) |
| P1b | Arid > tropical, following Jung et al. 2020 | tropical 17.4% vs arid 3.1%; arid > tropical in 0 of 200 bootstraps | **falsified** |
| P1c | ≥ 25% of outside points inside every marginal range | 63% (46–71%) | met |
| P2 | Held-out RMSE outside / inside ≥ 1.5 (falsifier < 1.1) | 1.18 (0.98–1.38); GBM 1.21 | not met, not falsified |
| P2b | Site-level partial Spearman (distance, RMSE given GPP) > 0 | 0.09 | met (weak) |
| P3a | Spearman (seed spread, \|error\|) < 0.3 | 0.41 (0.37–0.44) | not met |
| P3b | RMSE / RMS spread > 2, and larger outside support | 8.6 (7.6–9.7); 9.2 inside vs 5.2 outside | first part met, second not |
| P4 | Spearman (GPP_std, distance) < 0.3 (falsifier ≥ 0.5) | 0.32 (0.30–0.35) | not met, not falsified |
| P4b | Same, partial given GPP, < 0.3 | −0.01 (−0.03 to 0.02) | met |
| P4c | Relative spread outside / inside < 1.5 | 0.85 (0.81–0.88) | met |
| P5 | More extrapolation after FLUXNET2015 ends (2015–22 vs 2001–14) | 12.1% vs 11.8% (p = 0.06) | met on paper; negligible |
| H6 | Exploratory: per-site standardisation collapses support | 11.6% outside; held-out RMSE +15% | expectation half wrong |

Every number in this README is generated in [`results/SUMMARY.md`](results/SUMMARY.md) from the result files.

## What the towers alone can say about global GPP

Upscaling is a transport problem: $\theta=\sum_i w_i\,\mathbb E[Y\mid X=x_i,S=0]$, with towers as the source $S=1$. It rests on two assumptions:

- **A1 (no $S\to Y$):** towers and land share $\mathbb E[Y\mid X]$.
- **A2:** positivity.

Where A2 fails, the towers identify nothing about those land-months, and a selection diagram that treats the inputs as a single node cannot help; finer structure over the inputs might. An $L$-Lipschitz assumption gives sharp McShane–Whitney bounds, whose width grows as $L\,\cdot$ distance to support ([`docs/THEORY.md`](docs/THEORY.md)).

| Assumption | median tower slope $L_{50}$ | 90th-percentile slope $L_{90}$ | no smoothness |
|---|---|---|---|
| A1 exact | 123–135 PgC yr⁻¹ (±4.9%), *rejected by the towers* | 113–152 (±15%) | 109–184 (±30%) |
| A1 relaxed by δ (cross-tower error) | ±75% | ±84% | ±94% |

- **Rejecting $L_{50}$:** at the median slope, the towers' own data are inconsistent with the Lipschitz class for 88% of out-of-support points, while at $L_{90}$ only 5% are.
- **Breakdown value:** a ±10% bound needs $L \le L^* = 5.2$, which is 2.2× the median slope.
- **Evidence against exact A1 among the towers:** 31% of near-identical input pairs at different sites differ by more than 2 gC m⁻² d⁻¹.

## Figures

| | |
|:---:|:---:|
| <img src="figures/fig2_by_climate_and_joint_vs_marginal.png" width="440" alt="Out-of-support share by Köppen group, and joint versus marginal split"> | <img src="figures/fig3_uncertainty_does_not_notice.png" width="440" alt="MetaFlux GPP_std and relative spread against distance; surrogate error and spread against distance"> |
| Extrapolation by climate, and how much a one-variable check misses | Ensemble spread against distance to support |
| <img src="figures/fig5_bound_vs_L.png" width="440" alt="Half-width of the bound on global GPP against the Lipschitz constant"> | <img src="figures/fig7_network_design.png" width="440" alt="Ten greedily placed towers on the support map"> |
| Bound on global GPP against the smoothness constant | Where ten new towers help most |
| <img src="figures/fig4_lai_phase_offset.png" width="440" alt="Seasonal cycles of GPP, MODIS LAI and training-file LAI at deciduous towers"> | <img src="figures/fig6_by_year.png" width="440" alt="Out-of-support share by year, 2001 to 2022"> |
| LAI phase in the sample training files | No drift out of support after the tower record ends |

## Approach

| Component | Choice |
|---|---|
| Source | 15,948 tower site-months, using the exact `xcolumns` of the MetaFlux notebooks |
| Target | Every vegetated 0.25° cell × month, built from ERA5 (WeatherBench2 / ARCO-ERA5) and full-resolution MODIS MOD15A2H LAI, mirroring the paper's upscaling inputs. Validated against the training inputs at tower cells (r = 0.73–0.99). |
| Support | Pooled z-scores (paper Eq. 2). A cell-month is outside if it is farther from every tower month than τ, the 95th percentile of each tower's distance to towers more than 55 km away. 200-replicate site bootstrap; τ, metric and representation varied in robustness checks; a domain classifier gives AUC 0.91. |
| Labels | A surrogate of MetaFlux's MLP (2 × 350, 5 seeds) plus gradient boosting, under spatially blocked 10-fold CV with whole 55 km site clusters held out |
| Product | The MetaFlux v3.1 monthly GPP and GPP_std, with a 5° block bootstrap |

Details are in [`docs/METHODS.md`](docs/METHODS.md); theory is in [`docs/THEORY.md`](docs/THEORY.md).

## Repository layout

```
├── src/flux_positivity/     # the package: grid, physics, Köppen, support, bounds, data builders
│   └── config.py            # every analysis parameter, keyed to the pre-registration
├── pipeline/                # numbered steps 00–11 (make all)
├── scripts/check_lai_phase.py
├── tests/                   # unit tests, including a guard on the pre-registration hash
├── results/                 # JSON/CSV outputs, SUMMARY.md, run_info.json
├── figures/
└── docs/                    # PREREGISTRATION, DEVIATIONS, METHODS, THEORY, DATA_NOTE_LAI, ROADMAP, REPRODUCE
```

## Reproduce

```bash
python -m pip install -e ".[inputs,dev]"
make test        # seconds
make check-lai   # ~1 minute
make all         # ~35 minutes, public data only, no credentials
```

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for step timings, data licences and the determinism notes.

## Limitations

- **Monthly vectors only.** MetaFlux's LSTM members see 30-step sequences, so monthly-vector support is a lower bound on extrapolation.
- **A surrogate, not the production model.** MetaFlux's trained weights are not public; the surrogate tests the input space and the sites.
- **Different reanalysis versions.** Tower inputs are FLUXNET2015's ERA-Interim-based columns, while gridded inputs are ERA5; the size of the mismatch is quantified.
- **Bounds outside support only.** Inside support the bounds take MetaFlux's values as given.

Next steps, including the same positivity question for DifferLand's 2081–2099 space-for-time projection, are in [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Related work

- Jung et al. 2020, *Biogeosciences* 17:1343 (FLUXCOM extrapolation index and severity index).
- Meyer & Pebesma 2021, *Methods Ecol. Evol.* (area of applicability).
- Pallandt et al. 2022, *Biogeosciences* (pan-Arctic network design).
- Fries et al. 2026, arXiv 2605.19812 (FLUXtrapolation benchmark).
- Rozanov et al. 2026, KDD (CarbonBench).
- Beucler et al. 2024, *Sci. Adv.* (climate-invariant ML).
- Jalaldoust, Bellot & Bareinboim 2024, NeurIPS (partial transportability).
- Pearl & Bareinboim 2011, 2014 (transportability).

## Citation

If you use this work, please cite MetaFlux and this repository ([`CITATION.cff`](CITATION.cff)):

```bibtex
@software{bali_flux_positivity_2026,
  author  = {Bali, Rohan},
  title   = {Where does MetaFlux extrapolate, and does its uncertainty notice?
             A pre-registered positivity audit of tower-based carbon-flux upscaling},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/rohanbalixz/flux-upscaling-positivity}
}
```

## Acknowledgements

This work builds entirely on openly released data and code:

- MetaFlux (J. Nathaniel, J. Liu, P. Gentine), its training files and product;
- FLUXNET2015 and its site teams;
- ERA5 via WeatherBench2 and ARCO-ERA5 (Google Research);
- MODIS LAI via Microsoft Planetary Computer.

Any errors are mine.
