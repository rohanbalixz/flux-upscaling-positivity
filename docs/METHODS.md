# Methods

Each choice below was fixed in [`PREREGISTRATION.md`](PREREGISTRATION.md) unless [`DEVIATIONS.md`](DEVIATIONS.md) says otherwise. Every parameter lives in [`src/flux_positivity/config.py`](../src/flux_positivity/config.py).

## 1. Source: what the network was trained on

| Item | Detail |
|---|---|
| Files | `metaflux/data/sample/{base,target}/*_MM.csv` in [leap-stc/metaflux](https://github.com/leap-stc/metaflux): 189 base and 17 target files, i.e. 206 FLUXNET2015 sites, 1991–2014, monthly |
| Inputs | `P_ERA, LAI, VPD_ERA, TA_ERA, SW_IN_ERA`, the `xcolumns` of the public notebooks |
| Target | `GPP_NT_VUT_REF` |
| Sample | 15,948 site-months with all five inputs and GPP, from 204 of the 206 sites |

## 2. Target: what the network is asked about

The target is the five inputs at every 0.25° cell where MetaFlux v3.1 reports GPP, built to match the paper's upscaling description.

### ERA5

Source: the WeatherBench2 copy of ERA5 daily means at 0.25°, plus ARCO-ERA5. Both are public on Google Cloud Storage. Every third day from 2001 to 2022 is read, giving 2,679 days.

- **Temperature.** Daily mean 2 m temperature.
- **VPD.** $e_s(T) - e_s(T_d)$ from daily means (Tetens), as the paper describes for ERA5.
- **Precipitation.** Daily total.
- **Incoming shortwave.** WeatherBench2 stores *net* surface shortwave, so incoming shortwave is reconstructed as $SW_{net} / (1-\alpha)$, using ARCO-ERA5 forecast albedo sampled at 12 UTC on days 5, 15 and 25 of each month. The divisor is capped at 0.15 over permanent snow.

### MODIS LAI

Source: MOD15A2H C6.1, 500 m, 8-day composites, via Microsoft Planetary Computer.

- **Full-resolution tiles only.** The Planetary Computer COG overviews blend fill codes 249–255 into valid LAI. In a sample tile, about 22,000 2×2 blocks containing fill codes read as valid values. The overviews were therefore not used.
- **Composites.** All 290 tiles, two composites per month (nearest the 1st and the 17th), in 2008 and 2012. Each month takes the larger of its two composites to limit cloud-contaminated retrievals in the wet tropics. The two years are then averaged.
- **Cell value.** The mean of valid 500 m pixels in the cell. The shares of barren (253), snow/ice (252) and water (254) pixels are kept for the vegetated mask.

### Grid

ERA5 and MetaFlux share the 0.25° lattice, so mapping between them is an index permutation with no interpolation (tested in `tests/test_grid_physics.py`).

### Vegetated land

A cell counts as vegetated if MetaFlux has values there, barren + snow + water make up less than half of its 500 m pixels, and its maximum monthly LAI is at least 0.3. That gives 160,170 cells, 72% of MetaFlux's land area.

### Climatology and climate groups

- The primary target uses the 2001–2014 calendar-month climatology, the overlap of the tower record and the product. Climatological months are less extreme than single months, so this choice *understates* extrapolation.
- Köppen main groups are computed from the same ERA5 climatology with Beck et al. (2018) criteria, then checked at 11 reference locations.

### Input validation at tower cells (matched site-year-month, 2001–2014)

| Variable | r | Relative bias |
|---|---|---|
| Air temperature | 0.99 | +3% |
| VPD | 0.97 | −3% |
| Incoming shortwave | 0.97 | +6% |
| Precipitation | 0.73 | +13% |

Precipitation is the weakest link, so `log(1+P)` is a pre-registered robustness variant.

## 3. Support

- **Representation.** Pooled z-scores using tower means and SDs, as in MetaFlux paper Eq. 2: "expected value of the variable X on our training data".
- **Distance.** Euclidean distance from each target cell-month to the nearest tower site-month.
- **Threshold.** For each tower site-month, take the distance to the nearest site-month of a site *more than 55 km away*. Neighbouring towers (for example the CA-NS1–7 chronosequence) are not independent evidence. τ is the 95th percentile of these distances.
- **Rule.** A target cell-month is outside support if $d > τ$. This is the extrapolation-index / area-of-applicability logic (Jung et al. 2020; Meyer & Pebesma 2021), used here as a positivity diagnostic.
- **Reported quantities.**
  - The share of vegetated land-months outside support, weighted by area.
  - The share of MetaFlux GPP mass outside support.
  - Both, broken down by Köppen group.
  - The share of outside points that sit inside the tower 1st–99th percentile range on *every* input. This is the joint-versus-marginal check, the multivariate counterpart of the per-variable Hellinger overlap check in Beucler et al. (2024).
- **Uncertainty.** A site bootstrap with 200 replicates, which resamples sites and recomputes τ and every share.
- **Robustness.**
  - τ at the 90th and 99th percentiles.
  - log(1+P); a rank transform; Mahalanobis distance.
  - LAI realigned by three months (D1).
  - A gradient-boosted domain classifier (towers vs land, cross-fitted) as a density-ratio view of overlap.

## 4. Does leaving the support cost accuracy? (surrogate)

MetaFlux's trained weights are not public, so H2 and H3 use a surrogate that mirrors its MLP: two hidden layers of 350 units, ReLU, Adam, early stopping, pooled z-inputs, MSE, 5 random seeds. A gradient-boosted regressor serves as a second model class.

- **Cross-validation.** Spatially blocked 10-fold CV. Sites within 55 km of each other form a cluster (118 clusters), and whole clusters are held out together.
- **Outputs.** Each held-out tower-month gets a prediction, an error and its distance to the training folds.
- **Configurations.**
  - CSV inputs as trained;
  - LAI realigned (D1);
  - LAI dropped;
  - per-site standardisation, the public dataloader's convention (H6).
- **Intervals.** A cluster bootstrap over held-out towers (D2).

## 5. Does the product's uncertainty notice?

The unit is a vegetated cell × calendar month, using the 2001–2014 climatology of MetaFlux GPP and GPP_std. GPP_std is the spread across 3 architectures × 5 seeds.

- **P4.** The raw Spearman correlation of GPP_std with distance.
- **P4b.** The partial Spearman, controlling for GPP. Spread scales with flux magnitude, and magnitude also varies with distance.
- **P4c.** Relative spread (GPP_std / GPP) outside versus inside support.
- **Intervals.** A 5° spatial block bootstrap (D2). With 1.9 M spatially correlated cell-months, naive p-values would be meaningless.

## 6. Identification layer

See [`THEORY.md`](THEORY.md).

- **Anchors.** Tower site × calendar-month means, 2,448 in all.
- **Bounds outside support.** Nearest-anchor $L$-Lipschitz intervals clipped to $[0, y_{max}]$, where $y_{max}$ is the largest tower monthly GPP.
- **In support.** MetaFlux values are used, which is identification under A1.
- **Reported.**
  - $L_{50}$ and $L_{90}$ from cross-site anchor slopes;
  - the McShane–Whitney infeasible share at each;
  - the breakdown value $L^*$;
  - the 2 × 3 table of {A1 exact, A1 ± δ} × {$L_{50}$, $L_{90}$, no smoothness}.

## 7. Network design

- **Model of a new tower.** A hypothetical tower in cell $c$ contributes $c$'s 12 climatological input vectors to the support.
- **Selection.** Lazy greedy maximum coverage of the out-of-support GPP mass, choosing $k = 10$ from 6,000 random candidate cells that have at least one month outside support.
- **Baseline.** 20 draws of 10 random vegetated cells.
- **Prior work.** Hargrove et al. 2003; Sulkava et al. 2011; Pallandt et al. 2022.

## 8. Limitations

1. **Monthly vectors, not sequences.** The LSTM and BiLSTM members see 30-step sequences, so monthly-vector support is necessary, not sufficient.
2. **Different reanalysis versions.** Tower inputs are FLUXNET2015's ERA-Interim-based `_ERA` columns, while target inputs are ERA5. The validation table above bounds the size of this mismatch.
3. **The surrogate is not MetaFlux.** It tests the input space and the sites, not MetaFlux's weights.
4. **Only part of the result is bounded.** The bounds address the out-of-support part only; inside the support they take MetaFlux's values as given.
5. **LAI coverage.** The LAI climatology uses two years. LAI interannual variability is small next to the seasonal cycle, which is the signal that matters here.
