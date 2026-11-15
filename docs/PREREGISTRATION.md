# Where does MetaFlux extrapolate, and does its uncertainty notice?

**Pre-registration and analysis plan** · written Sun 15 Nov 2026, before any outcome analysis was run.

## 0. What had been seen when this was written

To keep the pre-registration honest, here is everything inspected before the hypotheses below were fixed:

- **Tower table.** 206 FLUXNET2015 sites from the public MetaFlux repo: 189 "base" and 17 "target" (all Tropical). There are 17,916 site-months from 1991 to 2014, of which 15,948 have all five inputs and GPP.
- **Product.** Grid and value ranges of the MetaFlux v3.1 monthly product for 2012. GPP runs from 1.10 to 45.1 gC m⁻² d⁻¹ with a mean of 3.09. GPP_std has a mean of 0.097.
- **LAI spot check.** One tile (h18v04) showed the training CSV's July LAI differing from MODIS 500 m LAI at the tower at about 30 European sites.
- **Prior literature.** Jung et al. (2020) found that FLUXCOM's predictor space represents the humid tropics well, while extremely cold and dry regions are represented worst. This prior is used below.

No support distance, error, correlation or bound had been computed.

## 1. Question and framing

MetaFlux (Nathaniel, Liu & Gentine, *Sci. Data* 2023) learns GPP from five inputs at about 200 eddy-covariance towers and applies the learned map to every 0.25° land cell. The inputs are precipitation, LAI, VPD, air temperature and incoming shortwave.

Upscaling is a transport problem (Pearl & Bareinboim 2011, 2014). Let *S* = 1 mark tower site-months and *S* = 0 mark land pixel-months. The upscaled value of a regional total is

  θ = Σ_cells area · Σ_months E[GPP | X = x, S = 1] evaluated at the pixel inputs x.

That equals the true regional total under two separable conditions:

- **A1, conditional invariance (no S → Y).** E[GPP | X, S = 1] = E[GPP | X, S = 0]. Towers are representative *given their inputs*.
- **A2, positivity.** Every pixel-month input vector lies within the support of tower inputs.

A1 is untestable outside the support. A2 is testable from inputs alone. This study:

- measures where A2 fails, in MetaFlux's own input space;
- tests with tower labels whether failing A2 is associated with larger error;
- tests whether the ensemble spread that MetaFlux ships notices; and
- states what can be said about global GPP when A2 fails and only a smoothness assumption is added.

The same two conditions organise Beucler et al. (2024, *Sci. Adv.*). There, physically motivated input transformations are chosen to make distributions overlap across climates (A2), and the transformed mapping must still transfer (A1).

## 2. Data (fixed)

| Role | Data | Detail |
|---|---|---|
| Source (training) | MetaFlux repo `metaflux/data/sample/{base,target}`, 206 sites | Columns `P_ERA, LAI, VPD_ERA, TA_ERA, SW_IN_ERA` → `GPP_NT_VUT_REF`, monthly. These are the variables of the public notebooks. |
| Target inputs, climate | ERA5 daily means at 0.25° (WeatherBench2 copy), every 3rd day, 2001–2022 | T2m (°C). VPD = es(T) − es(Td) in hPa, as the paper describes. P in mm/d. SW_in = SW_net / (1 − forecast albedo), with albedo from ARCO-ERA5 at 12 UTC on days 5, 15 and 25. |
| Target inputs, LAI | MODIS MOD15A2H C6.1, full-resolution 500 m tiles (Planetary Computer), years 2008 and 2012 | Two composites per month and the max taken, then averaged over the two years. Mean of valid pixels per cell. |
| Product | MetaFlux v3.1 monthly (Zenodo 10970911), 2001–2022 | GPP and GPP_std, ensemble of 3 architectures × 5 seeds. |
| Vegetated land | MetaFlux valid cells with (barren + snow + water) MODIS share < 0.5 and max monthly LAI ≥ 0.3 | Area weight cos(lat). |
| Climate zones | Köppen main groups A/B/C/D/E computed from the ERA5 2001–2014 climatology | Beck et al. (2018) criteria for the main groups. |

**Primary target set.** The vegetated cells × 12 calendar months, using the 2001–2014 climatology (the overlap of the tower record and the product).

**Why this is conservative.** Climatological months are less extreme than individual months, while the source set keeps individual site-months. Any out-of-support fraction found this way is therefore a lower bound for individual months.

## 3. Feature space and support (fixed)

- **Representation (primary).** Pooled z-normalisation with the mean and SD of the tower site-months. This is Eq. 2 of the paper ("expected value of the variable X on our training data").
  - *Robustness:* log(1 + P) in place of P; a rank (quantile) transform; Mahalanobis distance with the tower covariance.
  - *Exploratory:* per-site standardisation, which is what the public `dataloader.py` line 112 does when it is applied file by file.
- **Distance.** Euclidean distance d(x) from a target vector to the nearest tower site-month.
- **Threshold τ.** For each tower site-month, take the distance to the nearest site-month from a site more than 0.5° away (leave-neighbourhood-out). τ is the 95th percentile of those distances.
  - A target pixel-month is **outside support** if d(x) > τ.
  - *Robustness:* τ at the 90th and 99th percentiles, and a domain classifier (gradient-boosted, towers vs pixels, cross-fitted) whose density ratio is reported alongside.
- **Uncertainty.** A site bootstrap with 200 replicates. Each replicate resamples sites with replacement and recomputes τ and every fraction. Results are reported as the median and a 95% interval.

The threshold logic is the dissimilarity-index / area-of-applicability approach of Meyer & Pebesma (2021) and the extrapolation index of Jung et al. (2020). It is used here as a positivity diagnostic, not claimed as new.

## 4. Hypotheses, predictions and falsifiers

### H1. Where (inputs only)

- **Reported quantities:**
  - O1, the area-weighted fraction of vegetated pixel-months outside support;
  - O2, the share of MetaFlux GPP (2001–2014 mean) produced in those pixel-months;
  - O3, both fractions by Köppen group;
  - maps of d and of the number of months outside support per cell.
- **P1a.** O1 ≥ 10%.
- **P1b.** Following Jung et al. 2020, arid (B) has a higher outside fraction than tropical (A). *Falsifier:* A > B.
- **P1c (joint vs marginal).** At least 25% of outside-support pixel-months lie inside the tower 1st–99th percentile range on every single input. In other words, a one-variable-at-a-time overlap check would miss them. *Falsifier:* < 10%.

### H2. Does leaving the support cost accuracy? (tower labels)

- **Surrogate.** An MLP mirroring MetaFlux's MLP: 2 hidden layers of 350 units, pooled z-inputs, MSE loss, 5 seeds. A gradient-boosted tree model serves as a second model class.
- **Cross-validation.** Spatial 10-fold CV. Sites are grouped into clusters with members within 0.5° of each other, and whole clusters are held out together.
- **Measured.** For each held-out site-month: its distance to the training folds, d_train, and its absolute error.
- **P2.** RMSE for held-out site-months with d_train > τ is at least 1.5× the RMSE for those with d_train ≤ τ. *Falsifier:* ratio < 1.1.
- **P2b.** Site-level Spearman between median d_train and RMSE, as a partial correlation controlling for the site's mean GPP, is > 0.

### H3. Does a seed ensemble notice? (tower labels)

Same surrogate and folds. σ_ens is the SD across the 5 seeds.

- **P3a.** Spearman(σ_ens, |error|) < 0.3 across held-out site-months.
- **P3b.** RMSE / RMS(σ_ens) > 2, and larger outside support than inside.
- *Falsifier:* Spearman ≥ 0.5 **and** a ratio ≤ 1.5.

### H4. Does MetaFlux's own GPP_std notice? (product, no labels)

The unit is a vegetated cell-month, using the 2001–2014 climatology of GPP and GPP_std.

- **P4 (primary).** Spearman(GPP_std, d) < 0.3. *Falsifier:* ≥ 0.5, in which case the ensemble already flags extrapolation, and that will be reported as good news for MetaFlux.
- **P4b.** The partial Spearman controlling for GPP is < 0.3.
- **P4c.** The ratio median(GPP_std / GPP) outside support ÷ inside support is < 1.5.
- **Descriptive.** The fraction of cell-months with GPP greater than the highest tower monthly GPP that lie outside support, and the same for cells at the product's floor value.

### H5. After the tower record ends (inputs only, per year)

- **Measured.** The yearly out-of-support fraction O1(year) for 2001–2022, using per-year ERA5 months with LAI held at its climatology. This isolates climate-driven departure.
- **P5.** The mean of O1 over 2015–2022, after FLUXNET2015 ends, exceeds the mean over 2001–2014, and the OLS trend over 2001–2022 is > 0. *Falsifier:* the post-2014 mean is ≤ the 2001–2014 mean.

### H6. Normalisation moves the support (exploratory, no prediction tested)

The analysis is repeated with per-site and per-pixel standardisation, and the H2 surrogate is retrained with per-site standardisation.

**Expectation.** Marginal overlap becomes nearly perfect by construction, and the out-of-support fraction collapses. Cross-validated error rises, because the absolute level of each input is discarded. That is a trade of positivity (A2) against invariance (A1).

This will be asked as a question, which convention the production model used, not asserted.

## 5. Identification layer (definitions fixed; this is a computation, not a test)

- **Estimand.** θ = mean annual GPP (PgC yr⁻¹) over vegetated MetaFlux cells, 2001–2014 climatology.
- **Anchors.** Site × calendar-month means of tower inputs and GPP (about 2.4k points, which reduces noise).
- **Assumptions and the bound each gives:**

| Assumption on f = E[GPP \| X] | In support (d ≤ τ) | Outside support (d > τ) |
|---|---|---|
| A1 + L-Lipschitz | identified; MetaFlux value plugged in | McShane–Whitney envelope [max_s(y_s − L·d(x,x_s)), min_s(y_s + L·d(x,x_s))] ∩ [0, y_max] |
| A1, no smoothness (L = ∞) | identified | [0, y_max] (Manski-type), with y_max = highest tower monthly GPP |
| A1 relaxed: \|E[Y\|X,S=1] − E[Y\|X,S=0]\| ≤ δ | ± δ | envelope ± δ |

- **Choice of δ.** The H2 in-support cross-validated RMSE, used as a plausible scale for a mechanism discrepancy.
- **L.** The data cannot identify L, so it is reported through quantiles of the empirical slopes |y_i − y_j| / ‖x_i − x_j‖ between anchors from different sites at distances in [τ/2, 2τ]: L̂₅₀ and L̂₉₀.
- **Breakdown value L\*.** The largest L for which the bound on θ has a half-width ≤ 10% of MetaFlux's θ. It is compared with L̂₅₀ and L̂₉₀.
- The 2 × 3 table {A1 exact, A1 ± δ} × {L̂₅₀, L̂₉₀, ∞} is the summary of what a causal-invariance assumption buys inside the support versus what smoothness buys outside it.

## 6. Network design (descriptive)

- **Procedure.** Greedily add k = 10 hypothetical towers. Candidates are vegetated cells, each contributing its 12 climatological month vectors. The objective is the reduction in the GPP-weighted out-of-support share (O2).
- **Comparison.** The mean of 20 random sets of 10 vegetated cells.
- **Prior work.** Hargrove et al. 2003; Sulkava et al. 2011; Pallandt et al. 2022, who found that 15 well-placed Arctic sites improve representativeness by about 20%.

## 7. What gets reported regardless of outcome

- Every prediction above, with its pass or fail status.
- Inputs validation: at tower cells, the gridded inputs against the CSV inputs, with r and bias for each variable.
- The CSV LAI against MODIS at the 0.25° cell and at the 3 × 3 tower window.
- All robustness variants.

A failed prediction stays in the summary table and is not moved to an appendix.

## 8. Known limitations (stated in advance)

- **Monthly vectors only.** The LSTM and BiLSTM members see 30-step sequences, so monthly-vector support is a necessary condition for support, not a sufficient one.
- **Different reanalysis versions.** Tower inputs come from FLUXNET2015's ERA-Interim-based `_ERA` columns, while gridded inputs come from ERA5. Part of any source/target difference is a product difference, which §7 quantifies.
- **Approximate SW_in.** SW_in is reconstructed from net SW and albedo. It is validated against SW_IN_ERA at the towers.
- **Possibly different LAI products.** The CSV's LAI may come from a different MODIS product or scale than the one used for upscaling; §7 checks this.
- **Not the production model.** The surrogate in H2 and H3 is not MetaFlux. It tests whether this input space and these sites make distance predictive of error, and whether a seed ensemble notices.
- **A1 is untestable outside the support.** Everything outside the support is a bound, not an estimate.
