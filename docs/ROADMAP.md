# Roadmap

What this audit leaves open, ordered by expected value.

## 1. Uncertainty that grows where support ends

**The problem.** Both MetaFlux's ensemble spread and a surrogate seed ensemble fail to widen outside support. The surrogate understates held-out error 8.6×, with 21% coverage at a nominal 95%.

**Candidates:**

- conformal prediction under covariate shift, weighted by the domain classifier's density ratio;
- distance-aware ensembles;
- the Lipschitz intervals of [`THEORY.md`](THEORY.md) used directly as prediction intervals.

**How each would be judged.** Coverage reported separately inside and outside support, on spatially blocked held-out towers.

## 2. Invariance by design for land fluxes

Climate-invariant ML (Beucler et al. 2024) picks input transformations so that marginal distributions overlap across climates.

**The question.** This audit shows that 63% of the extrapolation is invisible to a per-variable check. Which land-surface transformations maximise *joint* support across regions and climates while preserving spatially blocked skill?

**Candidates:**

- fluxes scaled by VPD or potential evapotranspiration;
- LAI-normalised GPP;
- aridity indices;
- relative rather than absolute radiation.

The latent-heat scaling of Beucler et al. was weaker over land, so this is open.

## 3. Sequence-level support

The LSTM and BiLSTM members see 30-step windows, so support should also be computed on lagged windows (for example 3 months of each input). Monthly-vector support is only a lower bound on extrapolation.

## 4. The same audit on FLUXCOM-X / X-BASE inputs

Jung et al. (2020) found the humid tropics well represented in FLUXCOM's predictor space; in MetaFlux's five-input space they hold most of the out-of-support GPP. Running both input spaces through the same pipeline would show which predictors close the tropical gap.

## 5. Temporal positivity for DifferLand's space-for-time projection

DifferLand (Fang, Bowman, Zhao, Lian & Gentine, arXiv 2411.09654) learns maps from environment to parameters and projects the parameters to 2081–2099 under SSP2-4.5 by space-for-time substitution. The authors note that it is open whether those relationships hold under climate change.

**The positivity question.** Hold each pixel's vegetation, soil, age and height fixed. Does its 2081–2099 predictor vector fall inside the support of the 2001–2023 vectors the network learned from?

**Plan:**

1. Reproduce DifferLand's predictor matrix from the authors' driver files, and confirm that only MAT and MAP change in the projection.
2. Build future MAT and MAP by delta change from at least five CMIP6 SSP2-4.5 models (Pangeo CMIP6 on GCS), comparing 2081–2099 with 2001–2023 in each model.
3. Compute support with the same leave-region-out threshold as here, using blocks that match DifferLand's 4°×5° held-out patches.
4. Report the share of vegetated land, and of present-day flux, that goes off-support, with the across-GCM range, by vegetation type, and split into joint versus marginal.
5. Prior work on novel climates: Williams, Jackson & Kutzbach (2007, *PNAS*).

**Prediction, fixed before running.** Novelty will concentrate in boreal and tundra regions and in the hottest tropical forests, and a univariate MAT check will miss a large share of it.

## 6. Network design with real constraints

Repeat the greedy placement of [`results/network_design.json`](../results/network_design.json) under practical constraints: access, existing ICOS / AmeriFlux / AsiaFlux infrastructure, and cost.

**Variant objective.** Choose sites to minimise the *bound width* on regional GPP rather than the out-of-support mass. That ties tower placement directly to how well the regional total is identified.
