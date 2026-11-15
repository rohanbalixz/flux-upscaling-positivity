# Data note: LAI phase in the MetaFlux sample training files

**Summary.** In the public sample training files of MetaFlux, the `LAI` and `EVI` columns run about three months behind the same rows' GPP, shortwave and temperature. This note documents the evidence and gives a one-minute check.

**What it does not claim.** It is not known whether the production MetaFlux models were trained on these files. The note concerns the public sample data only, and is offered as a pointer for the MetaFlux team.

## Check it yourself (about one minute)

```bash
python scripts/check_lai_phase.py      # needs numpy + pandas; downloads 206 CSVs (~5 MB)
```

The script compares columns of the same rows with each other. No MODIS, ERA5 or other external data are involved. It also runs in CI against the live MetaFlux repository on every push.

## Evidence

**1. Within-file lag correlation.** The table shows the pooled within-site correlation of each column at month $t$ with GPP at month $t+k$, over 2000–2014.

| Column | k = −3 | k = −1 | k = 0 | k = +1 | Peak at |
|---|---|---|---|---|---|
| SW_IN_ERA | −0.23 | 0.42 | 0.61 | 0.63 | +1 |
| TA_ERA | 0.10 | 0.60 | 0.63 | 0.49 | 0 |
| **LAI** | **0.71** | 0.18 | −0.15 | −0.39 | **−3** |
| **EVI** | **0.72** | 0.17 | −0.16 | −0.38 | **−3** |

Shortwave and temperature behave as expected, while LAI and EVI peak three months early. LAI and EVI agree with each other at lag 0 (r = 0.79), so the offset is shared by the two remote-sensing columns.

**2. Per site.**

| Comparison | Sites where the CSV LAI is best aligned at +3 months |
|---|---|
| Against the site's own GPP seasonal cycle | 132 of 189 |
| Against MODIS MOD15A2H LAI in the 3×3 (1.5 km) window at the tower | 120 of 160 |

**3. Deciduous forests make the offset obvious ([Fig. 4](../figures/fig4_lai_phase_offset.png)).** At northern-hemisphere deciduous broadleaf towers (n = 23):

- GPP in the file peaks in June–July.
- MODIS LAI at the towers peaks in June–September.
- The file's LAI peaks in September–November and has a secondary spike in February.
- At Harvard Forest (US-Ha1), the file gives LAI ≈ 5 in September–December and ≈ 1 in June.

**4. Levels are fine.** Site-mean CSV LAI matches site-mean MODIS LAI at the tower closely (r = 0.94). Only the phase is affected.

## Why it matters (surrogate, spatially blocked CV)

| Configuration | Held-out R² | RMSE (gC m⁻² d⁻¹) |
|---|---|---|
| LAI as in the files | 0.53 | 2.67 |
| LAI dropped entirely | 0.51 | 2.73 |
| LAI realigned by +3 months | 0.63 | 2.39 |

As shipped, the LAI column contributes almost no information to a MetaFlux-like MLP. Realigned, it is worth about 0.1 in R².

At upscaling time, a model trained on the shifted series would receive correctly phased MODIS LAI. The input the network learned from and the input it is evaluated on would then differ systematically.

## A plausible origin, not verified

The offset is about three months, consistent across sites, and comes with a February artefact. That pattern looks like a time-indexing step in assembling the MODIS series, for example a composite or day-of-year index mapped to calendar months. It does not look like an ecological signal.
