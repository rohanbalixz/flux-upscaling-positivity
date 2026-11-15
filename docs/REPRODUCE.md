# Reproducing the analysis

Requirements:

- **Time and hardware.** About 35 minutes on a laptop with 8 or more cores, most of it spent downloading.
- **Space.** About 6 GB of disk.
- **Access.** Only public data, with no accounts or credentials.

```bash
git clone https://github.com/rohanbalixz/flux-upscaling-positivity
cd flux-upscaling-positivity
python -m pip install -e ".[inputs,dev]"     # or: conda env create -f environment.yml
make test                                    # unit tests, seconds
make check-lai                               # the LAI phase check, ~1 minute
make all                                     # fetch -> inputs -> analysis -> figures -> report
```

To keep the 1.9 GB MetaFlux download outside the repository, set `FLUXPOS_RAW=/path/with/space`.

## Steps

| Make target | Script | Time | Output |
|---|---|---|---|
| `fetch` | `pipeline/00_fetch.sh` | ~5 min | tower CSVs → `data/towers/`; MetaFlux v3.1 monthly product (checksum verified) → `data/raw/metaflux/` |
| `inputs` | `01_mask.py` | <1 min | analysis cells |
| | `02_era5.py` | ~10 min | `data/processed/era5_monthly.npz` |
| | `03_lai.py` | ~10 min | `data/processed/lai_monthly.npz` |
| | `04_validate_inputs.py` | <1 min | `results/inputs_validation.json`, LAI comparison CSVs |
| `analysis` | `05_support.py` | ~2 min | H1, H5, H6 and robustness → `results/support_results.json` |
| | `06_product_uncertainty.py` | ~1 min | H4 → `results/H4_product_uncertainty.json` |
| | `07_surrogate.py` | ~12 min | H2, H3 → `results/H2_H3_surrogate.json` |
| | `08_bounds.py` | <1 min | identification layer → `results/bounds.json` |
| | `09_network_design.py` | <1 min | `results/network_design.json` |
| `figures` | `10_figures.py` | <1 min | `figures/*.png` |
| `report` | `11_report.py` | <1 min | `results/SUMMARY.md`, `results/run_info.json` |

`make smoke` runs the two download builders on a few days and tiles, which checks the network code paths without the full download.

## Determinism

All seeds are fixed in `src/flux_positivity/config.py`. Multithreaded BLAS can change the order of floating-point sums, so a rerun of the surrogate (step 7) may differ in the third decimal; every other step is exact.

`results/run_info.json` records the package versions, the time and the commit of each run.

## Data sources and licences

| Data | Licence / access |
|---|---|
| FLUXNET2015 tower data (Pastorello et al. 2020), as redistributed in [leap-stc/metaflux](https://github.com/leap-stc/metaflux) | CC BY 4.0, via the MIT-licensed MetaFlux repository. Fetched, not committed. |
| MetaFlux v3.1 (Nathaniel, Liu & Gentine 2023) | [Zenodo 10970911](https://zenodo.org/records/10970911) |
| ERA5 (Hersbach et al. 2020) | Copernicus licence, via the public [WeatherBench2](https://weatherbench2.readthedocs.io) and [ARCO-ERA5](https://github.com/google-research/arco-era5) copies |
| MODIS MOD15A2H C6.1 (Myneni, Knyazikhin & Park) | NASA LP DAAC, via [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/modis-15A2H-061) |

The MIT licence of this repository covers its code. The data keep their original licences.
