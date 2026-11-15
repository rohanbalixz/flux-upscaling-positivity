# flux-upscaling-positivity: one command per step. `make help` lists targets.
PY ?= python
export PYTHONHASHSEED := 0

.PHONY: help install check-lai fetch inputs analysis figures report all test lint smoke clean

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-11s %s\n", $$1, $$2}'

install:  ## install the package with input and dev extras
	$(PY) -m pip install -e ".[inputs,dev]"

check-lai:  ## verify the LAI phase offset from the public CSVs (~1 min, numpy + pandas only)
	$(PY) scripts/check_lai_phase.py

fetch:  ## step 0: tower CSVs + MetaFlux v3.1 monthly product (1.9 GB, checksum verified)
	bash pipeline/00_fetch.sh

inputs:  ## steps 1-4: mask, ERA5 (~10 min), MODIS LAI (~10 min), input validation
	$(PY) pipeline/01_mask.py
	$(PY) pipeline/02_era5.py
	$(PY) pipeline/03_lai.py
	$(PY) pipeline/04_validate_inputs.py

analysis:  ## steps 5-9: support, product uncertainty, surrogate (~12 min), bounds, network design
	$(PY) pipeline/05_support.py
	$(PY) pipeline/06_product_uncertainty.py
	$(PY) pipeline/07_surrogate.py
	$(PY) pipeline/08_bounds.py
	$(PY) pipeline/09_network_design.py

figures:  ## step 10
	$(PY) pipeline/10_figures.py

report:  ## step 11: results/SUMMARY.md and results/run_info.json
	$(PY) pipeline/11_report.py

all: fetch inputs analysis figures report  ## everything, from scratch (~35 min)

smoke:  ## exercise the download code paths on a few days / tiles (needs internet)
	$(PY) pipeline/02_era5.py --smoke
	$(PY) pipeline/03_lai.py --smoke

test:  ## unit tests (synthetic data, no downloads)
	$(PY) -m pytest

lint:
	ruff check src pipeline scripts tests

clean:  ## remove generated caches (keeps fetched raw data)
	rm -rf data/processed logs .pytest_cache
