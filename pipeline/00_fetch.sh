#!/usr/bin/env bash
# Step 0: fetch the inputs the pipeline does not generate.
#   1. MetaFlux tower training files (206 FLUXNET2015 sites), leap-stc/metaflux
#   2. MetaFlux v3.1 monthly product (Zenodo 10970911, 1.9 GB zip)
# ERA5 and MODIS LAI are streamed by steps 2 and 3.
set -euo pipefail
cd "$(dirname "$0")/.."
RAW="${FLUXPOS_RAW:-$PWD/data/raw}"
mkdir -p data/towers/base data/towers/target "$RAW/metaflux"

python3 scripts/check_lai_phase.py --download-only

ZIP="$RAW/metaflux/METAFLUX_GPP_RECO_monthly.zip"
if [ ! -f "$ZIP" ]; then
  curl -L --retry 3 -o "$ZIP" \
    "https://zenodo.org/records/10970911/files/METAFLUX_GPP_RECO_monthly.zip?download=1"
fi
echo "verifying checksum"
python3 - "$ZIP" <<'EOF'
import hashlib, json, sys, urllib.request
meta = json.load(urllib.request.urlopen("https://zenodo.org/api/records/10970911"))
want = next(f["checksum"] for f in meta["files"] if f["key"] == "METAFLUX_GPP_RECO_monthly.zip")
algo, digest = want.split(":")
h = hashlib.new(algo)
with open(sys.argv[1], "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
assert h.hexdigest() == digest, f"checksum mismatch: {h.hexdigest()} != {digest}"
print(f"{algo} ok")
EOF
unzip -o -q "$ZIP" -d "$RAW/metaflux"
echo "done; set FLUXPOS_RAW=$RAW if you used a non-default location"
