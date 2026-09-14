#!/usr/bin/env bash
# Unpack the Kaggle provenance bundle into provenance/ and check it.
#
#   bash tools/import_provenance.sh ~/Downloads/provenance.zip "https://www.kaggle.com/code/..."
#
# The second argument is the link to the SAVED notebook version (Share -> copy
# link). Gate 2 section 3.1 asks for the hosted execution link, and this records
# it in provenance/run_manifest.json.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZIP="${1:?usage: tools/import_provenance.sh <provenance.zip> [kaggle_notebook_url]}"
URL="${2:-}"
[[ -f "$ZIP" ]] || { echo "error: $ZIP not found" >&2; exit 1; }
ZIP="$(cd "$(dirname "$ZIP")" && pwd)/$(basename "$ZIP")"
cd "$ROOT"

# Python's zipfile rather than unzip, which is not installed everywhere.
python3 - "$ZIP" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
names = z.namelist()
bad = [n for n in names if not n.startswith("provenance/") or ".." in n]
if bad:
    sys.exit(f"error: this does not look like the AGBE bundle: {bad[:5]}")
z.extractall(".")
print(f"extracted {len(names)} entries into provenance/")
PY

# GitHub refuses files over 100 MB, and the push fails at the end, not now.
big="$(find provenance -type f -size +95M)"
if [[ -n "$big" ]]; then
  echo "error: too large to push to GitHub:" >&2; echo "$big" >&2; exit 1
fi

missing=0
for f in adapter/adapter_model.safetensors adapter/adapter_config.json \
         training_log.json training_log.csv run_manifest.json checksums.json environment.json; do
  if [[ -f "provenance/$f" ]]; then echo "  ok       provenance/$f"
  else echo "  MISSING  provenance/$f"; missing=1; fi
done

python3 - "$URL" <<'PY'
import json, pathlib, sys
p = pathlib.Path("provenance/run_manifest.json")
if not p.exists():
    sys.exit(0)
m = json.loads(p.read_text())
if sys.argv[1]:
    m["kaggle_url"] = sys.argv[1]
    p.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
print()
for k in ("base_model", "base_model_revision", "train_examples", "epochs",
          "optimiser_steps", "trainable_params", "last_interval_loss", "train_loss",
          "kaggle_url"):
    print(f"  {k:22} {m.get(k)}")
PY

echo
if [[ "$missing" -eq 0 ]]; then
  echo "Bundle imported. Now run: bash tools/check_submission.sh"
else
  echo "Some files are missing: check the provenance cells' output in the saved Kaggle version." >&2
  exit 1
fi
