#!/bin/sh
# Regenerate docs/agello/reference/generated/ from the pins in pins.json, then check the reference.
#   regen.sh            the generators that only read sources and files (about 2 minutes)
#   regen.sh --all      also the Swift symbol graph (a scratch build) and the per-GPU-family
#                       engine runs (agello-bench, ~10 minutes, ~2 GB of scratch per family)
# Tools venv: ~/.venvs/agello-tools (model tables). Downloads: E2B (2.6 GB) into $SCRATCH for the
# model tables, deleted right after; check free disk space first.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${AGELLO_TOOLS_PYTHON:-$HOME/.venvs/agello-tools/bin/python}"
cd "$HERE"
python3 gen_c_api.py
python3 gen_protos.py
python3 gen_flags.py
python3 gen_patch_log.py
"$PY" -I gen_model_tables.py
python3 gen_versions.py
python3 gen_binaries.py
python3 gen_hf_configs.py
if [ "${1:-}" = "--all" ]; then
  ./gen_swift_api.sh
  ./capture_empirical.sh
fi
python3 check_refs.py
