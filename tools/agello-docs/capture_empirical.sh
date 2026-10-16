#!/bin/sh
# Run E4B once per Apple GPU family through agello-bench with telemetry and ML Drift's graph dumps,
# then render generated/kernels-by-family.md and generated/engine-run.md. Uses our runtime through
# agello-bench's .build-next (built against the agello-next worktree's dist/), about 2 minutes per
# family, and ~2 GB of engine cache per family in $SCRATCH, deleted at the end.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH="$HERE/../agello-bench"
B="$BENCH/.build-next/out/Products/Release/agello-bench"
MODEL="$(python3 -c "import json; print(json.load(open('$HERE/pins.json'))['models']['e4b-ours']['path'])")"
OUT="${SCRATCH:-${TMPDIR:-/tmp}}/agello-docs-empirical"
[ -x "$B" ] || { echo "build agello-bench first: see tools/agello-bench/README.md (.build-next)" >&2; exit 1; }
rm -rf "$OUT"
mkdir -p "$OUT"
for fam in "native:" "a17pro:Apple A17 Pro" "m2:Apple M2" "m1:Apple M1"; do
  name=${fam%%:*}
  sim=${fam#*:}
  mkdir -p "$OUT/graphs-$name"
  if [ -n "$sim" ]; then export AGELLO_SIMULATE_APPLE_GPU="$sim"; else unset AGELLO_SIMULATE_APPLE_GPU; fi
  (cd "$BENCH" && AGELLO_TELEMETRY_GRAPH_DIR="$OUT/graphs-$name" "$B" run --model "$MODEL" \
    --case '{"id":"t2k","kind":"speed","sentences":129,"late":false,"seed":0,"decode":16}' \
    --kv 8192 --mtp --telemetry 2 --label "$name" --cache "$OUT/cache" \
    --out "$OUT/run-$name.jsonl" 2> "$OUT/log-$name.txt")
  echo "$name: exit $?, $(ls "$OUT/graphs-$name" | wc -l | tr -d ' ') graphs"
done
unset AGELLO_SIMULATE_APPLE_GPU
rm -rf "$OUT/cache"
python3 "$HERE/gen_empirical.py" "$OUT"
status=$?
rm -rf "$OUT"
exit $status
