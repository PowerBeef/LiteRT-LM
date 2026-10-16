#!/bin/sh
# Build the LiteRTLM symbol graph in a scratch folder (outside the repos), render
# generated/swift-api-inventory.md, then delete the build (a few hundred MB).
# The agello-next worktree must be at the pin, with dist/CLiteRTLM.xcframework built from it.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(python3 -c "import json,sys; print(json.load(open('$HERE/pins.json'))['repos']['litert-lm']['path'])")"
SCRATCH="${SCRATCH:-${TMPDIR:-/tmp}/agello-docs-symgraph}"
rm -rf "$SCRATCH"
mkdir -p "$SCRATCH"
(cd "$ROOT" && swift package --scratch-path "$SCRATCH/build" dump-symbol-graph --minimum-access-level public \
  > "$SCRATCH/dump.log" 2>&1)
GRAPH="$(find "$SCRATCH/build" -name 'LiteRTLM.symbols.json' | head -1)"
if [ -z "$GRAPH" ]; then
  echo "no LiteRTLM symbol graph; see $SCRATCH/dump.log" >&2
  exit 1
fi
python3 "$HERE/gen_swift_api.py" "$GRAPH" "$SCRATCH/dump.log"
status=$?
rm -rf "$SCRATCH"
exit $status
