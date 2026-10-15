#!/bin/sh
# Agello: the build stamp behind litert_lm_get_build_info (bazel --workspace_status_command, set by
# --config=agello). Prints the commits of LiteRT-LM, LiteRT and ML Drift (the local checkouts when
# .bazelrc.user overrides them, else the WORKSPACE pins), each with a hash of its uncommitted
# changes, the build flags (AGELLO_BUILD_FLAGS, set by build-xcframework.sh), and a build id that
# changes whenever any of them does.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

# A checkout's commit, plus "+<hash of its diff>" when the tree has uncommitted changes.
describe() {
  commit="$(git -C "$1" rev-parse HEAD 2>/dev/null)" || { echo unknown; return; }
  if git -C "$1" diff --quiet HEAD 2>/dev/null; then
    echo "$commit"
  else
    echo "$commit+$(git -C "$1" diff HEAD | shasum -a 256 | cut -c1-12)"
  fi
}

# An overridden repository's checkout, or its WORKSPACE pin.
source_of() {
  dir="$(sed -n "s|^build \"*--override_repository=$1=\([^\"]*\)\"*\$|\1|p" .bazelrc.user 2>/dev/null | tail -n 1)"
  if [ -n "$dir" ]; then
    describe "$dir"
  else
    sed -n "s/^$2 = \"\(.*\)\"/\1/p" WORKSPACE
  fi
}

LITERT_LM="$(describe "$ROOT")"
LITERT="$(source_of litert AGELLO_LITERT_COMMIT)"
ML_DRIFT="$(source_of ml_drift ML_DRIFT_COMMIT)"
FLAGS="${AGELLO_BUILD_FLAGS:-unknown}"
ID="$(printf '%s\n%s\n%s\n%s\n' "$LITERT_LM" "$LITERT" "$ML_DRIFT" "$FLAGS" | shasum -a 256 | cut -c1-12)"

echo "STABLE_AGELLO_LITERT_LM $LITERT_LM"
echo "STABLE_AGELLO_LITERT $LITERT"
echo "STABLE_AGELLO_ML_DRIFT $ML_DRIFT"
echo "STABLE_AGELLO_FLAGS $FLAGS"
echo "STABLE_AGELLO_BUILD_ID $ID"
