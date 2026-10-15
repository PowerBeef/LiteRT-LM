#!/bin/sh
# Build CLiteRTLM.xcframework (iPhone and Mac slices, Metal accelerator linked statically) and
# unpack it into dist/, where Package.swift's binary target points.
# Usage: agello/build-xcframework.sh [extra bazel flags...]
# Writes dist/CLiteRTLM.xcframework and dist/BUILD-INFO.txt (commits, flags, toolchain, SHA-256s).
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export AGELLO_BUILD_FLAGS="--config=agello -c opt $*"
bazelisk build --config=agello -c opt "$@" //agello:CLiteRTLM
rm -rf dist/CLiteRTLM.xcframework
mkdir -p dist
unzip -q bazel-bin/agello/CLiteRTLM.xcframework.zip -d dist
# A repository overridden in .bazelrc.user (a local checkout) reports that checkout's commit.
source_of() {
  dir="$(sed -n "s|^build \"*--override_repository=$1=\([^\"]*\)\"*\$|\1|p" .bazelrc.user 2>/dev/null | tail -n 1)"
  if [ -n "$dir" ]; then
    echo "$(git -C "$dir" rev-parse HEAD)$(git -C "$dir" diff --quiet HEAD || echo ' (modified)') (local: $dir)"
  else
    sed -n "s/^$2 = \"\(.*\)\"/\1/p" WORKSPACE
  fi
}
{
  echo "LiteRT-LM: $(git rev-parse HEAD)$(git diff --quiet HEAD || echo ' (modified)')"
  echo "LiteRT: $(source_of litert AGELLO_LITERT_COMMIT)"
  echo "ML Drift: $(source_of ml_drift ML_DRIFT_COMMIT)"
  echo "Bazel: $(bazelisk version 2>/dev/null | sed -n 's/^Build label: //p')"
  echo "Xcode: $(xcodebuild -version | tr '\n' ' ')"
  echo "Flags: --config=agello -c opt $*"
  echo "Build id: $(sed -n 's/^STABLE_AGELLO_BUILD_ID //p' bazel-out/stable-status.txt)"
  echo "Built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for slice in dist/CLiteRTLM.xcframework/*/CLiteRTLM.framework; do
    binary="$(find "$slice" -name CLiteRTLM -type f | head -n 1)"
    echo "$(shasum -a 256 "$binary" | cut -d' ' -f1)  ${binary#dist/}"
  done
} > dist/BUILD-INFO.txt
cat dist/BUILD-INFO.txt
