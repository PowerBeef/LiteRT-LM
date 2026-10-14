#!/bin/sh
# Build CLiteRTLM.xcframework (iPhone and Mac slices, Metal accelerator linked statically) and
# unpack it into dist/, where Package.swift's binary target points.
# Usage: agello/build-xcframework.sh [extra bazel flags...]
# Writes dist/CLiteRTLM.xcframework and dist/BUILD-INFO.txt (commits, flags, toolchain, SHA-256s).
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
bazelisk build --config=agello -c opt "$@" //agello:CLiteRTLM
rm -rf dist/CLiteRTLM.xcframework
mkdir -p dist
unzip -q bazel-bin/agello/CLiteRTLM.xcframework.zip -d dist
{
  echo "LiteRT-LM: $(git rev-parse HEAD)$(git diff --quiet HEAD || echo ' (modified)')"
  echo "LiteRT: $(sed -n 's/^LITERT_REF = "\(.*\)"/\1/p' WORKSPACE)"
  echo "ML Drift: $(sed -n 's/^ML_DRIFT_COMMIT = "\(.*\)"/\1/p' WORKSPACE)"
  echo "Bazel: $(bazelisk version 2>/dev/null | sed -n 's/^Build label: //p')"
  echo "Xcode: $(xcodebuild -version | tr '\n' ' ')"
  echo "Flags: --config=agello -c opt $*"
  echo "Built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for slice in dist/CLiteRTLM.xcframework/*/CLiteRTLM.framework; do
    binary="$(find "$slice" -name CLiteRTLM -type f | head -n 1)"
    echo "$(shasum -a 256 "$binary" | cut -d' ' -f1)  ${binary#dist/}"
  done
} > dist/BUILD-INFO.txt
cat dist/BUILD-INFO.txt
