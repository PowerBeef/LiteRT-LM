#!/bin/sh
# Package our CLiteRTLM.xcframework for a GitHub release of PowerBeef/agello-litert-lm.
#
# Usage: agello/release-xcframework.sh <tag> [--publish]
#   e.g. agello/release-xcframework.sh agello-next-20261015
#
# Builds from the WORKSPACE pins (refuses .bazelrc.user overrides and uncommitted changes), then
# writes dist/release/<tag>/:
#   CLiteRTLM.xcframework.zip   the binary target
#   checksum.txt                its SwiftPM checksum (swift package compute-checksum)
#   Package.swift               this branch's package, pointed at the release URL and checksum
#   BUILD-INFO.txt              commits, flags, build id, SHA-256 of each slice
#   THIRD_PARTY_NOTICES.md      licenses of everything linked in (agello/THIRD_PARTY_NOTICES.md)
#   NOTES.md                    the release notes
# Without --publish it stops there (a dry run). With --publish it creates the release with gh,
# tagged at this commit. Publishing needs the user's OK.
# Agello then uses it with: REF=<this commit> TAG=<tag> Vendor/use-release-litert.sh
set -eu
TAG="${1:?usage: agello/release-xcframework.sh <tag> [--publish]}"
PUBLISH="${2:-}"
REPO="PowerBeef/agello-litert-lm"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if grep -q '^build "*--override_repository' .bazelrc.user 2>/dev/null; then
  echo "A release builds from the WORKSPACE pins: comment out the overrides in .bazelrc.user." >&2
  exit 2
fi
if ! git diff --quiet HEAD; then
  echo "Commit first: a release is built from a commit." >&2
  exit 2
fi
COMMIT="$(git rev-parse HEAD)"

agello/build-xcframework.sh

OUT="dist/release/$TAG"
rm -rf "$OUT"
mkdir -p "$OUT"
(cd dist && zip -qry "release/$TAG/CLiteRTLM.xcframework.zip" CLiteRTLM.xcframework)
CHECKSUM="$(swift package compute-checksum "$OUT/CLiteRTLM.xcframework.zip")"
echo "$CHECKSUM" > "$OUT/checksum.txt"
URL="https://github.com/$REPO/releases/download/$TAG/CLiteRTLM.xcframework.zip"
URL="$URL" CHECKSUM="$CHECKSUM" perl -pe \
  's|^      path: "dist/CLiteRTLM.xcframework"$|      url: "$ENV{URL}",\n      checksum: "$ENV{CHECKSUM}"|' \
  Package.swift > "$OUT/Package.swift"
if ! grep -q "checksum: \"$CHECKSUM\"" "$OUT/Package.swift"; then
  echo "Package.swift's binary target wasn't rewritten: check its form." >&2
  exit 1
fi
cp dist/BUILD-INFO.txt "$OUT/"
cp agello/THIRD_PARTY_NOTICES.md "$OUT/"

{
  echo "CLiteRTLM.xcframework built from $REPO@$COMMIT (branch agello-next): LiteRT-LM with"
  echo "LiteRT's Metal accelerator linked statically, iPhone and Mac slices."
  echo
  echo "SwiftPM checksum: \`$CHECKSUM\`"
  echo
  echo '```'
  cat dist/BUILD-INFO.txt
  echo '```'
  echo
  echo "Third-party licenses: THIRD_PARTY_NOTICES.md."
} > "$OUT/NOTES.md"

echo "Release files in $OUT (checksum $CHECKSUM)."
if [ "$PUBLISH" = "--publish" ]; then
  gh release create "$TAG" --repo "$REPO" --target "$COMMIT" --title "$TAG" \
    --notes-file "$OUT/NOTES.md" "$OUT/CLiteRTLM.xcframework.zip" "$OUT/checksum.txt" \
    "$OUT/THIRD_PARTY_NOTICES.md" "$OUT/BUILD-INFO.txt"
else
  echo "Dry run: add --publish to create the release (tag $TAG at $COMMIT)."
fi
