# Building, releasing and vendoring our LiteRT-LM

The source build lives on `agello-next` (worktree `../agello-arms/next/LiteRT-LM`), with LiteRT
and ML Drift from our forks (`PowerBeef/LiteRT`, `PowerBeef/ml-drift`, branch `agello-next`).

## Build

```sh
agello/build-xcframework.sh          # dist/CLiteRTLM.xcframework (iPhone and Mac slices) + dist/BUILD-INFO.txt
```

- `WORKSPACE` pins LiteRT and ML Drift by commit and archive sha256. To change them, uncomment
  the `--override_repository` lines in `.bazelrc.user` (local checkouts), then push, re-pin (the
  commit and `shasum -a 256` of `https://github.com/PowerBeef/<repo>/archive/<commit>.tar.gz`),
  comment the overrides out and rebuild: the framework comes out byte-identical.
- Push `agello-next` from the main checkout (`../LiteRT-LM`); from the worktree, Git LFS's
  pre-push lock check fails.

## Build provenance (P1.1)

`--config=agello` runs `agello/workspace_status.sh`, which stamps the LiteRT-LM, LiteRT and ML
Drift commits (each with `+<hash>` of its uncommitted changes), the build flags, and a 12-digit
build id that changes with any of them. `litert_lm_get_build_info()` returns them as JSON
(`LiteRTLM.BuildInfo` in Swift). agello-bench records it in each machine record
(`runtimeBuild`), Agello's harness in its device record, and Agello's `EngineCache` keeps one
cache folder per build id (`kv-8192-<id>`; Google's binary, without an id, keeps `kv-8192`).

## Release (P1.3)

```sh
agello/release-xcframework.sh <tag>              # dry run: dist/release/<tag>/
agello/release-xcframework.sh <tag> --publish    # creates the GitHub release (needs the user's OK)
```

It refuses local overrides and uncommitted changes, builds, and writes the zip, its SwiftPM
checksum (`checksum.txt`), a release `Package.swift`, `BUILD-INFO.txt`, `THIRD_PARTY_NOTICES.md`
and `NOTES.md`. Agello then vendors it with
`REF=<agello-next commit> TAG=<tag> Vendor/use-release-litert.sh` (its header lists the follow-up:
`Vendor/README.md`, `NOTICE`, the README section, the iPhone suites).

Dry run on 2026-10-15 (commit `57376f53`, build id `3b702497f7d5`): a 26 MB zip; its slices hash
to the BUILD-INFO values, and the release `Package.swift` parses with the URL and checksum.

## Third-party licenses (P1.4)

`python3 agello/third_party_notices.py > agello/THIRD_PARTY_NOTICES.md` lists the external
repositories among the framework's link inputs (so build tools aren't listed), with each license
(Cargo `license` fields for Rust crates, a table for C/C++) and its text. As of 2026-10-15: 101
components, all permissive (MIT and/or Apache-2.0 for most, BSD-2/3-Clause, Zlib, Unlicense, and
Ooura's FFT license); nothing copyleft. Regenerate it when dependencies change; the release
script ships it.

## Privacy manifest (P1.5)

Required-reason APIs referenced by the iOS slice (`nm -u`): `stat`, `fstat`, `lstat` (file
timestamp) and `mach_absolute_time` (system boot time). Agello's `PrivacyInfo.xcprivacy` already
declares both categories (`C617.1`, `35F9.1`) for Google's binary, so switching to our build
needs no manifest change. Also referenced, not on Apple's list: `clock_gettime`, `sysctl`,
`sysctlbyname`. Recheck on every release.
