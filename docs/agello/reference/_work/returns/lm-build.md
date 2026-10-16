# Returns: lm-build (A3: build-and-release.md, versions.md, other-platforms partial)

## Settings rows (for configuration.md)

| Name | Layer | Default | Where read | Effect on Apple | Version added | Agello sets it? | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `--config=agello` | Bazel config (`.bazelrc`) | not applied | [`.bazelrc:553-566`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/.bazelrc#L553-L566) | FST off, C++20 for C++ and Objective-C++, `-Wl,-undefined,error`, build stamp | ours (`43f0f5c3`, `c7c744eb`) | yes (both scripts) | LM-BLD-006 |
| `LITERT_LM_FST_CONSTRAINTS_DISABLED` | Bazel `--define` | unset | [`build_config/BUILD:225-243`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/build_config/BUILD#L225-L243) | The closed Gemma constraint provider isn't linked; data processors compile without it | upstream | yes (`=1`, via `--config=agello`) | Same as `--//build_config:enable_fst_constraints=false` |
| `//build_config:enable_*` bool flags (sentencepiece, huggingface tokenizer, minijinja, constrained decoding, llguidance, tool-use parser, gpu, npu, legacy artisan executors, vision, audio, debugger, diffusion, google tensor, model processors) | Bazel flag / `--define` | on, except diffusion, google tensor, debugger | [`build_config/BUILD:81-83`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/build_config/BUILD#L81-L83) | NPU forced off on iOS (not macOS); Google Tensor Android-only | upstream `200545f7` (2026-10-12) | no (defaults) | Table in build-and-release.md |
| `litert_runtime_link_mode` | Bazel `--define` (LiteRT) | static | [`runtime/executor/BUILD:252-279`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/BUILD#L252-L279) | `dynamic` empties the static GPU accelerator and sampler hooks (then GPU needs `dlopen` of prebuilt libs) | upstream | no (static) | |
| `//runtime/executor:gpu_accelerator` | Bazel string flag | `default` | [`runtime/executor/BUILD:179-194`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/BUILD#L179-L194) | None at the pin: no `select()` uses its derived settings; ours links Metal anyway | upstream | no | LM-BLD-061 |
| `--override_repository=litert=<dir>`, `ml_drift=<dir>` | `.bazelrc.user` (machine-local) | commented out | [`agello/workspace_status.sh:20-32`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/agello/workspace_status.sh#L20-L32) | Builds against local checkouts; stamped with their commits | ours | only during development; the release script refuses them | Exact form `build "--override_repository=<name>=<dir>"` |
| `AGELLO_BUILD_FLAGS` | environment variable (build) | `unknown` in the stamp | [`agello/workspace_status.sh:33`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/agello/workspace_status.sh#L33) | Part of the stamp and of the build id | ours (`c7c744eb`) | set by `build-xcframework.sh` | |
| `AGELLO_LITERT_COMMIT` / `AGELLO_LITERT_SHA256`, `ML_DRIFT_COMMIT` / `ML_DRIFT_SHA256` | `WORKSPACE` | `362759838…`, `15fa9e2…` | [`WORKSPACE:493-540`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/WORKSPACE#L493-L540) | Which LiteRT and ML Drift are linked | ours (`e309971f`) | yes | Re-pin procedure in build-and-release.md |
| `minimum_os_versions` of `//agello:CLiteRTLM` | Bazel rule attribute | iOS 27.0, macOS 27.0 | [`agello/BUILD:36-39`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/agello/BUILD#L36-L39) | The framework loads only on 27.0 and later | ours (`3b2327cf`) | yes | `Package.swift` still declares iOS 15 / macOS 12 |
| `CLiteRTLM` binary target (`path:` or `url:` + `checksum:`) | `Package.swift` (SwiftPM) | `path: "dist/CLiteRTLM.xcframework"` on `agello-next`; v0.18.0 URLs on `agello` | [`Package.swift:35-45`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/Package.swift#L35-L45) | Which native binary the app links | ours (`3b2327cf`) | Agello's vendored manifest uses v0.18.0's URLs | Rewritten by `release-xcframework.sh` and `Vendor/use-release-litert.sh` |
| `--publish` | `release-xcframework.sh` argument | dry run | [`agello/release-xcframework.sh:67-74`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/agello/release-xcframework.sh#L67-L74) | Creates the GitHub release | ours (`57376f53`) | never without the user's OK | |
| `REF`, `TAG` | environment variables (Agello `Vendor/use-release-litert.sh`) | required | Agello `93122b6` `Vendor/use-release-litert.sh` lines 10-29 | Vendors a released build | Agello | when switching to our build | |
| `ENABLE_YNNPACK` / `ExperimentalFlags.enableYnnpack` | environment variable / Swift | nil (engine default) | [`swift/ExperimentalFlags.swift:205-225`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ExperimentalFlags.swift#L205-L225) | Needs `--define=litert_enable_ynnpack=true`, set only for `linux_arm64`; no effect in our Apple build | upstream `96a60092` (after v0.18.0) | no | For swift-api.md to confirm |

## Terms (for glossary.md)

| Term | Meaning | Prefer / avoid |
| --- | --- | --- |
| `--config=agello` | Our Bazel configuration block in `.bazelrc` | prefer "`--config=agello`"; avoid "agello build flags" |
| Build stamp | The JSON `litert_lm_get_build_info()` returns (build id, three commits, flags) | prefer "build info" for the API, "stamp" for the mechanism |
| Build id | 12 hex digits of SHA-256 over the three commits and the flags; `dev` outside `--config=agello` | avoid "build number" |
| `BUILD-INFO.txt` | Text file beside the framework in `dist/` and in releases; not inside the binary | |
| Pin | Commit plus archive SHA-256 in `WORKSPACE` | avoid "lock" |
| Override | A `--override_repository` line in `.bazelrc.user` | avoid "local pin" |
| Static accelerator | GPU accelerator linked into `CLiteRTLM` and registered through `LiteRtStaticLinkedAcceleratorGpuDef` | avoid "built-in delegate" |
| Nightly tag | `<VERSION>-nightly.<YYYYMMDD>` on upstream's `nightly` branch, force-pushed | prefer the commit for reproducibility |
| Release branch | `release/vX.Y.Z`, where upstream release tags live | |
| C API version | `C_API_VERSION` / `LITERT_LM_C_API_VERSION_*`, independent of the LiteRT-LM version | avoid "API level" |
| Container version | Major.minor.patch in a `.litertlm` header; the writer's is 1.9.0 at the pin | avoid "file format version 1.9" without saying writer or file |
| `prebuilt/` | Google's binaries in Git LFS (accelerators, closed samplers, constraint provider, `libLiteRt`) | |
| Google's v0.18.0 binary | What Agello ships (`CLiteRTLM.xcframework` from the v0.18.0 release) | prefer this exact phrase in user docs |

## Contradictions

- **C5** (re-pinned builds byte-identical?): BUILD.md is right, the `agello-litert-update` skill is wrong. In the documented order the rebuild stamps a different LiteRT-LM field (`+<diff hash>` or the re-pin commit) and so a different build id into `kAgelloBuildInfoJson`, which `//c:engine` links; compare replies (agello-bench `short`) instead of bytes. BUILD.md's own 15/15 check has no results file. Claims: LM-BLD-029, LM-BLD-030.
- **C6** (agello-bench README): stale on three points. Our framework has a macOS `arm64` slice with Metal linked statically since `3b2327cf`; `research/3913` is an annotated tag (fork branches: `agello`, `agello-next`, `main`); the suites list matched `85b1a776` but the `agello` tip `61c03d8e` adds `helper-turn`, `helper-turn-check`, `helper-turn-cycles`, `thinking-mask`. Its WebGPU statement about Google's Mac binary is still right. Claims: LM-BLD-063, LM-BLD-064, LM-VER-029.
- **C16** (performance figures): the Agello README table (43/38/34/27 tok/s) measures Google's v0.18.0 binary (Agello's manifest has only ever named it, and the row predates the first source-build commit); VALIDATION's calibration measures our build `2e54fbdb9ca9` (LiteRT-LM `67d00f36`), four commits before the pin, with Google's run the same night alongside. Claims: LM-VER-023, LM-VER-024, LM-VER-025, LM-VER-026.
- **NEW** (C API doc): `c/conversation.h` marks `litert_lm_conversation_wait_until_done` "Added in version 0.2.0", but v0.18.0 (C API 0.2.0) doesn't have it; it landed upstream 2026-09-25 after v0.18.0's branch point. Worth an upstream note. Claim: LM-VER-007.
- **NEW** (`.bazelrc` comment): `build:agello`'s comment says the XCFramework's iOS slice doesn't use `build:macos`, but `--enable_platform_specific_config` applies `build:macos` on any Mac host (Bazel docs). Left open (no builds this run). Claim: LM-BLD-003.

## Notes for other pages

- **c-api.md / swift-api.md (lm-api):** `wait_until_done` version note (LM-VER-007); `litert_lm_get_last_error_code` removed (LM-VER-006); the two wrappers can't swap binaries (LM-VER-009); `ExperimentalFlags.enableYnnpack` is inert in our Apple build (LM-VER-018); `BuildInfo` (LM-BLD-036).
- **litert/runtime.md (rt-runtime):** registration order handle option → static def → `dlopen` (LM-BLD-018), the static initializer (LM-BLD-017), `litert_accelerator_library` (LM-BLD-016), Android dlopen list (LM-OTH-010).
- **components.md (lm-components):** no GPU sampler is linked in our build (`default_static_gpu_samplers` empty, LM-BLD-015); the logit mask runner arrived after v0.18.0 (LM-VER-014).
- **file-format.md (lm-components):** the reader checks only the major container version (LM-VER-020); writer 1.7 → 1.9 history (LM-VER-019, LM-VER-022).
- **conversation.md:** FST provider absent in our build; Python wheels ship it (LM-OTH-006, LM-BLD-055).
- **executors.md:** NPU executor compiled out on iOS, not macOS (LM-BLD-060, LM-OTH-001).
- **patches.md (B4):** our commits grouped by purpose (versions.md "Ours on top"); XNNPACK patch is upstream `c72e2a4ae0` (LM-BLD-023).
- **apple-platform.md (B3):** framework minimum OS 27.0 (LM-BLD-031), privacy manifest categories (LM-BLD-046), the required-reason imports seen with `nm -u` (not in the ledger).
- **memory.md / troubleshooting.md:** none.
- **contributing.md:** upstream builds Apple frameworks outside public CI (LM-BLD-052); releases on `release/vX.Y.Z`, nightlies force-pushed (LM-BLD-050, LM-VER-010).

## Open questions

- BUILD.md's re-pin check (15/15 identical replies, build `2e54fbdb9ca9`) has no results file.
- Whether `build:macos` lines reach the iOS slice through `--enable_platform_specific_config`, and which `-undefined` flag the linker honors when both are present.
- What the `exported_symbols` feature restricts in rules_apple 3.22.0 (our iOS binary exports 24,468 symbols).
- Whether `sysctl`/`sysctlbyname` in the binary read boot time (a required-reason category).
- Whether the NPU executor compiled into the macOS slice pulls in anything at run time.
- Which LiteRT and ML Drift commits Google built v0.18.0's binaries from (AUDIT says LiteRT `26895c9f`; not rechecked).
- That Google's `CLiteRTLM_mac` uses the WebGPU accelerator rests on the agello-bench README and AUDIT, not on generated evidence.
- No iPhone measurement of the pin's own build (`7023fa1a61d4`).
