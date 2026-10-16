# Returns: ours (B4) — patches.md, contributing.md

## Settings rows (for configuration.md)

Switches our patches add (all ours, agello-next only; Google's v0.18.0 binary that Agello ships has none of them). `--config=agello` is already proposed by lm-build (LM-BLD-006).

| name | layer | default | where read (citation link) | effect on Apple | version added | Agello sets it? | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `AGELLO_BOUNDED_ATTENTION` | environment (LiteRT ML Drift delegate) | on (unset or any value not starting with `0`) | [`bounded_attention.cc:293-307`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/ml_drift_delegate/delegate/composite/bounded_attention.cc#L293-L307) | `0` turns off the bounded-attention rewrite (select and softmax over the whole KV capacity again); part of the program cache key | LiteRT `ede28c64` (ours) | no (A/B only; the harness can pass `TEST_RUNNER_AGELLO_BOUNDED_ATTENTION=0`) | read once per process (function-local static) |
| `AGELLO_BOUNDED_WINDOW` | environment (LiteRT delegate) | on | [`bounded_attention.cc:293-307`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/ml_drift_delegate/delegate/composite/bounded_attention.cc#L293-L307) | `0` keeps the filled-length bound but drops the local-window start | LiteRT `31386317` (ours) | no | part of the program cache key (RT-DLG-034) |
| `AGELLO_SHARED_WEIGHT_CACHE` | environment (LiteRT delegate) | on (anything but `"0"`) | [`delegate_kernel.cc:110-125`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/ml_drift_delegate/delegate/delegate_kernel.cc#L110-L125) | `0` brings back the upstream weight-cache key (one 2.2 GB copy per context size) | LiteRT `97e06d36` (ours) | no | RT-DLG-055; the draft PR says the name must change for upstream |
| `AGELLO_VERIFY_DIRECT_COPIES` | environment (LiteRT Metal buffer handler) | off (only `1…` enables) | [`buffer_handler_metal.mm:253-258`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/ml_drift_delegate/delegate/buffer_handler_metal.mm#L253-L258) | runs ML Drift's conversion after each direct host copy and reports an error anomaly on any differing byte; slow | LiteRT `584f3461` (renamed from `AGELLO_VERIFY_DIRECT_READ` of `e17213ea`) | no (harness: `TEST_RUNNER_AGELLO_VERIFY_DIRECT_COPIES=1`) | Apple only |
| `AGELLO_SIMULATE_APPLE_GPU` | environment (ML Drift `GetGpuInfoFromDeviceDescription`; LiteRT delegate cache keys) | unset | [`gpu_info.cc:1365-1375`](https://github.com/PowerBeef/ml-drift/blob/15fa9e2ba632145992299e3f04786dc1f575373c/ml_drift/common/gpu_info.cc#L1365-L1375), [`delegate_kernel.cc:859-862`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/ml_drift_delegate/delegate/delegate_kernel.cc#L859-L862) | an Apple GPU name (e.g. `Apple A17 Pro`) makes family and core-count kernel choices follow that GPU; keys its own program and weight caches | ML Drift `ef08a67`, LiteRT `36275983` (ours) | no (harness passes it through for simulation runs) | code paths only; speed and arithmetic stay the real GPU's (MD-KRN-080) |
| `AGELLO_TELEMETRY` | environment (agello/telemetry) | level 1 (counters, anomalies) | [`agello_telemetry.cc:53`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/agello/telemetry/agello/agello_telemetry.cc#L53) | 0–4, clamped; 3 adds trace and debug groups, 4 per-kernel GPU timing (one encoder per kernel, changes timing) | LiteRT-LM `4aed9274` (ours) | no | LM-CAPI-040; also settable through `Telemetry.level` |
| `AGELLO_TELEMETRY_OUT` | environment (agello/telemetry) | unset | [`agello_telemetry.cc:53`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/agello/telemetry/agello/agello_telemetry.cc#L53) (same file, `std::atexit` at line 759) | writes the snapshot JSON at exit (and `<path>.trace.json` at level ≥ 3) | `4aed9274` (ours) | no | LM-CAPI-040 |
| `AGELLO_TELEMETRY_GRAPH_DIR` | environment (ML Drift Metal `InferenceContext`) | unset | [`inference_context.mm:638`](https://github.com/PowerBeef/ml-drift/blob/15fa9e2ba632145992299e3f04786dc1f575373c/ml_drift/metal/inference_context.mm#L638) | dumps each built or restored GPU graph as `gpu-graph-<n>-<nodes>-nodes.json`, independent of the level | ML Drift `55196d7` (ours) | no | MD-MTL-034 |
| `ExperimentalFlags.enableBenchmark` (interaction) | Swift | nil | [`llm_executor_settings_utils.cc:220-235`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_executor_settings_utils.cc#L220-L235) | true → `is_benchmark` → active (busy) GPU wait; false → passive wait (ours `645aaf65`) | v0.18.0 (flag); passive branch ours | yes, true at every load (`ChatEngine.swift:142` at `93122b6`) | so our passive waits never apply in the app (OUR-034; RT-RUN-028) |

## Terms (for glossary.md)

| term | meaning | prefer / avoid |
| --- | --- | --- |
| shipped wrapper | The Swift package surface Agello vendors: `agello` `85b1a776`, v0.18.0 plus two Swift patches, with Google's prebuilt v0.18.0 binary | prefer "shipped wrapper"; avoid "our fork" for it (the fork has more) |
| documented build, `agello-next` | Our source build: LiteRT-LM `7b872544` with LiteRT `362759838` and ML Drift `15fa9e2` from our forks | prefer "our build at `<commit>`"; avoid "our LiteRT-LM" without a commit |
| draft (upstream) | A text in `docs/agello/upstream/`, not posted | avoid "filed", "reported" for drafts |
| overlapped upstream | Google landed a change in the same place after our base (e.g. LiteRT `bd326784`); ours needs a rebase | avoid "merged upstream" unless Google's code makes ours unnecessary |
| fixed upstream | Google's code makes our patch unnecessary from a named commit on (XNNPACK via LiteRT-LM `c05e67a2`) | |
| copybara-service | Google's internal mirror; it authors almost every merged LiteRT-LM pull request | avoid "the LiteRT-LM bot" |
| CLA | Google's Contributor License Agreement (individual or corporate), needed before LiteRT or ML Drift accepts a pull request | |
| track tag | The plan item in our commit subjects (`[T4]`, `[M1.2]`, `[P6]`, `[SIM]`, `[CPU]`) | explain or drop in upstream text |
| working-tree build | A measured build whose build info shows `+<hash>` (uncommitted changes) after a commit | say "commit X plus local changes", not "commit X" |
| orphaned handler | A released session's context handler held only by the resource manager; `ed998d80` skips its snapshot | |
| raw snapshot | A KV snapshot copied byte for byte from shared `MTLBuffer`s, no layout conversion (`aec81cb7`) | avoid "zero-copy" (it copies) |
| modification notice | The header comment a modified Apache-2.0 file carries; only `swift/Conversation.swift` has one | |

## Contradictions

- **C3** — The shipped wrapper (agello `85b1a776`) carries two Swift patches (failed tool calls answered with error responses; the weak stream reference), both in `swift/Conversation.swift`; Vendor/README, NOTICE and the skill are right, and Agello `CLAUDE.md` line 24 ("one tool-call patch") is stale since Agello vendored `85b1a776` on 2026-10-14. Claims: OUR-002, OUR-003 (with LM-SWIFT-019, -023, -046, LM-VER-001).
- **C4** — The landing page (`.github/README.md` on the default branch `agello`, unchanged since `7c4ee32d`) lists one patch, no `agello-next`, calls `research/3913` a branch (it's an annotated tag, `2880d6c9` → `7cebaa0c`), and says the Metal accelerator is "a prebuilt binary even upstream", while LiteRT's base has its source and `agello-next` links it statically. The resolution lists what the page should say; the outline's "open since 2026-10-08" date is not established from GitHub (oldest ml-drift forks 2026-10-09 UTC; PublicEvent recorded at creation, 2026-05-14). Claims: OUR-025, OUR-026, OUR-027, OUR-031.
- **C7** — Agello README/TESTING (220 + 48 + 23 = 291) are older than the tests: at `93122b6` the sources declare 227 / 49 / 23 test methods, and VALIDATION's iPhone run reports unit 227 passed + 1 skipped and ui 22/22 (screenshot tour skipped as opt-in). Rule recorded in contributing.md: test counts are Agello facts, stated in Agello's README/TESTING as harness measurements with date and commit; the reference doesn't carry them. Claim: PRC-017.
- **C8** — Seven drafts: four LiteRT-LM issues, three LiteRT PRs, one ML Drift PR. PLAN-MAC's "3 issues, 3 PRs" copies the subject of `8a1df223`, the commit that added all seven drafts and already miscounted them. Claim: OUR-016.
- **C17** — AUDIT (rev `13d5aa4c`, sha256 `4a691f03…`) and BASELINE (rev `13d5aa4c`) describe the 2048-block file; the reference and Agello use `a1693e22` (sha256 `6353a233…`, 3,660,726,272 bytes, 737,280 bytes larger). The older file isn't deleted: it's replaced at the head of the same file name and still downloadable at `13d5aa4c`. Claims: OUR-028, OUR-029.
- **NEW-1 (skill on LiteRT-LM pull requests)** — Agello's `agello-litert-update` skill says outside pull requests "have been closed, not merged"; GitHub shows #2688 (author association CONTRIBUTOR) merged on 2026-07-19, among 30 of 2,497 merged PRs not authored by copybara-service. The policy text (issues only) is unchanged. Claim: PRC-003.
- **NEW-2 (direct-copies draft vs upstream)** — Google's LiteRT `bd326784` (2026-10-14) writes unlocked data in place into shared Metal buffers, the part of our `584f3461` that removed the per-unlock command queue; the draft's write rows need rework. Claim: OUR-011.
- **NEW-3 (XNNPACK drop condition reached upstream)** — RT-CPU-016's "drop when TensorFlow's pin passes `c72e2a4a`" is met on upstream LiteRT-LM main since `c05e67a2` (TensorFlow `5d071809`, XNNPACK `0a66ebb5`); the patch can go on our next rebase. Claim: OUR-012.
- **NEW-4 (ML Drift PR draft incomplete)** — the bounded-attention draft's ML Drift PR lists `6ae7c68` and `2184a7f`, not `15fa9e2` (window in ConvWaveMatrix). Claim: OUR-017.
- **NEW-5 (draft numbers without raw records)** — suite times "367 s instead of 966 s" (stream-stall draft) and weight-cache file sizes 2.2/4.4/6.6 GB (weight-cache draft, T11 note) are in no results file. Claim: PRC-008.

## Notes for other pages

- **litert/cpu-xnnpack.md (rt-runtime):** the drop condition for our XNNPACK patch is met on upstream LiteRT-LM main from `c05e67a2` (2026-10-13): TensorFlow `5d071809` pins XNNPACK `0a66ebb5`, 219 commits after `c72e2a4ae0` (OUR-012). Not a disagreement, an update after the page's pins.
- **litert/ml-drift-delegate.md (rt-delegate), "The Metal buffer handler":** upstream LiteRT `bd326784` (after our base) now memcpy's converted data into shared buffers on unlock (OUR-011); worth one line where the upstream unlock path is described.
- **versions.md / maintenance.md (Run C):** upstream main moved after our bases: LiteRT-LM main pins LiteRT `9b8c7f77` and TensorFlow `5d071809`; files we patch that changed upstream (rebase hotspots): LiteRT `buffer_handler_metal.mm`, `delegate_kernel.cc`, `delegate_kernel_litert.cc`, `litert_op_selector.cc`; ML Drift `fully_connected.cc`, `softmax.cc`, `softmax1x1.cc`, `gpu_operation.cc`, `gpu_info.cc` (OUR-013, OUR-015). Our base `df6df5b8` is not an ancestor of LiteRT-LM main (main's `a82895c4` plus two nightly checksum commits).
- **models/gemma4-e4b.md (mod-e4b):** HF revision facts for C17: `13d5aa4c` file 3,659,988,992 bytes sha256 `4a691f03…`; `a1693e22` 3,660,726,272 bytes sha256 `6353a233…`; card commit `b58c33c6` (OUR-028).
- **memory.md / apple-platform.md (C1):** the iPhone harness device record in `results/iphone/2026-10-15/agello/probes-ours.json` reads `"profile": "12 GB RAM · app limit 8.6 GB"` while VALIDATION.md says "app limit 6.4 GB" for the same run (evidence for C1; not a claim of ours).
- **troubleshooting.md (Run C):** symptom "first token after dropping a conversation mid-stream takes ~600 s" = the strong `StreamContext` reference, fixed by our Swift patch (OUR-005, OUR-006; LM-SWIFT-020).
- **maintenance.md (Run C):** after each wrapper change update `Vendor/README.md`, `NOTICE`, README "Vendored LiteRT-LM" and `CLAUDE.md` together (C3); update the fork landing page (C4); re-read upstream CONTRIBUTING.md before posting.
- **stack.md / versions.md:** none of our runtime patches reach users today (consistent with RT-DLG-083, MD-KRN-085).
- **No disagreement with Run A pages found.** LM-VER-001, LM-SWIFT-019/023/024/046, LM-BLD-063, RT-CPU-016/017 and LM-EXE-038 agree with what I checked.

## Open questions

- Whether Google's CLA check accepts commits carrying a `Co-Authored-By:` trailer that names a model, or such trailers must be removed from pull-request commits.
- When ML Drift's sources became public (2026-10-08 per our notes; GitHub only shows forks from 2026-10-09 UTC and a PublicEvent at creation).
- Whether our LiteRT and ML Drift commits rebase cleanly onto upstream main (not attempted: no merges in pinned checkouts).
- Whether Agello could run without benchmark mode (so passive waits apply), and what it would lose; not measured.
- `d1488ce7`'s Mac suite results are in its commit message; the parity records that cover it (results/mac/2026-10-14/agello/parity-ours/) carry no build id.
- Apache 2.0 §4 notices: 8 of 57 modified source files carry no Agello mention and only one has a header notice (PRC-015); whether inline comments suffice before we distribute our framework is a decision for the user.
