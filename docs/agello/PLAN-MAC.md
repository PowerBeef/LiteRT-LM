# Plan: the work that doesn't need the iPhone (2026-10-15)

Everything here is built and measured on the Mac (agello-bench, Agello with `PLATFORM=mac`).
Nothing ships without an iPhone run (`inference`, `probes`, `files`, `thinking`, `calibrate`,
`memory`), plus an A17/A18-class run for the Apple9 fallbacks. Pushes, a GitHub release of our
framework, model uploads and anything posted upstream each need the user's OK.

Rules for every step: one concern per commit; a verified chunk before each commit (unit tests
where the code allows, agello-bench ABAB with `compare.py` for speed, the digit-copy suites for
fidelity, Agello's Mac suites for behavior); a track note with the numbers; raw records in
`results/mac/<date>/`.

## P1. Ship readiness (make the iPhone run the last step)

| Step | Change | Done when |
| --- | --- | --- |
| P1.1 Build provenance | `litert_lm_get_build_info()` (C) and `Telemetry`-style Swift access: LiteRT-LM, LiteRT and ML Drift commits, build flags, toolchain, and a short build id, stamped by `build-xcframework.sh` through Bazel defines (`dev` outside the script) | the bench's machine record and Agello's harness report carry it; Google's binary reports none |
| P1.2 Caches keyed by build | Agello's `EngineCache` puts an engine's GPU caches under the build id when the runtime has one (our builds); Google's binary keeps today's folders, so no migration for users | switching builds on the Mac needs no hand-renamed cache folders; unit test for the path logic |
| P1.3 Release tooling | `agello/release-xcframework.sh` (fork): pinned build with overrides off, zip, `swift package compute-checksum`, release notes from BUILD-INFO, a `Package.swift` for the release; dry run unless `--publish`. Agello `Vendor/use-release-litert.sh` vendors the wrapper and points `Package.swift` at a release URL and checksum | a dry run produces a zip, checksum and package that Agello builds against from a local file URL |
| P1.4 Licenses | Third-party licenses of everything linked into `//agello:CLiteRTLM` from a Bazel dependency query, as `agello/THIRD_PARTY_NOTICES.md` shipped in the release zip; the Agello `NOTICE` text to use when the app switches | every external repository in the query has a license entry |
| P1.5 Privacy manifest | `nm -u` of our iOS slice against the required-reason API list, compared with Agello's `PrivacyInfo.xcprivacy` | findings in `docs/agello/BUILD.md`; manifest entries prepared for the switch if any category is new |

## P2. T4 b1b: local layers read only their window

35 of E4B's 42 attention layers (and 3 of the drafter's 4) keep keys in `(pos − 512, pos]`, but
their QK product, the bounded reduction and the PV product still cover every filled key.

| Step | Change | Done when |
| --- | --- | --- |
| P2.1 Know the layer | The `runtime_bmm` parser keeps the `is_global` attribute (parsed today, then dropped); the bounded-attention pass reads the window size from the local mask's subgraph (`LESS(pos, iota + W)`), or skips | unit test: local tails get a window, global ones don't |
| P2.2 Window start in kernels | ML Drift: a start bound next to the end bound: FC destination start (QK, decode/verify), FC source start (PV), softmax and softmax-1x1 reduction start; start = `max(0, params[0] − W + 1)` rounded down to the kernel's alignment (keys below the window are masked, so rounding reads zero-weight logits) | ML Drift kernel tests for the new bounds; digit copies exact |
| P2.3 Prefill | The same source start in the prefill convolution kernels (`conv_apple_mpp`, `conv_wave_matrix`, `conv_generic`, `conv_wave_memory`) | prefill at long prompts faster, fidelity exact |
| P2.4 Measure | Decode and prefill at filled lengths 2K–30K, window 32K, MTP on and off | decode at 30K filled ≥ 1.35× (plan target), ±2% at short contexts, replies identical |

## P3. T3: the first reply after a helper

Titles, memories, dictation and web planning wipe Agello's live conversation, so the next message
re-reads the whole chat; a KV snapshot copies every position of every cache to the host (62 ms at
8K, more memory than the filled part needs).

| Step | Change | Done when |
| --- | --- | --- |
| P3.1 Measure (spike S8) | Agello probe: time to first token after a helper at 2K, 4K and 8K of history, today | numbers in the T3 note |
| P3.2 Keep the conversation | Agello's `LiteRTChatEngine` keeps the main conversation and runs helpers in a second one (LiteRT-LM switches sessions with a KV snapshot), behind a memory budget from `ContextBudget` (fallback: today's wipe and reseed); `agello-inference-rules` updated | `inference`, `probes`, `thinking` green; the probe shows no re-read; unit tests for the budget decision |
| P3.3 Cheaper snapshots | LiteRT-LM: a session snapshot copies only the filled positions, GPU to GPU (`LiteRtCopyKvCacheMetal`-style blit) instead of locking every cache to the host | `kv_clone` at 8K from 62 ms to a few ms; snapshot memory proportional to the filled length; same greedy continuation |

## P4. T2: host costs per token

| Step | Change | Done when |
| --- | --- | --- |
| P4.1 Thinking budget | Skip the logit mask while every token is allowed (`thinking_budget_constraint.cc`, the mask runner) | Think mode within 3% of plain decode; same output |
| P4.2 Per-step allocations and copies | Reuse the per-step tensor buffer; update the token-history mask input incrementally instead of copying the history each step | the `decode_allocations` and `token_history_copies` counters drop to zero per step; decode +1–2% |

## P5. T11: one weight copy for every context size

LiteRT-LM's GPU weight cache stores a 2.2 GB copy per context size; Agello keeps one cache folder per
size (the two most recent) to bound it.

| Step | Change | Done when |
| --- | --- | --- |
| P5.1 Investigate | Why the weight cache's key depends on the KV size (the delegate's `unique_model_identifier` over the subgraph); whether weights can be keyed by content | a written decision in the T11 note |
| P5.2 Shared weights | Weight blobs keyed by content, shared across context sizes; the program cache stays per size | 8K → 4K → 16K → 8K leaves one weight copy; context switch time measured |

## P6. T7: an open Metal sampler

Implement LiteRT-LM's `LiteRtTopKMetalSampler_*_Static` interface with ML Drift kernels (top-k,
softmax, Philox), wired through `default_static_gpu_samplers`; greedy and seeded sampling must
match the CPU sampler. Done when greedy output is identical, sampling passes a χ² test, and decode
is at least the CPU sampler's (+2% target), MTP included.

## P7. T6: dispatch overhead

Measure first: GPU idle between kernels in a decode step (Metal System Trace with the kernel names
from T0), and the encode time per step. Act (fusing the elementwise chains around the norms, or
pre-encoding) only if the idle time is worth it.

## P8. Upstream drafts

`docs/agello/upstream/`: the stream deadlock (Swift wrapper), the tool-call patch, bounded
attention softmax (delegate), direct host copies (buffer handler). Drafts only.

## Order

P1 → P2 → P3 → P4 → P5 → P6 → P7, with P8 alongside. Each step lands on its own, with numbers.
