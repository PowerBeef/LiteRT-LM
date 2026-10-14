# Telemetry: probes into our LiteRT-LM build

Goal: see what every turn, step and kernel does in our own build (LiteRT-LM + LiteRT + ML Drift
on Metal), catch errors and silent degradations as they happen, and point at optimization
opportunities with numbers. Cheap enough to leave on in development builds; detailed modes are
opt-in. Research behind it: four inventories of the three code bases and of Apple's APIs
(findings below, with the probe each one calls for).

## What exists today, and why it isn't enough

| Facility | Where | Problem |
| --- | --- | --- |
| `BenchmarkInfo` | LiteRT-LM `io_types.cc`, `tasks.cc` | Benchmark mode changes behavior (busy-wait on GPU, forced prefill waits, stats reset); per-step marks go to a copy that's discarded (`tasks.cc:222, 536`); unsynchronized reads while the execution thread writes |
| `ExecutorStats` | LiteRT-LM executor | Always on, MTP acceptance and timings, but only reachable through benchmark mode |
| LiteRT profiler | `litert/runtime/profiler.*` | On Metal, an attached profiler replays the whole GPU graph on every inference (`delegate_kernel.cc:123`) |
| `LITERT_PERFETTO_TRACE_EVENT` | ~30 call sites in LiteRT | Defined empty (`perfetto_profiling.h:18-21`) |
| ML Drift `InferenceContext::Profile` | `inference_context.mm:390-546` | Per-kernel timestamps, but an isolated replay; offline |
| RuntimeDebugger | LiteRT-LM, compile-time | Dumps every tensor to disk; far too heavy for timing |
| Kernel names | ML Drift | Every kernel is `ComputeFunction`; no pipeline, encoder or command-buffer labels, so Instruments can't attribute GPU time |
| Logs | absl to stderr, LiteRT logger, os_log in Swift | Three sinks; nothing structured |

## Design

One small library, `@agello_telemetry` (`agello/telemetry/` in the LiteRT-LM fork, its own Bazel
repository so LiteRT and ML Drift can depend on it), with a C ABI and header-only C++ helpers.
Every probe is an inline check of one atomic level before any work.

| Level | Name | Records | Cost when on |
| --- | --- | --- | --- |
| 0 | off | nothing; probes are a load and a branch | ~1 ns |
| 1 | counters (default in our builds) | fixed set of counters and anomaly events | one relaxed atomic add |
| 2 | spans | per-kind aggregates (count, total, max, log2 histogram), DynamicTracing signposts | ~40 ns per span |
| 3 | trace | spans also go to per-thread ring buffers, written as a Chrome/Perfetto trace | +~20 ns |
| 4 | deep | per-kernel GPU timing (one encoder per kernel), numerical probes, graph dumps | changes timing; for diagnosis only |

- **Counters** (fixed enum, no strings): prompt and generated tokens; runs per signature (prefill
  128/1024, decode, verify, drafter); host buffer locks, copies and bytes; command buffers
  committed; GPU waits and wait time; mask fills and bytes; embedding lookups (CPU model runs);
  sampler path (GPU/CPU) and fallbacks; logit-mask path; KV copies and context switches; program
  and weight cache hits, misses and rebuilds; allocations on the decode path; run-option
  parses; stop reasons (EOS, max tokens, KV full, cancel); errors by status code.
- **Spans** (fixed enum of kinds, plus interned names for LiteRT's existing trace sites): turn,
  template render, tokenize, prefill, prefill chunk, embedding lookup, mask fill, parameter fill,
  bind, run, GPU wait, decode step, sample, constraint, detokenize, stream callback, MTP draft and
  verify, KV clone and restore, task queue wait, engine and session init phases, delegate invoke.
  Signposts go to an `os_log` handle with the DynamicTracing category (about 11 ns per pair until
  Instruments records; PointsOfInterest costs about 410 ns per signpost even when nothing
  records, so only turn-level spans use it).
- **GPU time**: every command buffer gets a completion handler that adds `GPUEndTime -
  GPUStartTime` to the current run's signature, plus `kernelStartTime` scheduling latency. Level
  4 adds per-kernel timing: one encoder per kernel with stage-boundary timestamps (the only
  sampling point Apple GPUs support on the classic API), or Metal 4 counter heaps with precise
  timestamps where the queue is Metal 4.
- **Anomalies** (always on from level 1): a bounded ring of events with severity, code, two
  integers and a short static message, mirrored to `os_log` at error/fault level. Watchdogs:
  waits past 1 s and 5 s with the waiting session and task states, sessions destroyed with
  pending tasks, streams dropped without cancel, invalid sampled tokens, sampler or mask-runner
  CPU fallbacks, GPU family not recognized (ML Drift falls back to Apple1), weight-cache load
  failures and weights prepared on every load, stale program-cache entries, GPU models split
  across partitions or left on the CPU, KV context switches, constraint creation skipped.
- **Memory** on every snapshot: `phys_footprint`, its peak, `ledger_tag_graphics_footprint`,
  `MTLDevice.currentAllocatedSize`, `os_proc_available_memory` (iOS), thermal state, low power.
- **Numerical probes** (logits on the CPU sampler path, done): NaN and Inf count (an error
  anomaly), flat rows, max logit, top-1 margin, near ties (top two within 1/16: four FP16 steps
  for logits in [16, 32), where FP16 rounding can flip the argmax), and from level 2 the entropy
  and top-1 probability at temperature 1. Model and MTP drafter logits have separate gauges.
  Levels 1–3 check the first row of every 32nd sampler call (0.38 ms for 262,144 logits, about
  0.1% of decode); level 4 checks every row. Next: stats kernels on the GPU (ML Drift optional
  nodes on rotary angles and attention scores) for FP16 saturation and precision loss.
- **Gauges**: value distributions (count, mean, min, max, and the 10th, 50th and 90th
  percentiles of a uniform sample of up to 4,096 values), in the snapshot's `gauges`.
- **Kernel identity** (always, no per-dispatch cost): ML Drift sets pipeline labels from node
  names; debug groups per kernel at level 3+.

### Interfaces

- C (LiteRT-LM, `c/telemetry.h`): `litert_lm_telemetry_set_level`, `_get_level`,
  `_snapshot_json` (counters, span aggregates, GPU time, anomalies, memory),
  `_reset`, `_write_trace(path)`; the environment variable `AGELLO_TELEMETRY=<level>` sets the
  level at startup.
- Swift: `LiteRTLM.Telemetry.level`, `.snapshot()`, `.reset()`, `.writeTrace(to:)`.
- agello-bench: each record carries the telemetry delta of its case; `telemetry.py` turns a run
  into a report (per-step host and GPU breakdown, achieved bandwidth against the weights read per
  token, top kernels, anomalies, regressions against a baseline run).
- Agello's harness: `DeviceHarness.record` adds the delta for each scenario, so every iPhone or
  Mac run carries the engine's own account of what happened.

## Using it

| What | How |
| --- | --- |
| Level | `AGELLO_TELEMETRY=<0-4>` (environment, at load), `litert_lm_telemetry_set_level`, `LiteRTLM.Telemetry.level`, agello-bench `--telemetry N` |
| Snapshot | `litert_lm_telemetry_snapshot_json`, `Telemetry.snapshot()`, or `AGELLO_TELEMETRY_OUT=<path>` for any binary at exit |
| Trace | level 3, then `litert_lm_telemetry_write_trace` / agello-bench `--trace FILE`; open in Perfetto or `chrome://tracing` |
| Instruments | level 2+: subsystem `com.patricedery.agello.litertlm`, category DynamicTracing (os_signpost intervals) |
| Per case | agello-bench resets before each case and stores the snapshot in the record's `telemetry`; the engine load's goes to `<out>.load-telemetry.json` |
| Report | `python3 tools/agello-bench/telemetry.py results/run.jsonl` (also a load snapshot, `results/run.jsonl.load-telemetry.json`, or an Agello `harness-report.json`) |
| GPU graphs | `AGELLO_TELEMETRY_GRAPH_DIR=<dir>` (any level): ML Drift writes each initialized graph there, built or restored from the program cache, as `gpu-graph-<n>-<kernels>-nodes.json` (kernel names, work groups, FLOPs, input and output shapes); `telemetry.py --graph <dir>` summarizes them |

Build notes: LiteRT-LM's probes come with `@agello_telemetry` in the tree. LiteRT's trace sites and
the Metal command-buffer timing live in our LiteRT and ML Drift forks (branch `agello-next`); until
those are pushed and pinned in the WORKSPACE, build them from the local checkouts with
`--override_repository=litert=…` and `--override_repository=ml_drift=…` in `.bazelrc.user`.

Overhead on the M6 (speed sweep, ABAB, same build): levels 1 and 2 within ±0.3% of level 0 for
prefill and decode, replies identical.

## First readings (M6, 2K prompt, 128 generated, kv 8192)

| | Without MTP | With MTP (4 tokens per round on this text) |
| --- | --- | --- |
| Step | 21.7 ms per token, of which the model run 21.0 ms (it doesn't return early: async falls back to sync on Metal) | 7.5 ms per token |
| GPU | 20.7 ms per decode run in ~9 command buffers; 95% busy over the turn; 106 GB/s effective weight bandwidth | verify 22.6 ms per run; drafter 0.66 ms per run; 83% busy; 98 GB/s |
| Host per token | sampling 0.50 ms (CPU sampler: the GPU sampler is closed source), parameter fill 0.24 ms, embedding lookups 0.14 ms | drafter steps' host work between GPU runs (CPU sampling, readbacks, lookups) leaves the GPU idle 17% |
| Prefill | 2,049 tokens in 3 chunks, 932 ms, 98% GPU | same |
| Other | 2 CPU model runs per prompt token (embedder, per-layer embedder): 4,483 inferences per case; one tensor buffer allocated per decode step; token history copied for every mask fill | |
| Load (warm cache) | delegate graph conversion ~950 ms of a 1.1 s load | |

### GPU kernels (level 4, 2K prompt, 64 generated)

Shares of summed kernel time (each kernel in its own encoder, so absolute times run high):

| | T = 8,192 | T = 32,768 |
| --- | --- | --- |
| Decode | int4 matmuls 68%, int8 matmuls 11%, LM head (int2) 6%, RMS norms 4%, softmax 3.6% | int4 matmuls 61%, **softmax 12.7%** (3.1 ms per token, 0.8 ms at 8K) |
| Prefill | Neural Accelerator int8 convolutions 39%, mask select 11.5%, softmax 16.4%, int4→int8 weight conversion 8.4%, activation quantization 3.7% | **mask select 23%, softmax 37%**, convolutions 20.5% (same work as at 8K), weight conversion 4.4% |

The matmul work doesn't depend on T; the select and the softmax scale with it. Bounding them to
the filled length (T4, first step) should bring 32K prefill close to its 8K speed (about 2×) and
add about 8% to decode at 32K. The int4 weights are converted to int8 on every prefill run (T9:
native int4 for the Neural Accelerator path).

### Load, delegation and host copies (2K prompt, 64 generated, MTP)

| | Reading |
| --- | --- |
| Delegation | 6 compiled models, 20,392 nodes: 20,335 on the GPU in 5 partitions (one per GPU subgraph: no CPU round trips inside a model), 3 in XNNPack, 54 on TFLite's own CPU kernels (the CPU embedders) |
| Caches | Cold load: 5 program-cache misses, weight cache missed, built, flushed and reloaded (5 misses, 5 hits), 8.3 s. Warm: 5 program-cache hits, 1.1 s, of which delegate graph conversion 0.5 s |
| GPU graphs | Decode 1,766 kernels (15.9 GFLOP), verify 1,817 (63.5), prefill-1024 1,678 (6,318), prefill-128 1,632 (790), drafter 164. In decode, 302 RMS-norm kernels, 321 multiplies, 247 reshapes, 132 strided slices and 85 selects; 1,010 kernels fuse several ops, many of them a cast → quantize-dequantize → cast chain, sometimes twice in a row |
| Host copies | 113 GPU → host reads and 274 host → GPU writes per case (4.9 MB per generated token, prefill included). Writes are cheap (126 MB of per-layer embeddings per prefill at about 40 GB/s). Reads are not: the logits buffers aren't packed, so each read converts ML Drift's layout on the CPU: 1.7 ms per 2 MB verify readback, 0.47 ms per 512 KB drafter readback, about 50 ms of a 494 ms decode turn (10%) |
| Async | The executor asks for asynchronous runs; the Metal backend has no post-dispatch event, so every run completes synchronously (fact `gpu_async`) |

A packed logits output (a plain shared `MTLBuffer` read by `memcpy`, or no copy at all) would
recover most of the 10%; it belongs to T5, and T7 (sampling on the GPU) removes the readback.
The decode graph's 1,766 dispatches per token point at T6 (encoding) and at fusing the
elementwise chains around the norms.

### Logits (level 4, 2K prompt, 128 generated, MTP)

| | Model (129 positions) | Drafter (96 drafts) |
| --- | --- | --- |
| NaN / Inf | 0 | 0 |
| Max logit | mean 23.8, max 29.3 (under Gemma's soft cap of 30) | mean 17.5 |
| Top-1 margin | min 1.03, p10 8.9, p50 12.6: no near ties | min 0.13, p50 12.3 |
| Top-1 probability | p10 1.000, min 0.27 | p10 0.998, min 0.52 |
| Entropy, nats | p90 0.004, max 4.0 | p90 0.019, max 3.2 |

On this repetitive text the model is nearly certain at almost every position, which matches the
100% MTP acceptance. The near-tie count is the one to watch on long contexts at FP16 (#3913):
positions whose top two logits differ by less than FP16's resolution.

### Metal waits

The Metal delegate busy-spins a CPU core for every GPU wait by default; benchmark mode isn't the
cause. Asking for passive waits, with command buffers still committed as they're encoded (LiteRT
fork 77ed9cab), measured −0.3% decode and +0.4% prefill against busy-waiting (within noise), and
2.0 ms of process CPU per generated token instead of 29.3.

These point at T2 (parameter fill, per-step allocation, history copies), T5 (async and buffer
locks), T7 (an open GPU sampler, which also closes most of the MTP gap to Google's build), T11
(load), and T12 (MTP drafting on the GPU).

## Plan

1. Done (2026-10-14): core library, C and Swift APIs, LiteRT-LM probes (turns, executor, MTP,
   sampler, execution manager watchdogs), LiteRT's trace macros mapped to spans, command-buffer
   GPU time, agello-bench integration and `telemetry.py`; overhead within noise.
2. Done (2026-10-14): pipeline labels, GPU family probe and per-kernel timing; the GPU graph
   dump (built or restored); LiteRT buffer lock and copy counters with per-size spans, program
   and weight cache events, delegation coverage per compiled model, the async fallback fact.
3. Done (2026-10-14): numerical probes on logits (CPU sampler path) and gauges; Agello's
   harness carries them in each scenario's snapshot. Next: GPU-side stats kernels, invariant
   checks (KV positions, sampled ids against constraints).
