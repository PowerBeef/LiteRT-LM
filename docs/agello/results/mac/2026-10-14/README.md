# Mac results, 2026-10-14

Raw records behind TELEMETRY.md, PARITY.md (its section "After T5's first step") and
tracks/T5-unified-memory.md. Mac mini M6 (Mac18,5, GPU family Apple11, 16 GB), macOS 27.0.1. The
model is `gemma-4-E4B-it-fp16-rope.litertlm` (the file Agello ships). Home-directory paths are
replaced by `~`.

## `bench/`: agello-bench records (JSONL), each with the engine's telemetry

Most cases are `{"id":"t2k","kind":"speed","sentences":129,"decode":128}`: a 2,049-token prompt
at a KV capacity of 8,192, run natively on macOS with the Mac slice. `<file>.load-telemetry.json`
is the engine load's snapshot. Read them with `tools/agello-bench/telemetry.py` and
`compare.py`.

| File | What | Cited in |
| --- | --- | --- |
| `tele-t2k.jsonl` | First telemetry readings, MTP off and on, level 2 | TELEMETRY.md, First readings |
| `overhead.jsonl`, `l23.jsonl` | Telemetry overhead: levels 0, 1 and 2, then 2 and 3, ABAB | TELEMETRY.md (levels 1–3 within ±0.3%) |
| `busywait.jsonl`, `passive.jsonl`, `chunked.jsonl` | Metal wait modes, busy-waiting (`active`) against passive, in that order; `chunked` is the build with LiteRT `77ed9cab` (commit while encoding) | TELEMETRY.md, Metal waits |
| `kernels.jsonl` | Level 4, per-kernel GPU timing at KV 8,192 and 32,768 | TELEMETRY.md, GPU kernels |
| `item2.jsonl`, `item2b.jsonl` | Host copies, caches and delegation, MTP on: a cold program cache, then a warm one (2 runs); 65 generated | TELEMETRY.md, Load, delegation and host copies |
| `item3-mtp.jsonl`, `item3-nomtp.jsonl` | Logits probes at level 4 | TELEMETRY.md, Logits |
| `direct-ab-mtp.jsonl`, `direct-ab-nomtp.jsonl` | T5 direct reads (LiteRT `e17213ea`, `next`) against the pinned build (LiteRT `5083cfae`, `base`), ABAB, 1 warm-up and 3 repetitions per arm | T5 note, "Direct reads" |
| `direct-ab2-mtp.jsonl`, `direct-ab2-nomtp.jsonl` | Same, with direct writes and tiled reads (LiteRT `584f3461`) | T5 note, "Reads and writes" |
| `direct2-verify.jsonl` | `AGELLO_VERIFY_DIRECT_COPIES=1` with direct writes, before the tiled reads: every direct copy checked against ML Drift's conversion (the final build's check is `agello/inference-t5-verify.json`) | T5 note, Verification |

| `t4-base.jsonl` | Before T4 (LiteRT `584f3461`): 2K prompt at windows 8K and 32K, MTP off and on | T4 note |
| `t4-ab-kv{4096,8192,32768}{,-mtp}.jsonl` | T4's pass off and on in the same binary (LiteRT `aa05bd2d`, `AGELLO_BOUNDED_ATTENTION=0` for `off`), ABAB, 1 warm-up and 3 repetitions per arm | T4 note |
| `t4-fid.jsonl` | The digit-copy suites (`short`, `long`, `cross4096`, `xl`) at a 32K window, off then on | T4 note, Fidelity |

Frameworks (SHA-256 of the macOS slice): `base` `a0cdb032…`, T5 reads `055bd719…`, T5 final
`c7793a8e…`, T4 `434de205…`.

## `agello/`: Agello's `harness-report.json` (Mac, `PLATFORM=mac`)

The iPhone app running on the Mac with the iOS slice, so Metal through the same path as the
phone. Each scenario of our builds carries `telemetry`.

| File | Build | Suite | Cited in |
| --- | --- | --- | --- |
| `calibrate-google-nightly.json` | Google's 0.19.0 nightly (20261013) | `calibrate` | PARITY.md |
| `calibrate-ours-before.json` | Ours, as first measured for parity | `calibrate` | PARITY.md |
| `calibrate-ours-t5.json` | Ours, LiteRT `584f3461` (iOS slice `89f43bc1…`) | `calibrate` | PARITY.md, After T5's first step |
| `inference-before-t5.json` | Ours, LiteRT `5083cfae` (iOS slice `23c4b7e7…`) | `inference`, 21/21, 139 s | T5 note: session snapshots before |
| `inference-t5.json` | Ours, LiteRT `584f3461` | `inference`, 21/21, 119 s | T5 note: session snapshots after |
| `inference-t5-verify.json` | Same, with `TEST_RUNNER_AGELLO_VERIFY_DIRECT_COPIES=1` | `inference`, 21/21 | T5 note: 10,480 copies, none different |
| `calibrate-ours-t4.json`, `calibrate-ours-t4-off.json` | Ours, LiteRT `aa05bd2d`, T4's pass on, and off (`TEST_RUNNER_AGELLO_BOUNDED_ATTENTION=0`); the `on` run after the `off` one | `calibrate` | PARITY.md, T4 note |
| `calibrate-ours-t4-cold.json` | Same build, the pass on, the first runs after the cache key changed: one-off stalls of about 430 ms in the CPU embedders (cold caches) | `calibrate` | T4 note |
| `inference-t4.json`, `probes-t4.json`, `thinking-t4.json`, `files-t4.json` | Ours, LiteRT `aa05bd2d` | 21/21, 12/12 (3 skipped), 2/2, 3/3 | T4 note |

Calibration runs are single runs per window. Google's and our first calibration were recorded
about two hours before the T5 run.
