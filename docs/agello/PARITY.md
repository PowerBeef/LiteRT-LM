# Parity: our source build against Google's nightly (M1.6)

Our build (`agello-next`: the 0.19.0-nightly.20261013 source, the tool-call patch, LiteRT's Metal
accelerator linked statically, response formats without the closed constraint provider) against
Google's prebuilt binaries of the same nightly. Mac mini M6 (GPU family Apple11), macOS 27.0.1;
Agello runs as an iPhone app on the Mac (`PLATFORM=mac`), so both use the iOS slice and Metal.
Raw results: every scenario's metrics in `results/mac/2026-10-14/agello/parity-ours/` and
`parity-google-nightly/` (one folder per suite), calibration reports beside them.

## Agello's suites

| Suite | Google nightly | Ours |
| --- | --- | --- |
| inference | 21/21, 173 s | 21/21, 149 s |
| probes (lab-model probes 11–13 excluded) | 11/11 | 12/12 (probe 15, failing tool calls, needs our tool-call patch) |
| files | 3/3, 39.5 s | 3/3, 32.6 s |
| thinking | 1/2 | 1/2 |
| calibrate | 4/4 windows | 4/4 windows |

Shared by both builds, so not ours to fix in parity:

- `test02_ReasoningStaysOutOfCache` fails on margin (the first reasoning is too short on the M6
  for the test's floor to separate kept from dropped; see BASELINE.md).
- Probe 9 (voice turn latency) stalled 605 s in both: its reseed step abandons a stream, so the
  stream's final callback released the last reference to its conversation, on LiteRT-LM's
  callback thread; the native delete then waited for the task running that callback until the
  10-minute timeout (a thread sample shows the chain; telemetry flagged it as "session destroyed
  while its tasks ran"). Fixed in the Swift wrapper on `agello` and `agello-next`: a stream holds
  its conversation weakly, so the conversation is deleted where the caller drops it, never inside
  the callback. (Releasing the context on another queue instead fixed this case but let a
  deletion overlap the next conversation, which stalled Google's v0.18.0 the same way.)

## Calibration (Agello defaults: MTP on, a prompt filling a third of the window)

| Window | Prompt tokens | Prefill tok/s, Google / ours | Decode tok/s, Google / ours | TTFT s | Peak footprint GB |
| --- | --- | --- | --- | --- | --- |
| 4,096 | 1,285 | 2,047 / 1,971 (−3.7%) | 73.6 / 65.9 (−10.5%) | 0.63 / 0.65 | 2.50 / 2.53 |
| 8,192 | 2,302 | 2,162 / 2,188 (+1.2%) | 69.0 / 63.5 (−8.0%) | 1.06 / 1.05 | 2.79 / 2.81 |
| 16,384 | 4,398 | 1,619 / 1,610 (−0.6%) | 73.9 / 66.3 (−10.3%) | 2.72 / 2.73 | 3.77 / 3.86 |
| 32,768 | 8,607 | 1,092 / 1,064 (−2.6%) | 61.9 / 51.3 (−17.1%) | 7.88 / 8.09 | 5.61 / 5.90 |

Without MTP, decode matches: the engine benchmark (1,024 prefill / 256 decode) gives 49.4 tok/s
for ours natively and 49.8 for Google's v0.18.0 binary in Agello on the same Mac.

## After T5's first step (2026-10-14)

Our build with direct host copies on Metal (LiteRT `584f3461`, tracks/T5-unified-memory.md),
same calibration (`results/mac/2026-10-14/agello/calibrate-ours-t5.json`), against the same Google results:

| Window | Decode tok/s, Google / ours before / ours now | Now vs Google | Prefill tok/s, Google / now | TTFT s | Peak footprint GB |
| --- | --- | --- | --- | --- | --- |
| 4,096 | 73.6 / 65.9 / 76.5 | +4.0% | 2,047 / 2,040 | 0.63 / 0.63 | 2.50 / 2.62 |
| 8,192 | 69.0 / 63.5 / 73.4 | +6.3% | 2,162 / 2,236 | 1.06 / 1.03 | 2.79 / 2.83 |
| 16,384 | 73.9 / 66.3 / 75.1 | +1.6% | 1,619 / 1,625 | 2.72 / 2.71 | 3.77 / 3.82 |
| 32,768 | 61.9 / 51.3 / 63.0 | +1.7% | 1,092 / 1,101 | 7.88 / 7.82 | 5.61 / 5.92 |

The MTP gap is closed, still with the CPU sampler: the readbacks it needs no longer go through ML
Drift's per-element conversion. Single runs per window, so read the margins as within a few
percent. Peak footprint is 1–6% above Google's (unchanged by this step).

## After T4's first step (2026-10-14)

Attention select and softmax bounded to the filled length (LiteRT `aa05bd2d`,
tracks/T4-attention.md). Same calibration (`results/mac/2026-10-14/agello/calibrate-ours-t4.json`):

| Window | Decode tok/s, Google / ours | Prefill tok/s, Google / ours | TTFT s, Google / ours | Peak footprint GB |
| --- | --- | --- | --- | --- |
| 4,096 | 73.6 / 83.5 (+13%) | 2,047 / 2,447 (+20%) | 0.63 / 0.53 | 2.50 / 2.66 |
| 8,192 | 69.0 / 76.0 (+10%) | 2,162 / 2,939 (+36%) | 1.06 / 0.78 | 2.79 / 2.90 |
| 16,384 | 73.9 / 75.5 (+2%) | 1,619 / 2,678 (+65%) | 2.72 / 1.64 | 3.77 / 3.82 |
| 32,768 | 61.9 / 65.4 (+6%) | 1,092 / 2,347 (+115%) | 7.88 / 3.67 | 5.61 / 5.95 |

Single runs per window, recorded about five hours apart.

## After P2–P7 (2026-10-15)

The local window (T4 step 2), kept conversations (T3), host costs (T2), one weight copy (T11) and
the FP16 greedy path, build `c5460a024e71`; same calibration
(`results/mac/2026-10-15/agello/calibrate-final.json`), single runs:

| Window | Decode tok/s, Google / ours | Prefill tok/s, Google / ours | TTFT s, Google / ours | Peak footprint GB, Google / ours | Load s |
| --- | --- | --- | --- | --- | --- |
| 4,096 | 73.6 / 87.6 (+19%) | 2,047 / 2,467 (+21%) | 0.63 / 0.52 | 2.50 / 2.46 | 1.8 |
| 8,192 | 69.0 / 79.0 (+14%) | 2,162 / 3,030 (+40%) | 1.06 / 0.76 | 2.79 / 2.84 | 0.8 |
| 16,384 | 73.9 / 80.9 (+9%) | 1,619 / 2,843 (+76%) | 2.72 / 1.55 | 3.77 / 3.66 | 1.0 |
| 32,768 | 61.9 / 82.1 (+33%) | 1,092 / 2,581 (+136%) | 7.88 / 3.34 | 5.61 / 5.47 | 1.5 |

All four sizes loaded from one cache folder (T11), so the whole calibration took 20 s. Agello's
Mac suites on the same build: unit 227 (1 skipped), inference 21/21, probes 11/11, thinking 2/2,
files 3/3.

## Verdict

- Correctness: parity. Same suites pass, digit copies exact, replies identical to Google's on the
  short suite.
- Prefill and TTFT: at parity, then ahead since T4's first step: prefill +20% at 4K to +115% at
  32K, TTFT halved at 32K. Memory: footprint 1–6% above Google's.
- Decode with MTP: at parity since T5's first step (above). Before it, −8% to −17%: our build
  samples on the CPU (`sampler = cpu`, the GPU sampler being closed source), MTP samples four to
  five times per round, and each 262,144-logit readback went through ML Drift's per-element
  conversion. Gate G1 is met without T7; an open Metal sampler (T7) and drafting on the GPU (T12)
  remain speed work, not parity work.
