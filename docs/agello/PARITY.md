# Parity: our source build against Google's nightly (M1.6)

Our build (`agello-next`: the 0.19.0-nightly.20261013 source, the tool-call patch, LiteRT's Metal
accelerator linked statically, response formats without the closed constraint provider) against
Google's prebuilt binaries of the same nightly. Mac mini M6 (GPU family Apple11), macOS 27.0.1;
Agello runs as an iPhone app on the Mac (`PLATFORM=mac`), so both use the iOS slice and Metal.
Raw results: Agello's `build/mac-harness-next/` (ours) and `build/mac-harness-nightly/` (Google's)
on the development Mac; the calibration reports are copied to `results/mac/2026-10-14/agello/`.

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
- Probe 9 (voice turn latency) stalls 605 s in both: its reseed step abandons a stream and opens
  a new conversation on the same engine; LiteRT-LM waits on the old session's work up to its
  10-minute timeout. Telemetry now flags such waits; the cause is under investigation.

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
same calibration (`build/mac-harness-t5/`), against the same Google results:

| Window | Decode tok/s, Google / ours before / ours now | Now vs Google | Prefill tok/s, Google / now | TTFT s | Peak footprint GB |
| --- | --- | --- | --- | --- | --- |
| 4,096 | 73.6 / 65.9 / 76.5 | +4.0% | 2,047 / 2,040 | 0.63 / 0.63 | 2.50 / 2.62 |
| 8,192 | 69.0 / 63.5 / 73.4 | +6.3% | 2,162 / 2,236 | 1.06 / 1.03 | 2.79 / 2.83 |
| 16,384 | 73.9 / 66.3 / 75.1 | +1.6% | 1,619 / 1,625 | 2.72 / 2.71 | 3.77 / 3.82 |
| 32,768 | 61.9 / 51.3 / 63.0 | +1.7% | 1,092 / 1,101 | 7.88 / 7.82 | 5.61 / 5.92 |

The MTP gap is closed, still with the CPU sampler: the readbacks it needs no longer go through ML
Drift's per-element conversion. Single runs per window, so read the margins as within a few
percent. Peak footprint is 1–6% above Google's (unchanged by this step).

## Verdict

- Correctness: parity. Same suites pass, digit copies exact, replies identical to Google's on the
  short suite.
- Prefill, TTFT, memory: parity (within ±5%; footprint +1–6%).
- Decode with MTP: at parity since T5's first step (above). Before it, −8% to −17%: our build
  samples on the CPU (`sampler = cpu`, the GPU sampler being closed source), MTP samples four to
  five times per round, and each 262,144-logit readback went through ML Drift's per-element
  conversion. Gate G1 is met without T7; an open Metal sampler (T7) and drafting on the GPU (T12)
  remain speed work, not parity work.
