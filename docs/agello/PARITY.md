# Parity: our source build against Google's nightly (M1.6)

Our build (`agello-next`: the 0.19.0-nightly.20261013 source, the tool-call patch, LiteRT's Metal
accelerator linked statically, response formats without the closed constraint provider) against
Google's prebuilt binaries of the same nightly. Mac mini M6 (GPU family Apple11), macOS 27.0.1;
Agello runs as an iPhone app on the Mac (`PLATFORM=mac`), so both use the iOS slice and Metal.
Raw results: Agello's `build/mac-harness-next/` (ours) and `build/mac-harness-nightly/` (Google's).

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

## Verdict

- Correctness: parity. Same suites pass, digit copies exact, replies identical to Google's on the
  short suite.
- Prefill, TTFT, memory: parity (within ±5%; footprint +1–5%).
- Decode with MTP: **not at parity** (−8% to −17%). Gate G1 applies: telemetry shows our build
  samples on the CPU (`sampler = cpu`, the GPU sampler being closed source), and MTP samples four
  to five times per round, each a 262,144-logit readback. Next: T7, an open Metal sampler; then
  T12 (drafting on the GPU).
