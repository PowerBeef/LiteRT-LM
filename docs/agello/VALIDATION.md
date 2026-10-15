# Validation on the iPhone

One section per iPhone run of our build. Raw reports in `results/iphone/<date>/`.

## 2026-10-15: iPhone 17 Pro, build `2e54fbdb9ca9`

A19 Pro (Apple10), 12 GB, app limit 6.4 GB, iOS 27.0.1. Runtime: LiteRT-LM `67d00f36`, LiteRT
`97e06d36`, ML Drift `2184a7f` (everything through P6 and the T11 re-pin). Agello `main` at
`ecd8dc6`. The phone ran unattended overnight.

| Suite | Result |
| --- | --- |
| unit | 227 passed, 1 skipped |
| inference | 21/21 |
| probes (01–10, 14) | 10/11: `test06` killed by iOS while loading a CPU engine (open issue 1) |
| files | 3/3 |
| thinking | 2/2 |
| web | 3/3 |
| memory | 2/3: the 3.5 GB envelope killed after its first message; Google's v0.18.0 does the same the same night |
| calibrate | passes; numbers below |
| ui | not run: XCUITest needs the passcode entered on the phone to enable automation mode, so it needs someone there |

Highlights:

- **T3, kept conversations:** `test11` (title helper, then recall) prefilled 409 tokens with
  2 KV copies, and the recall is right. Probe 1 keeps the context: helper 0.39 s, +0.24 GB.
- **T11, one weight cache:** probe 14 leaves one folder, 2.47–2.56 GB, through 8K → 4K → 16K → 8K.
  Loads take 7.2 s (cold), then 1.5, 1.7 and 1.0 s.
- **#3913:** probe 10 (position boundary) passes.

### Calibration, phone cool (thermal state 0), MTP on

Each build ran a warm-up calibration, then a 15-minute cool-down, then the measured run; ours
before and after Google's (`calibrate-cooled-*.json`):

| Window | Decode tok/s, Google / ours | Prefill tok/s, Google / ours | TTFT s, Google / ours | Peak GB, Google / ours |
| --- | --- | --- | --- | --- |
| 4K (1,285 tokens) | 37.2 / 38.7–40.1 | 850 / 1,059–1,365 | 1.51 / 0.94–1.21 | 2.55 / 2.56–2.58 |
| 8K (2,302) | 38.5 / 39.7–42.2 | 1,135 / 1,561–1,603 | 2.03 / 1.44–1.47 | 2.58 / 2.72–2.89 |
| 16K (4,398) | 30.8 / 39.4–46.8 | 779 / 1,363–1,455 | 5.64 / 3.02–3.23 | 3.33 / 4.02–4.13 |
| 32K (8,607) | 20.3 / 37.3–44.6 | 467 / 1,120–1,247 | 18.41 / 6.90–7.68 | 5.06 / 5.88–5.93 |

The higher peak at 16K and 32K is the kept conversation (open issue 2). Replies are short (27–56
tokens), so decode rates carry about ±10% between runs.

Thermal state matters more than any change measured here. The same calibration right after 20
minutes of other suites read 22 / 26 / 31 / 31 tok/s decode, at thermal state 2 ("serious").
`CalibrationScenarioTests` now records the thermal state after load and after the reply.

### Memory envelopes (`MemoryEnvelopeScenarioTests`, footprint minus ballast)

| | Ours | Ours, telemetry off | Google's v0.18.0 |
| --- | --- | --- | --- |
| Before loading, tests 1 → 2 → 3 | 0.62 → 2.02 → 1.42 GB | 0.58 → 1.95 → 1.42 GB | 0.31 → 1.58 → 1.48 GB |
| After the first text message (8K) | 2.80–2.99 GB | 2.79–2.97 GB | 2.56–2.65 GB |
| 16K | 3.50 GB | 3.50 GB | 3.50 GB |
| 3.5 GB envelope | killed after the first message | same | same |

Memory left over between tests happens with Google's binary too; telemetry costs nothing
measurable. Ours is about 0.25 GB higher after the first message: the kept conversation's KV copy
at 8K (0.23 GB).

## Open issues

1. **CPU backend: 5.5 GB of XNNPACK buffers.** On the Mac, our CPU engine for E4B at 8K sits at
   6.3 GB of footprint against 0.84 GB for Google's v0.18.0 and 1.0 GB for Google's nightly,
   with the same weight cache file and the same speed (11 tok/s). `malloc_history`: 5.56 GB in
   130 `reshape_convert_operator` calls, XNNPACK's `convert_nc_qs8_qc8`. It allocates
   channels × batch floats per operator to hold one repeated scale, for the INT8 KV caches.
   Upstream XNNPACK still has that code; Google's internal build doesn't take this path. On the
   iPhone, probe 6's CPU engine is killed (6.88 GB); Google's binary passes it. Agello offers CPU
   in Settings → Advanced and falls back to it when the GPU load fails. Before Agello ships our
   runtime: fix the CPU path or offer only the GPU with it.
2. **Kept conversations hold one KV copy until the next message**, not only while the helper
   runs. The chat's snapshot waits in its session until it's restored: +0.23 GB at 8K, about
   +0.9 GB at 32K, between messages. The policy budget is right (one copy); the T3 note and
   Agello's docs said "nothing afterwards". Restoring right after the helper would release it.
3. **`ui` suite:** run it attended.
4. **The plan's A17/A18-class run** (Apple9 fallbacks, #3913 below 2,048) is still to do.
