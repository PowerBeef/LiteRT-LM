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
| ui | 22/22 later that morning (build `ab729fd0b59d`, the screenshot tour skipped as opt-in), in 13.4 min. It needs Settings → Developer → Enable UI Automation turned on first: otherwise the runner waits 60 s for a passcode prompt and fails |

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

### A17/A18-class paths, simulated (no older device available)

ML Drift picks kernels from the Apple GPU family it infers from the device name. With
`AGELLO_SIMULATE_APPLE_GPU="Apple A17 Pro"` (ML Drift `ef08a67`, LiteRT `362759838`; Agello's
harness passes it through), it picks the Apple9 kernels of an iPhone 15 Pro or 16 Pro on our
hardware: simdgroup-matrix convolutions instead of the Neural Accelerator ones, and every other
per-family choice. The simulated name keys its own program and weight caches. This covers the code
paths; speed, thermals and the older hardware's own arithmetic stay this device's. Every scenario's
telemetry reported "ML Drift family Apple9".

| Check | Result |
| --- | --- |
| Mac, digit copies at a 32K window (`short`, `long`, `cross4096`) | 70/70 exact, replies identical to the Apple10 paths (65/65) |
| Mac, prefill / decode with MTP (700 tokens, 8K) | 703 / 144.8 tok/s against 2,547 / 162.0 on the Apple10 paths |
| iPhone 17 Pro, `inference` | 21/21 |
| iPhone 17 Pro, `probes` (01–10, 14) | 11/11, digit fidelity and the position boundary included |
| iPhone 17 Pro, `calibrate` | passes; prefill 249–288 tok/s (TTFT 4.6 / 8.0 / 16.4 / 34.6 s at 4K / 8K / 16K / 32K), decode 27.6–36.4 tok/s with MTP, thermal state 1–2 |
| iPhone 17 Pro, `memory`, each envelope in a fresh launch | 4.0 GB: every step passes; 3.5 GB: killed after the first message, starting from 1.29 GB in use (as on the Apple10 paths and with Google's binary) |

The Apple9 paths are correct. Prefill is 4–5× slower there because these GPUs have no Neural
Accelerators. Google's binary would use the same kernels on these devices, and our bounded
attention still runs. The local-window start (T4 step 2) was only in the Neural Accelerator
convolution then; `ConvWaveMatrix` has it since (below).
Run in sequence, the memory suite carries 0.8–1.4 GB from one test to the next, which made the
4.0 GB envelope fail its audio step at the 400 MB stop. Fresh launches are the fair simulation.

## 2026-10-15, afternoon: follow-ups

Agello `6d389e6` and `1220c74` on build `11beef01e805` (`agello-next` `12a0a818`, ML Drift
`ef08a67`); the kernel window on build `fc71d704a349` (the same with ML Drift `15fa9e2`). Reports:
`*-restore-after-helper*`, `calibrate-follow-up-*`, `*-sim-a17-window.json`.

### The kept conversation comes back right after the helper (open issue 2)

Agello's `endHelper` now restores the chat's conversation as soon as the helper ends (it asks the
runtime for the token count, which swaps the snapshot back), and dictation runs inside a helper
scope too. The copy lives only while the helper runs.

| | Before (`2e54fbdb9ca9`) | Now | Google's v0.18.0 |
| --- | --- | --- | --- |
| Calibration peak, 4K / 8K | 2.56–2.58 / 2.72–2.89 GB | 2.55 / 2.73 GB | 2.55 / 2.58 GB |
| Calibration peak, 16K / 32K | 4.02–4.13 / 5.88–5.93 GB | 3.59 / 4.93 GB | 3.33 / 5.06 GB |
| 5.0 GB envelope, after the first text message | 2.80–2.99 GB | 2.66 GB | 2.56–2.65 GB |
| `inference` | 21/21 | 21/21 | |

### Calibration measures decode on a longer reply

The calibration's replies (40–54 tokens) were too short to measure decode. Each window now gets a
follow-up asking for ten lines (121–124 tokens), recorded as `longDecodeTokPerSec` with the
footprint and thermal state after it, and since then Low Power Mode. Two runs back to back:

| Window | Follow-up decode tok/s, run 1 / run 2 | Short-reply decode tok/s, run 1 / run 2 |
| --- | --- | --- |
| 4K | 79.8 / 48.1 | 38.9 / 22.2 |
| 8K | 75.3 / 46.8 | 40.4 / 26.2 |
| 16K | 71.5 / 47.0 | 44.3 / 32.4 |
| 32K | 65.4 / 46.3 | 42.2 / 31.5 |

Both runs reported thermal state 0 throughout, yet the second was 29–40% slower on the follow-up
(25–43% on the short replies). The thermal state doesn't show the phone's performance state:
compare builds only after the same cool-down, alternating them, three runs or more.

### A17/A18-class paths: the local window in `ConvWaveMatrix` (ML Drift `15fa9e2`)

The simdgroup-matrix convolution that runs attention products on Apple9 GPUs (no Neural
Accelerators) now honors the local-window start, as the Neural Accelerator kernel has since T4
step 2. A PV product starts its K loop at the window's first slice; a QK product runs no K
iterations in workgroups whose keys all lie below the window, where the fused mask masks them.

| Check | Result |
| --- | --- |
| Mac, A17 paths, 32K window, window off / on in one binary | prefill +2.4% for a 2K prompt, +20.0% at 16K, +35.8% at 30K (581 → 789 tok/s); replies identical |
| Mac, digit copies (`short`, `long`, `cross4096`) | 70/70 exact, identical to the paths without the change |
| iPhone 17 Pro, A17 paths: `inference` | 21/21 |
| iPhone 17 Pro, A17 paths: probes 04, 05, 10 (digits with and without MTP, raw stream, position boundary) | 3/3 |
| iPhone 17 Pro, A17 paths: `calibrate` | prefill 265 / 302 / 288 / 278 tok/s at 4K / 8K / 16K / 32K (before: 277 / 288 / 268 / 249); TTFT 31.0 s at 32K (before 34.6) |

On the phone the gain grows with the prompt (+12% for 8,607 tokens), as on the Mac, but the earlier
run was hotter at 16K and 32K (thermal state 2 against 1): the Mac's A/B is the measurement.

## Open issues

1. **CPU backend: 5.5 GB of XNNPACK buffers. Fixed the same day** (`agello-next` `b8f807ec`,
   build `ab729fd0b59d`).
   - **Symptom.** Our CPU engine for E4B at 8K sat at 6.3 GB of footprint on the Mac, against
     0.84 GB for Google's v0.18.0 and 1.0 GB for Google's nightly. On the iPhone, iOS killed it.
   - **Cause** (`malloc_history`): 5.56 GB in 130 `reshape_convert_operator` calls. XNNPACK's
     dequant-BMM rewrite turns `dequant(INT8 KV cache) → batch matmul` into
     `convert(qint8 → qcint8) → batch matmul`. The convert then sized its per-channel scales as
     channels × every dim but the last, where the matmul reads batch dims × channels: n² × batches
     for a transposed K cache, about 0.5 GB for one 8K cache.
   - **Why only our build.** Upstream XNNPACK fixed this on 2026-09-23 (`c72e2a4ae0`).
     TensorFlow's XNNPACK pin (`d89ef666`), which our source build uses, predates the fix;
     Google's binaries include it.
   - **Fix.** The fork's WORKSPACE declares XNNPACK at TensorFlow's pin with upstream's diff
     applied (`agello/patches/xnnpack_qs8_qc8_scales.patch`). Drop it when TensorFlow's pin moves
     past that commit.
   - **Results.** Mac, E4B on CPU at 8K: 0.94–2.09 GB instead of 6.49–9.61 GB; three digit
     copies exact and byte-identical to the unpatched build; decode 11.0 → 12.0 tok/s. iPhone:
     probe 6 passes, CPU reply correct, 3.67 GB with 4.92 GB still available
     (`probe06-ours-xnnpack-fix.json`); `inference` 21/21 (`inference-ours-xnnpack-fix.json`).
2. **Kept conversations held one KV copy until the next message. Fixed** (Agello `6d389e6`):
   the chat's conversation is restored as soon as the helper ends, so the copy lives only while
   the helper runs (above).
3. **The plan's A17/A18-class run:** the Apple9 code paths pass in simulation (above). A real
   device would still show speed, thermals, and the hardware arithmetic behind #3913 below 2,048
   (our model fix is exact by construction there). A cloud device farm can provide one.
4. **The 3.5 GB envelope** (the harshest 8 GB-phone case) fails with every build, Google's included.
   Memory isn't fully returned after an engine unloads, and the base before loading varies from
   0.5 to 1.5 GB.
