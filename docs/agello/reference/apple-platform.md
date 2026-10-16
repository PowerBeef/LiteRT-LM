# Apple platform

> **Scope:** everything Apple-specific about running this stack: Apple GPU families (Apple's
> `MTLGPUFamily` and ML Drift's own naming) and what ML Drift does on each; iOS memory (unified
> memory, the app memory limit, jetsam, `os_proc_available_memory`, the increased-memory-limit and
> extended-virtual-addressing entitlements) and what Agello measured; Metal storage modes, residency
> sets, argument buffers, Metal 4 and tensor operations as this stack uses them; the GPU simulation
> switch and its limits; the iPhone app on a Mac; the simulator; static linking of the Metal
> accelerator on iOS. Contradiction C1 (the app memory limit of an iPhone 17 Pro) is resolved here.
> **Out of scope:** how ML Drift detects the GPU and builds kernels ([ml-drift/architecture.md](ml-drift/architecture.md),
> [ml-drift/kernels.md](ml-drift/kernels.md)); the Metal backend's internals ([ml-drift/metal-backend.md](ml-drift/metal-backend.md));
> LiteRT's buffers and the delegate ([litert/runtime.md](litert/runtime.md), [litert/ml-drift-delegate.md](litert/ml-drift-delegate.md));
> the build ([litert-lm/build-and-release.md](litert-lm/build-and-release.md)); where every byte of
> E4B lives ([memory.md](memory.md)).
> **Pinned to:** LiteRT-LM `7b872544`, LiteRT `362759838`, ML Drift `15fa9e2`; what Agello ships:
> LiteRT-LM `85b1a776` with Google's prebuilt v0.18.0 `CLiteRTLM.xcframework`; Agello `93122b647a`.
> **Applies to:** Apple (iOS, macOS). **Last verified:** 2026-10-15

## Summary

- **Two family names.** Apple's `MTLGPUFamily` is what `supportsFamily(_:)` answers; ML Drift's
  "Apple<N>" is a label it derives from the Metal device *name*. They agree for the A14 to A19 and
  M1 to M5 chips; on this Mac's M6, Apple's API answers Apple11 while ML Drift says Apple10, which
  changes nothing because Apple10 is ML Drift's highest gate.
- **Two kernel paths in practice.** Agello requires the iPhone 15 Pro class, so its iPhones should be
  Apple9 (A17 Pro, A18) or Apple10 (A19). Apple10 prefill runs `ConvAppleMPP` on the GPU Neural
  Accelerators; Apple9 prefill runs `ConvWaveMatrix` (simdgroup matrices). Decode is the same kernel
  set on both.
- **The app memory limit of an iPhone 17 Pro (C1)** is 8 GiB (8,589,934,592 bytes, shown as
  "8.6 GB") with Agello's entitlements, read as footprint plus `os_proc_available_memory()`. The
  "6.4 GB" in VALIDATION.md is 6 GiB, the same reading taken early in a run, before its first engine
  load, in 6 of the 36 runs of 2026-10-15; later reads in those runs give 8 GiB. No record measures
  the limit without the entitlement.
- **Metal usage is classic Metal 2/3**: shared buffers, argument buffers (every GPU graph on
  Apple6+), a residency set that is created but not requested by
  default, no Metal 4 host objects. Metal 4 appears only as the MSL 4.0 family check and as Metal
  Performance Primitives inside `ConvAppleMPP`'s shader source.
- **The Mac runs the iPhone app** with the iOS slice of `CLiteRTLM` and the same Metal accelerator;
  its limit is the Mac's physical memory. **The simulator** is excluded by project rule; our
  framework has no simulator slice, but Google's shipped v0.18.0 framework does, with the Metal
  accelerator in it.

## Concepts

| Term | Meaning here |
| --- | --- |
| `MTLGPUFamily` (Apple) | Apple's feature families; a device supports a family if `MTLDevice.supportsFamily(_:)` returns true. Apple1 = 1001 … Apple11 = 1011, Metal3 = 5001, Metal4 = 5002 in the iOS 27.1 SDK. |
| ML Drift family | `AppleInfo::Family`, from a fixed table of GPU names ([ml-drift/architecture.md](ml-drift/architecture.md#apple-gpu-families)). Not read from `MTLGPUFamily`. |
| GPU Neural Accelerators | Apple's matrix-multiplication units inside each GPU core of the M5 and A19 GPU families. ML Drift reaches them through `ConvAppleMPP` (Metal Performance Primitives tensor operations). Not the Apple Neural Engine, which this stack doesn't use. |
| Unified memory | CPU and GPU share system memory; Metal allocations count in the app's footprint. |
| Footprint | `phys_footprint` from `task_info(TASK_VM_INFO)`: the number the per-app limit applies to. |
| App memory limit | The footprint at which iOS may terminate the app (`per-process-limit` in a jetsam report). Agello reads it as footprint + `os_proc_available_memory()`. |
| Jetsam | iOS's termination of processes for memory; a jetsam event report names the reason. |
| `recommendedMaxWorkingSetSize` | Metal's estimate of how much the GPU can allocate without hurting performance. Not the app limit. |
| Storage mode | `MTLStorageMode.shared` (CPU and GPU), `.private` (GPU only), `.memoryless` (tile memory), `.managed` (macOS only). |
| Residency set | `MTLResidencySet` (iOS 18 and later): a set of allocations Metal keeps GPU-resident. |
| Argument buffer | A Metal buffer holding references to a kernel's resources; tier 1 or tier 2 by device. |
| "Designed for iPad/iPhone" | An iOS app running unmodified on an Apple silicon Mac (`ProcessInfo.isiOSAppOnMac`). |

## How it works

### GPU families: Apple's and ML Drift's

Apple's Metal feature set tables (revision of 2026-05-21) map chips to families: A14 and the M1
series to Apple7; A15, A16 and the M2 series to Apple8; A17 Pro, the A18 series, the M3 and M4 series
to Apple9; the A19 series and the M5 series to Apple10. The table has no M6 row and no Apple11 column.
<!-- c:APL-030 --> The iOS 27.1 SDK nevertheless defines `MTLGPUFamilyApple10 = 1010` and
`MTLGPUFamilyApple11 = 1011`, and `MTLGPUFamilyMetal4 = 5002` (iOS 26 and later). <!-- c:APL-031 -->

Agello's device harness records Apple's answer: it asks `supportsFamily` for every raw value from
1001 to 1020 and keeps the highest one the device accepts, minus 1000
(`AgelloDeviceTests/DeviceHarness.swift:113-118`, Agello `93122b647a`). <!-- c:APL-032 --> On the
iPhone 17 Pro (iPhone18,1, A19 Pro) that is 10; on this Mac mini (Mac18,5, M6, running the iPhone
app) it is 11. <!-- c:APL-033 -->

ML Drift doesn't ask Metal. It lowercases `[MTLDevice name]`, finds the longest key of its table that
the name contains, and maps the chip to a family
([`ml_drift/common/gpu_info.cc:683-737`](https://github.com/PowerBeef/ml-drift/blob/15fa9e2ba632145992299e3f04786dc1f575373c/ml_drift/common/gpu_info.cc#L683-L737),
[`ml_drift/common/gpu_info.cc:739-767`](https://github.com/PowerBeef/ml-drift/blob/15fa9e2ba632145992299e3f04786dc1f575373c/ml_drift/common/gpu_info.cc#L739-L767);
details on [ml-drift/architecture.md](ml-drift/architecture.md#gpu-detection)). Side by side:

| Chip | Apple's table (2026-05-21) | `supportsFamily` observed | ML Drift |
| --- | --- | --- | --- |
| A14, M1 series | Apple7 | | Apple7 |
| A15, A16, M2 series | Apple8 | | Apple8 |
| A17 Pro, A18 series, M3, M4 series | Apple9 | | Apple9 |
| A19 series, M5 series | Apple10 | A19 Pro: 10 | Apple10 |
| A20 Pro | not listed | | Apple10 |
| M6 | not listed | M6 (Mac18,5): 11 | Apple10 |
| any name not in ML Drift's table | | | Apple1 |

The two agree on every chip both list; they differ on the M6, where ML Drift's label is one family
below Apple's answer, and ML Drift alone lists an A20 Pro. <!-- c:APL-034 --> The M6 difference has
no effect on kernels: Apple10 is the highest family ML Drift tests, and the native M6 column of
[generated/kernels-by-family.md](generated/kernels-by-family.md) uses `conv_apple_mpp` like any
Apple10 GPU. <!-- c:APL-035 -->

The Metal Shading Language version, the other input to the kernel gates, does come from Metal:
ML Drift reports MSL 3.2 when the device supports `MTLGPUFamilyMetal3` on iOS 18 or later, and MSL
4.0 when it supports family 5002 (Metal 4) on iOS 26 or later
([`ml_drift/metal/common.mm:225-272`](https://github.com/PowerBeef/ml-drift/blob/15fa9e2ba632145992299e3f04786dc1f575373c/ml_drift/metal/common.mm#L225-L272)).
Apple says Metal 4 runs on the A14 and M1 and later. <!-- c:APL-036 --> So on Agello's iOS 27
deployment target, every Apple7 or newer device should report MSL 4.0, and `SupportsConvAppleMPP`
(Metal, Apple GPU, ML Drift family Apple10 or later, MSL 4.0 or later;
[`ml_drift/common/kernels/conv_apple_mpp.cc:536-540`](https://github.com/PowerBeef/ml-drift/blob/15fa9e2ba632145992299e3f04786dc1f575373c/ml_drift/common/kernels/conv_apple_mpp.cc#L536-L540))
reduces in practice to the family test. <!-- c:APL-037 -->

### What ML Drift does on each family

| | Apple10 (A19 series, M5 series, M6) | Apple9, Apple8, Apple7 (A14 … A18, M1 … M4) |
| --- | --- | --- |
| Hardware Apple describes | GPU Neural Accelerators in each GPU core | simdgroup matrix instructions (Apple7 and later) |
| Prefill weight products | `ConvAppleMPP`, INT8 weights × activations quantized per token, INT32 accumulation | `ConvWaveMatrix`, FP16, INT4/INT8 weights read directly or converted |
| Prefill attention | `ConvAppleMPP` FP16 | `ConvWaveMatrix` FP16 |
| Decode, verify, MTP drafter | `fc1x1_int*_weights` FullyConnected, FP16 | same kernels and counts |
| E4B `prefill_1024` nodes (8K window) | 1,634 | 1,424 |
| E4B `decode` nodes | 1,726 | 1,726 |

The kernel and node-count rows are from [ml-drift/kernels.md](ml-drift/kernels.md#on-apple) (MD-ARC-048, MD-ARC-049)
and [generated/kernels-by-family.md](generated/kernels-by-family.md); the Apple10 column was measured
natively on the M6 and the other column with the simulation switch below.

Apple describes the GPU Neural Accelerators as dedicated matrix-multiplication hardware in every
shader core of the M5 and A19 GPUs, reached at the lowest level through Metal Performance Primitives'
TensorOps. <!-- c:APL-038 --> Apple also says TensorOps code runs on every Apple GPU from M1 to M5 and
falls back to shader implementations where there are no Neural Accelerators. <!-- c:APL-039 -->
ML Drift doesn't rely on that fallback: it builds `ConvAppleMPP` only for Apple10.

Apple's table lists "SIMD-scoped matrix multiply operations" from Apple7, the same threshold at
which ML Drift registers its 8×8×8 simdgroup matrix operations (with MSL 2.3 or later), so
`ConvWaveMatrix` is possible on Apple7 and newer, consistently with Apple's feature table.
<!-- c:APL-040 -->
[`ml_drift/metal/environment.mm:79-100`](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/metal/environment.mm#L79-L100)

### Which devices Agello runs on

The Agello app target requires `iphone-performance-gaming-tier`, deploys to iOS 27.0 and is
iPhone-only (`TARGETED_DEVICE_FAMILY = 1`) (`Agello.xcodeproj/project.pbxproj:679-687`, Agello
`93122b647a`). <!-- c:APL-041 --> Apple defines that capability as the graphics performance and
gaming features of the iPhone 15 Pro and iPhone 15 Pro Max (A17 Pro), available from iOS 17.
<!-- c:APL-042 --> Taking "equivalent to the iPhone 15 Pro" to mean A17 Pro or newer, Agello's
iPhones are Apple9 (A17 Pro, A18 series) or Apple10 (A19 series) in Apple's table, so ML Drift runs
one of two prefill paths on them, `ConvWaveMatrix` or `ConvAppleMPP`, and the same decode kernels
on both. <!-- c:APL-043 --> Apple doesn't list the qualifying models in the pages read for this
reference; see [Open questions](#open-questions).

On the phone itself, our build's telemetry records `Apple A19 Pro GPU, ML Drift family Apple10`
([ml-drift/architecture.md](ml-drift/architecture.md#apple-gpu-families)); the iPhone reports in
`docs/agello/results/iphone/` don't record which kernels ran.

### Memory on iOS

#### Unified memory and the footprint

On Apple GPUs the CPU and GPU share system memory; the storage mode decides who can touch an
allocation: shared memory is reachable from both, private memory from the GPU only.
<!-- c:APL-045 --> ML Drift creates every `MTLBuffer` with shared storage and standalone textures with
private storage ([ml-drift/metal-backend.md](ml-drift/metal-backend.md#memory)); in our build the
2D textures of E4B's graph are buffer-backed, so its tensors sit in shared buffers. A host read or
write through a LiteRT lock still copies the whole tensor
([litert/runtime.md](litert/runtime.md#tensor-buffers)).

Because the memory is shared, Metal allocations count in the app's footprint. In the generated
engine run on the M6 (agello-bench, E4B, 8K window, MTP on), the footprint was 1,556.1 MB, of which
the graphics ledger held 886.8 MB, 57.0% ([generated/engine-run.md](generated/engine-run.md),
"Memory"). <!-- c:APL-022 --> The iPhone breakdowns Agello records (the kernel's ledgers for
graphics, media and neural memory) show 0 GB in the neural ledger, including while the test host's own engine was loaded (`memory-host-engine-breakdown.json`).

#### The limit and how to read it

When an app's resident memory crosses the per-process limit the system imposes on apps, the app
becomes eligible for termination; a jetsam event report for an app terminated this way records the
reason `per-process-limit`. <!-- c:APL-021 -->
`os_proc_available_memory()` returns the current limit minus the app's footprint at the time of the
call, and 0 if the caller isn't an app or is already over its limit. <!-- c:APL-007 --> Apple
adds that memory limits can change during the app's life cycle and don't necessarily match the
device's physical memory. <!-- c:APL-008 -->

Agello's `MemoryProbe` reads the footprint as `phys_footprint` from `task_info(TASK_VM_INFO)` and the
headroom from `os_proc_available_memory()`; `DeviceProfile.current` sets the app limit to their sum,
or to nil when the available value is 0 (`Agello/Support/Diagnostics.swift:14-56`,
`Agello/Inference/DeviceProfile.swift:16-22`, Agello `93122b647a`). <!-- c:APL-012 -->

#### The entitlements

Agello's `Agello.entitlements` declares `com.apple.developer.kernel.extended-virtual-addressing`,
`com.apple.developer.kernel.increased-memory-limit`,
`com.apple.developer.background-tasks.continued-processing.gpu` and the App Group
`group.com.patricedery.agello` (`Agello/Agello.entitlements:5-21`, Agello `93122b647a`).
<!-- c:APL-011 -->

- **Increased memory limit.** Apple: the entitlement tells the system that core features may
  perform better above the default limit; the higher limit exists only on some device models, and
  the app should check `os_proc_available_memory()`. It is available from iOS 15. <!-- c:APL-009 -->
- **Extended virtual addressing.** Apple: for apps that need a larger addressable space, for example
  games that memory-map assets to stream to the GPU; available from iOS 14. <!-- c:APL-010 --> LiteRT-LM maps each model section
  with `mmap(PROT_READ | PROT_WRITE, MAP_PRIVATE)` and, on Apple, `madvise(MADV_DONTNEED)` so pages
  load on first use; a failed map returns "Failed to map, error: " followed by `strerror(errno)`, which
  for `ENOMEM` is the "Cannot allocate memory" that the entitlement's comment in Agello quotes.
  <!-- c:APL-028 --> E4B's file is 3,660,726,272 bytes (3.41 GiB) ([generated/models/e4b-ours.md](generated/models/e4b-ours.md)).
  <!-- c:APL-029 -->

#### What Agello measured on the iPhone 17 Pro

All iPhone records are in `docs/agello/results/iphone/2026-10-15/` (iPhone18,1, A19 Pro,
iOS 27.0.1), from Agello builds whose `Agello.entitlements` declares both memory entitlements.

| Reading (footprint + available) | Bytes | Shown as | Where | When in the process |
| --- | --- | --- | --- | --- |
| 8 GiB | 8,589,934,592 | "8.6 GB" | every reading in all 36 reports, except those in the next row | all readings in 30 runs; after the first engine load in the other 6 |
| 6 GiB | 6,442,450,944 | "6.4 GB" | in 6 of the 36 runs: the device header; `inference-ours` `00_model`; `calibrate-ours` `appLimitGB` (read once, so all four windows); `realLimitGB` of the first envelope scenario in `memory-google`, `memory-host-idle` and `memory-sim-a17-test02_E4B_at4_0GB-fresh`; `memory-google`'s 5.0 GB "ballast" step | early in the run, before its first engine load |
| Physical memory | 12,291,735,552 | "12 GB RAM" | `ProcessInfo.physicalMemory`, every report | |
| `recommendedMaxWorkingSetSize` | 8,589,950,976 | 8.59 GB | every report | |

Outside those early readings, the app limit is exactly 8 GiB: 8 × 1,073,741,824 = 8,589,934,592
bytes, which Agello's `%.1f` of bytes / 10⁹ prints as "8.6 GB". Every other footprint and available
pair read at the same moment (`memory.footprintGB` + `memory.availableGB`, a step's `footprintGB` +
`availableGB`) sums to 8.590 GB; `footprintPeakGB` + `availableGB` doesn't always, because the two
are read at different times. <!-- c:APL-001 --> <!-- c:APL-006 --> In `inference-ours.json` the first scenario, recorded at 07:25:00Z, reads 0.312 +
6.130 = 6.442 GB, and the next, at 07:25:02Z after the engine load, 2.573 + 6.016 = 8.590 GB.
<!-- c:APL-002 --> Six of the 36 iPhone reports (`inference-ours`, `calibrate-ours`, `memory-google`,
`probe06-google`, `memory-host-idle`, `memory-sim-a17-test02_E4B_at4_0GB-fresh`) carry
"app limit 6.4 GB" in their device header, which is read once when the harness starts; every one of them reads
8,589,934,592 bytes later in the same run. <!-- c:APL-003 --> `calibrate-ours.json`, for example,
records `appLimitGB` 6.44 for all four windows (read once before the first), while footprint plus
available in the same four scenarios is 8.59 GB. <!-- c:APL-024 -->

The physical memory is 12,291,735,552 bytes, so the steady limit is 69.9% of it
(8,589,934,592 / 12,291,735,552). <!-- c:APL-016 --> The GPU's `recommendedMaxWorkingSetSize` is
8,589,950,976 bytes, 16,384 bytes above 8 GiB; it is Metal's performance estimate, not the jetsam
limit. <!-- c:APL-017 -->

#### C1: 8.6 GB or 6.4 GB?

**Resolution, for the 2026-10-15 reports.** Both numbers come from the same reading
(`phys_footprint` + `os_proc_available_memory()`), on the same iPhone 17 Pro with the
increased-memory-limit entitlement. In 6 of the 36 runs, readings taken early in the run, before its
first engine load, returned 6 GiB (6,442,450,944 bytes, "6.4 GB") and later readings in the same run
8 GiB (8,589,934,592 bytes, "8.6 GB"); the other 30 runs read 8 GiB from their first reading.
Agello's ENGINEERING.md ("about 8.6 GB") states the 8 GiB value; VALIDATION.md's header copies the
device header of `inference-ours.json`, the night's first report (read at 07:25:00Z), which was one of
the early 6 GiB readings, and the iPhone results README carries the same "app limit 6.4 GB". The
archived plan's "~6.4 GB" comes from an earlier device finding (2026-10-09) with no raw record, so
whether it was also an early reading is unknown. <!-- c:APL-004 -->

**Without the entitlement.** No file in `docs/agello/results/` measures the limit without it. The
"~3.5 GB without, ~6.4 GB with" note that Agello's tests carried from 2026-10-09
(`iRTChatTests/DeviceProfileTests.swift:16-17` at Agello `8b3851e`) and the "~3.5 GB" comment in
`Agello.entitlements` have no raw record, so the default limit is unverified here. <!-- c:APL-005 -->
Apple's statement that the higher limit exists only on some models means an 8 GB iPhone may differ
too; it hasn't been measured.

Why the limit reads lower early in some runs isn't established; Apple only says limits can change
during an app's life cycle (see [Open questions](#open-questions)).

#### How Agello uses the limit

`DeviceProfile` (`Agello/Inference/DeviceProfile.swift`, Agello `93122b647a`):

| Decision | Rule | Worked numbers |
| --- | --- | --- |
| Default KV size | 8,192 tokens when physical memory ≥ 7,000,000,000 bytes, else 4,096 | iPhone 17 Pro: 12,291,735,552 ≥ 7·10⁹ → 8,192 |
| Largest context offered | 32,768 when the app limit ≥ 6,500,000,000 bytes, else 16,384 (also when unknown) | 8,589,934,592 → 32,768; 6,442,450,944 → 16,384 |
| Stop a running reply | available memory > 0 and < 400,000,000 bytes, checked on each memory warning and at each ~100 ms streaming flush | |
| Keep the chat's KV cache across a helper | only on a runtime with a build id (ours, not Google's v0.18.0), and only when available ≥ 28,672 × `maxNumTokens` + 400,000,000, or available = 0 | 16,384 tokens: 469,762,048 + 400,000,000 = 869,762,048 bytes |

<!-- c:APL-023 --> <!-- c:APL-013 --> <!-- c:APL-015 -->

The largest-context rule is read only by the KV-cache stepper in Settings → Advanced
(`Agello/Views/AdvancedSettingsView.swift:79`), shown when a manual KV-cache size is set, and by unit
tests. The stepper re-reads the limit when the view is evaluated; because 6 GiB (6,442,450,944 bytes)
is below the 6,500,000,000-byte threshold, if it is evaluated while an iPhone 17 Pro's limit reads
6 GiB, as early readings did in 6 of 36 harness runs on 2026-10-15, it ends at 16,384 tokens instead
of 32,768. <!-- c:APL-014 -->

#### The 8 GB envelope, and why three runs didn't test it

Agello's memory suite simulates a smaller limit on the 12 GB phone: it reads the limit once with no
engine loaded, allocates incompressible "ballast" equal to that reading minus the target, then loads
E4B at 8K and runs a text, an image, an audio, an 8K and a 16K turn
(`AgelloDeviceTests/MemoryEnvelopeScenarioTests.swift:46-120`). When that single reading returns the early 6 GiB,
the ballast is 2 GiB too small, and the effective limit is the later 8 GiB minus the ballast:

| Report | Target | Limit read | Ballast | Effective limit |
| --- | --- | --- | --- | --- |
| `memory-google.json` | 5.0 GB | 6.442 GB | 1.442 GB | 8.590 − 1.442 = 7.147 GB |
| `memory-host-idle.json` | 5.0 GB | 6.442 GB | 1.442 GB | 8.590 − 1.442 = 7.147 GB |
| `memory-sim-a17-test02_E4B_at4_0GB-fresh.json` | 4.0 GB | 6.442 GB | 2.442 GB | 8.590 − 2.442 = 6.147 GB |

<!-- c:APL-025 --> Every other envelope scenario on 2026-10-15 read 8.590 GB, so its effective limit
equals its target. Results where the target held:

| Target | Reports that recorded every step (through "long 16K") | Reports that end after "text" |
| --- | --- | --- |
| 5.0 GB | `memory-ours`, `memory-ours-telemetry0`, `memory-google-host-idle`, `memory-host-engine-breakdown`, `memory-host-engine-unload-log`, `memory-sim-a17`, `memory-restore-after-helper-test01_E4B_at5_0GB-fresh` | none |
| 4.0 GB | `memory-ours`, `memory-ours-telemetry0`, `memory-google`, `memory-google-host-idle`, `memory-host-idle`, `memory-host-engine-breakdown`, `memory-host-engine-unload-log`, `memory-sim-a17` | none |
| 3.5 GB | `memory-host-idle`, `memory-google-host-idle` | `memory-ours`, `memory-ours-telemetry0`, `memory-google`, `memory-host-engine-breakdown`, `memory-host-engine-unload-log`, `memory-sim-a17`, `memory-sim-a17-test03_E4B_at3_5GB-fresh` |

<!-- c:APL-026 --> <!-- c:APL-027 --> A report saves after every step, so a report that ends after
"text" means the image step never recorded; what stopped it isn't in the report. Some completed
steps are marked not `ok` (a failed reply, not a termination); the step-level detail and the
footprints are for [memory.md](memory.md).

### Metal in this stack

| Feature | Apple's minimum (feature set tables) | What this stack does | Details |
| --- | --- | --- | --- |
| Shared buffers | all | Every ML Drift `MTLBuffer` | [ml-drift/metal-backend.md](ml-drift/metal-backend.md#memory) |
| Argument buffers tier 2 | Apple6 | Requested by LiteRT-LM on Apple for every executor, the decoder included; dropped only on tier 1 | below |
| Residency sets | Apple6 (32 per queue) | Created; attached and requested only when enabled (off by default) | below |
| Binary archives | Apple3 | Not used: every kernel is compiled from source at load | [ml-drift/metal-backend.md](ml-drift/metal-backend.md) |
| SIMD-scoped matrix multiply | Apple7 | `ConvWaveMatrix` | [ml-drift/kernels.md](ml-drift/kernels.md) |
| Tensors, ML encoding, argument tables, command allocators (Metal 4 host API) | Apple7 | Not used | below |
| Metal Performance Primitives tensor operations | Metal 4 shaders | `ConvAppleMPP` on Apple10 only | [ml-drift/kernels.md](ml-drift/kernels.md#convapplempp-the-gpu-neural-accelerators) |

#### Argument buffers

LiteRT-LM's `SetCommonGpuOptions` sets `SetUseMetalArgumentBuffers(true)` under `__APPLE__`
([`runtime/executor/litert_compiled_model_executor_utils.cc:1032-1034`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/litert_compiled_model_executor_utils.cc#L1032-L1034)).
The vision, audio and embedding executors call it
([`runtime/executor/vision_litert_compiled_model_executor.cc:94-97`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/vision_litert_compiled_model_executor.cc#L94-L97)),
<!-- c:APL-051 -->
and so does the main executor's `CreateCompilationOptions`
([`runtime/executor/llm_executor_settings_utils.cc:160-164`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_executor_settings_utils.cc#L160-L164)). <!-- c:X-CFG-009 -->
LiteRT's Metal delegate drops the option only on `MTLArgumentBuffersTier1` devices
([`ml_drift_delegate/delegate/delegate_metal.mm:344-352`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/ml_drift_delegate/delegate/delegate_metal.mm#L344-L352)).
<!-- c:RT-DLG-006 --> With tier 2 from Apple6 <!-- c:APL-052 -->, every Gemma graph, the decoder
included, binds kernel arguments through argument buffers on every iPhone Agello supports. The same
call and setter are in the v0.18.0 source behind Google's binary. What that changes per dispatch
(`useResource` per resource, one argument buffer per kernel) is on
[ml-drift/metal-backend.md](ml-drift/metal-backend.md#argument-binding).

#### Residency sets

LiteRT's `GpuBackendMetal` creates a residency set on iOS 18 and later in every case, and attaches it
to the command queue, requests residency and starts a 100 ms heartbeat only when the option is on
([`ml_drift_delegate/delegate/gpu_backend_metal.mm:69-92`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/ml_drift_delegate/delegate/gpu_backend_metal.mm#L69-L92);
[ml-drift/metal-backend.md](ml-drift/metal-backend.md#residency-sets)). The option is LiteRT-LM's
`AdvancedSettings::gpu_enable_metal_residency_set`, false by default, exposed as
`litert_lm_engine_settings_set_gpu_enable_metal_residency_set` and Swift's
`ExperimentalFlags.gpuEnableMetalResidencySet` (nil unless set)
([`runtime/executor/llm_executor_settings.h:212-214`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/llm_executor_settings.h#L212-L214),
[`swift/ExperimentalFlags.swift:189-203`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ExperimentalFlags.swift#L189-L203)).
Agello sets `enableBenchmark` and `filterChannelContentFromKvCache` but not this flag
(`Agello/Inference/ChatEngine.swift:141-147`), so residency is not requested in Agello's
configuration. <!-- c:APL-048 --> Apple lists residency sets from Apple6, with at most 32 per queue,
and notes that they don't track hazards. <!-- c:APL-049 -->

#### Metal 4 and tensor operations

None of the three repositories uses the Metal 4 host API at the pins: a search of their Objective-C++,
C++, header and Swift sources finds no `MTL4…` type, `MTLTensor`, `newTensorWithDescriptor` or
`MTLBinaryArchive`. In host code Metal 4 appears only as ML Drift's `supportsFamily` check for
`MTLGPUFamilyMetal4` (5002), which selects MSL 4.0
([`ml_drift/metal/common.mm:258-270`](https://github.com/PowerBeef/ml-drift/blob/15fa9e2ba632145992299e3f04786dc1f575373c/ml_drift/metal/common.mm#L258-L270)).
Otherwise it reaches this stack as shader source: `ConvAppleMPP`'s generated kernel includes
`<MetalPerformancePrimitives/MetalPerformancePrimitives.h>` and calls `mpp::tensor_ops::matmul2d`
([`ml_drift/common/kernels/conv_apple_mpp.cc:142`](https://github.com/PowerBeef/ml-drift/blob/15fa9e2ba632145992299e3f04786dc1f575373c/ml_drift/common/kernels/conv_apple_mpp.cc#L142),
[`ml_drift/common/kernels/conv_apple_mpp.cc:302`](https://github.com/PowerBeef/ml-drift/blob/15fa9e2ba632145992299e3f04786dc1f575373c/ml_drift/common/kernels/conv_apple_mpp.cc#L302)).
<!-- c:APL-053 --> Apple lists tensors, machine-learning encoding, argument tables and command
allocators as Metal 4 features of Apple7 and later. <!-- c:APL-054 --> So the
GPU Neural Accelerators are used for prefill on Apple10, while Metal 4 tensors, the ML encoder and
the Metal 4 command model are not used anywhere.

### The GPU simulation switch

`AGELLO_SIMULATE_APPLE_GPU` (ours, ML Drift `ef08a67`) makes ML Drift build `AppleInfo` from the
variable instead of the device name
([`ml_drift/common/gpu_info.cc:1365-1376`](https://github.com/PowerBeef/ml-drift/blob/15fa9e2ba632145992299e3f04786dc1f575373c/ml_drift/common/gpu_info.cc#L1365-L1376));
what it changes and what it doesn't is on
[ml-drift/architecture.md](ml-drift/architecture.md#the-simulation-switch-ours). Agello's harness
passes it to the app as `TEST_RUNNER_AGELLO_SIMULATE_APPLE_GPU`, which `xcodebuild` hands to the
test process without the prefix (`scripts/device-harness.sh:78-79`). <!-- c:APL-056 -->

Limits that matter on Apple:

- **The name must contain a table key.** The value is lowercased and matched like a device name, so
  "Apple A17 Pro" works but "A17 Pro" contains no key (the keys start with "apple "): it leaves the
  type unknown and the family at Apple1, and our LiteRT telemetry reports a `kGpuFamilyUnknown`
  anomaly. <!-- c:APL-057 -->
  [`ml_drift_delegate/delegate/gpu_backend_metal.mm:167-181`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/ml_drift_delegate/delegate/gpu_backend_metal.mm#L167-L181)
- **Metal still describes the real device.** The harness's `supportsFamily` reading stays 10 in the
  iPhone reports run with "Apple A17 Pro" (`*-sim-a17*.json`), and the MSL version, threadgroup
  limits and maximum buffer length come from the real `MTLDevice`. <!-- c:APL-058 -->
- **Memory isn't simulated.** The same reports show the phone's 8 GiB limit; the 8 GB envelope is a
  separate ballast mechanism ([above](#the-8-gb-envelope-and-why-three-runs-didnt-test-it)).
- **A newer family than the hardware.** On iOS 26 or later the MSL 4.0 test passes on any Apple7 or
  newer GPU, so simulating an Apple10 name on Apple9 hardware would select `ConvAppleMPP` there;
  Apple says TensorOps fall back to shader code without Neural Accelerators, but this configuration
  hasn't been run. <!-- c:APL-060 -->
- **Caches.** The simulated name joins the program and weight cache keys, so real and simulated
  caches never mix ([ml-drift/architecture.md](ml-drift/architecture.md#the-simulation-switch-ours)).
- **What ships.** The switch exists only in our builds; Google's v0.18.0 binary doesn't read it.

### The iPhone app on a Mac

"Designed for iPad/iPhone" runs the unmodified iOS app on an Apple silicon Mac, on the frameworks Mac
Catalyst uses, without recompiling for the Mac. <!-- c:APL-062 --> The app is an iOS binary, so it
links the iOS framework: Agello's vendored `Package.swift` gives the `LiteRTLM` target `CLiteRTLM`
(Google's iOS xcframework) on iOS and `CLiteRTLM_mac` only on macOS
(`Vendor/LiteRT-LM/Package.swift:34-55`, Agello `93122b647a`), the same declarations as the shipped
wrapper's
[`Package.swift:35-55`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/Package.swift#L35-L55).
On the Mac the app therefore runs the `ios-arm64` slice and its Metal accelerator, the same code as on
the iPhone, not Google's macOS framework. <!-- c:APL-061 -->

What the app sees there (Mac mini M6, `docs/agello/results/mac/2026-10-14/agello/`):

| Field | Value |
| --- | --- |
| `ProcessInfo.isiOSAppOnMac` | true |
| `UIDevice.model`, system | "iPad", "iPadOS 27.0" |
| `hw.machine`, `hw.model` | "iPad8,6", "Mac18,5" |
| Metal device name | "Apple M6" (ML Drift: Apple10) |
| `supportsFamily` highest | 11 |
| Physical memory | 17,179,869,184 bytes (16 GiB) |
| Footprint + available | 17,179,869,184 bytes: the whole physical memory |
| `recommendedMaxWorkingSetSize` | 12,713,115,648 bytes (74.0% of physical) |

<!-- c:APL-018 --> <!-- c:APL-019 --> So the Mac has no jetsam-style app limit to measure (the testing skill says as much); the harness
refuses the `memory` suite (and `ui`, `screenshots`) with `PLATFORM=mac` and uses the Mac's
provisioning UDID as the destination (`scripts/device-harness.sh:55-67`). <!-- c:APL-065 -->
`os_proc_available_memory()` returned non-zero values (6.31 to 16.92 GB) in every Mac harness record
of 2026-10-14, while `DeviceProfile.keepsConversationAcrossHelpers` documents 0 as "not reported (the
Mac)" and keeps the conversation without a memory test in that case
(`Agello/Inference/DeviceProfile.swift:50-58`); so on the Mac the rule's memory test runs rather
than being skipped. <!-- c:APL-020 -->

A native macOS process (agello-bench with our `macos-arm64` slice) runs the same Metal accelerator;
the generated engine run is one ([generated/engine-run.md](generated/engine-run.md)).

### No simulator path

Agello never builds, tests or takes screenshots in the simulator (Agello `CLAUDE.md`, `docs/TESTING.md`),
and its device tests skip there (`AgelloDeviceTests/DeviceHarness.swift:286-299`). What stands behind
that rule:

- **Our framework has no simulator slice**: `//agello:CLiteRTLM` builds iOS device `arm64` and macOS
  `arm64` only, minimum OS 27.0
  ([`agello/BUILD:27-52`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/agello/BUILD#L27-L52)).
- **Google's shipped framework does have one.** The v0.18.0 `CLiteRTLM.xcframework.zip` that Agello's
  `Package.swift` pins (checksum `d765b995…`) contains `ios-arm64` and `ios-arm64-simulator` slices;
  both binaries declare a minimum of iOS 15.0 built with the iOS 26.4 SDK, link `Metal.framework`, and
  define `LiteRtStaticLinkedAcceleratorGpuDef` and the `ml_drift::ConvAppleMPP` and `ConvWaveMatrix`
  classes. <!-- c:APL-066 --> So "no Metal path in the simulator" is not a property of the shipped
  binary: it is our build's missing slice plus a project rule, and whether Google's simulator slice
  runs E4B is untested. <!-- c:APL-069 -->
- **The simulator's GPU isn't the device's.** Apple: the simulator translates Metal calls to the
  Mac's GPU and offers capabilities similar to Apple family 2, without emulating the simulated
  device's GPU. <!-- c:APL-067 --> ML Drift marks the device as a simulator when its Metal device name
  contains "simulator" (case-sensitive, on the original name) and then disables image buffers; the
  family still comes from the name table, so a name without a chip in it would be Apple1.
  <!-- c:APL-068 -->
  [`ml_drift/metal/environment.mm:103-104`](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/metal/environment.mm#L103-L104)
  [`ml_drift/common/gpu_info.cc:1594-1608`](https://github.com/PowerBeef/ml-drift/blob/15fa9e2ba632145992299e3f04786dc1f575373c/ml_drift/common/gpu_info.cc#L1594-L1608)

### Static linking of the Metal accelerator

Our build links LiteRT's ML Drift Metal accelerator into `CLiteRTLM` on iOS and macOS
(`apple_static_gpu_accelerator`, ours `43f0f5c3`), and LiteRT registers the static definition before
it would try `dlopen`; the whole path is on
[litert-lm/build-and-release.md](litert-lm/build-and-release.md#linking-the-metal-accelerator-statically).
The generated engine run logs "Statically linked GPU accelerator registered." followed by "GPU Metal"
registration ([generated/engine-run.md](generated/engine-run.md)).

Google's shipped v0.18.0 binary does the same: its `ios-arm64` slice defines
`LiteRtStaticLinkedAcceleratorGpuDef`, the global the static accelerator's initializer fills, and
carries the "GPU Metal" accelerator name and the MetalPerformancePrimitives include in its strings.
<!-- c:APL-071 --> It was built against the iOS 26.4 SDK, so ML Drift's MSL 4.0 branch (compiled only
when the SDK allows iOS 26) can be in it, provided its ML Drift source matches the pinned one, which
isn't known ([versions.md](versions.md)).

## On Apple

Every section of this page is Apple-only. What differs between the documented build and what Agello
ships:

| Item | Documented build (agello-next) | Shipped (Google's v0.18.0 binary) |
| --- | --- | --- |
| Slices | iOS device `arm64`, macOS `arm64`; minimum 27.0 | iOS device and simulator `arm64` (minimum 15.0, SDK 26.4); separate macOS framework |
| Metal accelerator | Statically linked (ours `43f0f5c3`) | Statically linked (defines `LiteRtStaticLinkedAcceleratorGpuDef`) |
| `AGELLO_SIMULATE_APPLE_GPU`, telemetry `gpu` fact | yes | no |
| Argument buffers, residency default | requested by LiteRT-LM on Apple for every executor, the decoder included; residency off by default (code at the pins) | LiteRT and ML Drift versions inside aren't pinned |
| On a Mac, inside the iPhone app | iOS slice | iOS slice (`CLiteRTLM`, not `CLiteRTLM_mac`) |

## Settings

The rows live in [configuration.md](configuration.md#the-apple-app); the ones that change Apple behavior:

| Setting | Layer | Default | Effect on Apple | Agello |
| --- | --- | --- | --- | --- |
| `com.apple.developer.kernel.increased-memory-limit` | entitlement | absent | Higher app limit on supported models (8 GiB measured on iPhone 17 Pro) | set |
| `com.apple.developer.kernel.extended-virtual-addressing` | entitlement | absent | Larger address space for the mapped model | set |
| `UIRequiredDeviceCapabilities` = `iphone-performance-gaming-tier` | Info.plist | none | iPhone 15 Pro class and newer only | set |
| `ExperimentalFlags.gpuEnableMetalResidencySet` / `gpu_enable_metal_residency_set` | Swift / `AdvancedSettings` | nil / false | Attach and request the residency set | not set |
| `use_metal_argument_buffers` | LiteRT GPU option | unset (false); requested by LiteRT-LM on Apple for every executor, the decoder included | Argument-buffer binding (dropped on tier 1) | inherited |
| `AGELLO_SIMULATE_APPLE_GPU` | environment (ours) | unset | Another Apple GPU's kernel choices | harness only |
| `PLATFORM=mac` | harness | `iphone` | Runs the iPhone app on the Mac | harness only |

## Limits and failure modes

| Symptom | Cause | What to do |
| --- | --- | --- |
| App limit reads 6.4 GB on a 12 GB iPhone | Read early in the run, before the first engine load (6 of 36 runs on 2026-10-15) | Read it again later, or don't cache it (Apple's advice for `os_proc_available_memory`) |
| The Settings KV-cache stepper stops at 16,384 tokens on an iPhone 17 Pro | `DeviceProfile` read the early 6 GiB value (threshold 6,500,000,000) | Reopen Settings; consider a threshold below 6 GiB or a later read |
| An 8 GB envelope run passes too easily | The limit was read early, ballast 2 GiB too small | Check `realLimitGB` is 8.59 in the report |
| Engine re-creation fails to map a section ("Cannot allocate memory") | Address space exhausted by mappings | The extended-virtual-addressing entitlement |
| Kernels slow on a new chip | Its name isn't in ML Drift's table → Apple1 | Telemetry `gpu_family_unknown`; add the name upstream |
| `AGELLO_SIMULATE_APPLE_GPU` has no effect, or everything is slow | Value without the "Apple " prefix → Apple1 | Use "Apple A17 Pro", "Apple M2", … |
| Prefill numbers differ between iPhone generations | Apple10 runs INT8 `ConvAppleMPP`, Apple9 FP16 `ConvWaveMatrix` | Expected; compare replies, not bits |
| Something works on the Mac but not on the phone | The Mac's limit is its physical memory, its GPU is bigger | Every change still needs an iPhone run |
| Tests silently skipped | Run in the simulator | Use the iPhone or `PLATFORM=mac` |

## Where in the code

| Concept | Symbol | Citation |
| --- | --- | --- |
| Name → ML Drift family | `AppleInfo`, `GetGpuFamily` | [`ml_drift/common/gpu_info.cc:739-767`](https://github.com/PowerBeef/ml-drift/blob/15fa9e2ba632145992299e3f04786dc1f575373c/ml_drift/common/gpu_info.cc#L739-L767) |
| Metal device → `GpuInfo` | `CreateGpuInfoFromMetalDevice` | [`ml_drift/metal/environment.mm:30-106`](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/metal/environment.mm#L30-L106) |
| MSL version from Metal families | `GetMaxSupportedMTLLanguageVersion` | [`ml_drift/metal/common.mm:257-270`](https://github.com/PowerBeef/ml-drift/blob/15fa9e2ba632145992299e3f04786dc1f575373c/ml_drift/metal/common.mm#L257-L270) |
| Neural Accelerator gate | `SupportsConvAppleMPP` | [`ml_drift/common/kernels/conv_apple_mpp.cc:536-540`](https://github.com/PowerBeef/ml-drift/blob/15fa9e2ba632145992299e3f04786dc1f575373c/ml_drift/common/kernels/conv_apple_mpp.cc#L536-L540) |
| Simulation switch (ours) | `GetGpuInfoFromDeviceDescription` | [`ml_drift/common/gpu_info.cc:1365-1376`](https://github.com/PowerBeef/ml-drift/blob/15fa9e2ba632145992299e3f04786dc1f575373c/ml_drift/common/gpu_info.cc#L1365-L1376) |
| Simulator branch | `SupportsImageBuffer` | [`ml_drift/common/gpu_info.cc:1594-1608`](https://github.com/PowerBeef/ml-drift/blob/15fa9e2ba632145992299e3f04786dc1f575373c/ml_drift/common/gpu_info.cc#L1594-L1608) |
| Storage modes | `CreateBuffer`, texture allocation | [`ml_drift/metal/buffer.mm:65-95`](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/metal/buffer.mm#L65-L95), [`ml_drift/metal/metal_spatial_tensor.mm:72-110`](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/metal/metal_spatial_tensor.mm#L72-L110) |
| Argument buffers requested | `SetCommonGpuOptions` | [`runtime/executor/litert_compiled_model_executor_utils.cc:1009-1041`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/litert_compiled_model_executor_utils.cc#L1009-L1041) |
| Decoder options (SetCommonGpuOptions call) | `CreateCompilationOptions` | [`runtime/executor/llm_executor_settings_utils.cc:160-164`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_executor_settings_utils.cc#L160-L164) |
| Argument buffers dropped (tier 1) | `CreateMlDriftMetalDelegate` | [`ml_drift_delegate/delegate/delegate_metal.mm:344-352`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/ml_drift_delegate/delegate/delegate_metal.mm#L344-L352) |
| Residency set | `GpuBackendMetal` constructor, `InitResidencySet` | [`ml_drift_delegate/delegate/gpu_backend_metal.mm:107-121`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/ml_drift_delegate/delegate/gpu_backend_metal.mm#L107-L121) |
| Residency option | `litert_lm_engine_settings_set_gpu_enable_metal_residency_set` | [`c/engine.h:1071-1092`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/c/engine.h#L1071-L1092), [`runtime/executor/llm_executor_settings_utils.cc:97-100`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_executor_settings_utils.cc#L97-L100) |
| Model mapping | `MemoryMappedFile::Create` | [`support/util/memory_mapped_file_posix.cc:132-145`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/support/util/memory_mapped_file_posix.cc#L132-L145) |
| Static accelerator (ours) | `apple_static_gpu_accelerator` | [`runtime/executor/BUILD:252-279`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/BUILD#L252-L279) |
| Static registration | `LiteRtStaticLinkedAcceleratorGpuDef` | [`litert/runtime/accelerators/gpu/ml_drift_metal_accelerator.cc:167-182`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/accelerators/gpu/ml_drift_metal_accelerator.cc#L167-L182), [`litert/runtime/accelerators/gpu_registry.cc:88-125`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/accelerators/gpu_registry.cc#L88-L125) |
| Framework slices | `//agello:CLiteRTLM` | [`agello/BUILD:27-52`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/agello/BUILD#L27-L52) |
| App limit (Agello) | `MemoryProbe`, `DeviceProfile.current` | `Agello/Support/Diagnostics.swift:14-56`, `Agello/Inference/DeviceProfile.swift:16-22` (Agello `93122b647a`) |
| Policy (Agello) | `DeviceProfile` | `Agello/Inference/DeviceProfile.swift:31-77` |
| Harness device record | `DeviceHarness` | `AgelloDeviceTests/DeviceHarness.swift:66-118` |
| Entitlements | `Agello.entitlements` | `Agello/Agello.entitlements:5-21` |

## Open questions

- **Why the limit reads lower early in some runs.** On the iPhone 17 Pro, readings taken before the
  run's first engine load were 6 GiB in 6 of 36 runs, and later readings in those runs were 8 GiB.
  Whether this follows the app's lifecycle state (launch versus foreground-active), the test runner,
  the engine load, or time isn't known;
  an app-only probe that logs the reading every 100 ms from `didFinishLaunching` would settle it.
- **The default limit without the entitlement**, on the iPhone 17 Pro and on an 8 GB iPhone, is
  unmeasured; the "~3.5 GB" figure has no record.
- **An 8 GB iPhone's limit with the entitlement** is unmeasured; Apple says the increase exists only
  on some models.
- **Which iPhones meet `iphone-performance-gaming-tier`.** Apple's support page lists them through a
  filter the pages read here don't show; this page assumes A17 Pro and newer.
- **Kernels on the iPhone.** No iPhone report records kernel names; the Apple10 iPhone path is
  inferred from the code, the telemetry family and the M6 tables.
- **A19 (non-Pro), A18 and M5 device names** aren't observed; their families come from the tables.
- **M6 and Apple11.** Whether Apple11 adds features ML Drift could use, and whether Apple's tables
  will list the M6 as Apple11, is unknown; Apple's 2026-05-21 tables stop at Apple10.
- **Google's simulator slice.** Whether the shipped v0.18.0 simulator slice can run E4B on the
  simulator's Apple2-class device isn't tested (and Agello's rule is not to try).
- **Residency sets' effect** on latency or memory pressure on the iPhone isn't measured.
- **Continued processing with GPU.** Agello's ENGINEERING.md says the iPhone 17 Pro on iOS 27 reports
  no background GPU resource; no results file records it.
