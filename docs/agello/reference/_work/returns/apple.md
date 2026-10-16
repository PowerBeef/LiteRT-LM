# Returns: apple (B3), apple-platform.md

## Settings rows (for configuration.md)

name | layer | default | where read (citation link) | effect on Apple | version added | Agello sets it? | notes
--- | --- | --- | --- | --- | --- | --- | ---
`com.apple.developer.kernel.increased-memory-limit` | app entitlement (Apple) | absent | iOS; Agello `Agello/Agello.entitlements:10-12` (93122b647a) | Raises the per-app limit on supported models: 8 GiB (8,589,934,592 bytes) measured on iPhone 17 Pro (APL-001) | iOS 15.0 (APL-009) | yes | Unentitled default unmeasured (APL-005)
`com.apple.developer.kernel.extended-virtual-addressing` | app entitlement (Apple) | absent | iOS; `Agello/Agello.entitlements:5-9` | Larger address space for the mmap'd 3.66 GB model (APL-010, APL-028) | iOS 14.0 | yes |
`UIRequiredDeviceCapabilities = iphone-performance-gaming-tier` | Info.plist (Xcode build setting `INFOPLIST_KEY_UIRequiredDeviceCapabilities`) | none | App Store install check; `Agello.xcodeproj/project.pbxproj:679` | iPhone 15 Pro class only: Apple9/Apple10 GPUs (APL-041..043) | iOS 17.0 | yes |
`IPHONEOS_DEPLOYMENT_TARGET` | Xcode build setting | — | `project.pbxproj:681` | 27.0: every `@available(iOS 18/26)` Metal branch active; MSL 4.0 on Metal 4 devices (APL-037) | — | yes (27.0) |
`ExperimentalFlags.gpuEnableMetalResidencySet` / `litert_lm_engine_settings_set_gpu_enable_metal_residency_set` / `AdvancedSettings::gpu_enable_metal_residency_set` | Swift / C / C++ | nil / false | [`runtime/executor/llm_executor_settings_utils.cc:97-100`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_executor_settings_utils.cc#L97-L100) | true: attach the "LiteRT Model Residency Set" to the queue, requestResidency, 100 ms heartbeat (iOS 18+); false: set created but not requested | C API 0.2.0 | no (APL-048) |
`use_metal_argument_buffers` | LiteRT GPU option → ML Drift `ModelHints` | set true by LiteRT-LM under `__APPLE__` | [`runtime/executor/litert_compiled_model_executor_utils.cc:1032-1034`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/litert_compiled_model_executor_utils.cc#L1032-L1034), [`ml_drift_delegate/delegate/delegate_metal.mm:344-352`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/ml_drift_delegate/delegate/delegate_metal.mm#L344-L352) | Argument-buffer binding on tier-2 devices (Apple6+), dropped on tier 1 (APL-051) | — | inherited |
`AGELLO_SIMULATE_APPLE_GPU` (ours) | environment variable | unset | [`ml_drift/common/gpu_info.cc:1365-1376`](https://github.com/PowerBeef/ml-drift/blob/15fa9e2ba632145992299e3f04786dc1f575373c/ml_drift/common/gpu_info.cc#L1365-L1376) | ML Drift family/core-count decisions of the named GPU; value must contain "Apple …" (APL-057); not Metal's family, MSL version or memory (APL-058) | ML Drift `ef08a67` | harness only, as `TEST_RUNNER_AGELLO_SIMULATE_APPLE_GPU` (APL-056) | Not in the shipped binary
`PLATFORM` (`iphone` / `mac`) | Agello harness env | `iphone` | `scripts/device-harness.sh:55-67` (Agello) | `mac`: runs the iPhone app on the Mac (iOS slice, same Metal path), refuses `ui`, `screenshots`, `memory` (APL-065) | — | harness only |

## Terms (for glossary.md)

term | meaning | prefer / avoid
--- | --- | ---
Apple family (Apple's) | `MTLGPUFamily.appleN`, what `MTLDevice.supportsFamily` answers | prefer "Apple's `MTLGPUFamily` AppleN" when it matters which one
ML Drift family | `AppleInfo::Family`, derived from the Metal device name by ML Drift's table | prefer "ML Drift family AppleN"; avoid using "Apple10" alone for an M6 (Apple says Apple11)
GPU Neural Accelerators | Apple's matrix units in each GPU core of M5/A19-class GPUs, reached by `ConvAppleMPP` via MPP TensorOps | prefer; avoid "Neural Engine", "ANE" (not used)
App memory limit | The footprint at which iOS may jetsam the app; read as `phys_footprint` + `os_proc_available_memory()` | prefer "app memory limit"; avoid "available RAM", "app RAM"
Footprint | `phys_footprint` (`TASK_VM_INFO`), the quantity the limit applies to | prefer "footprint"; avoid "resident size", "RSS"
Jetsam | iOS terminating a process for memory (`per-process-limit` and other reasons) | prefer "jetsam" or "terminated for memory"; avoid "OOM crash"
Working set (`recommendedMaxWorkingSetSize`) | Metal's performance estimate for GPU allocations | avoid using it as the app limit
"Designed for iPad/iPhone" / iOS app on Mac | The unmodified iOS app on an Apple silicon Mac (`isiOSAppOnMac`) | prefer "the iPhone app on the Mac"; avoid "Catalyst" (different thing)
Simulation switch | `AGELLO_SIMULATE_APPLE_GPU`: another Apple GPU's kernel choices on this device | prefer "simulated family"; avoid "emulated GPU" (speed, memory and arithmetic stay real)
Early limit reading | The 6 GiB value footprint + available returns in the first seconds after launch on an iPhone 17 Pro | avoid quoting it as the app limit

## Contradictions

- **C1** (app memory limit, iPhone 17 Pro: 8.6 GB vs 6.4 GB). Both are footprint + `os_proc_available_memory()` on the same iPhone 17 Pro with the increased-memory-limit entitlement: 8.6 GB is 8 GiB (8,589,934,592 bytes), the value for the life of the app; 6.4 GB is 6 GiB (6,442,450,944 bytes), what the same reading returns in the first seconds after a launch (6 of 36 iPhone reports, each later reading 8 GiB). VALIDATION.md and the results README copied the device header of the first overnight report; no record measures the limit without the entitlement. Claims APL-001, APL-002, APL-003, APL-004, APL-005, APL-006, APL-024.
- **NEW (simulator)**: "the simulator has no Metal path for LiteRT-LM" (Agello CLAUDE.md, docs/TESTING.md; Run A build-and-release.md and metal-backend.md) vs Google's shipped v0.18.0 xcframework, which has an `ios-arm64-simulator` slice that links Metal.framework and defines the statically linked Metal accelerator and ML Drift kernels. Resolution: our framework has no simulator slice and Agello's rule excludes the simulator; the shipped binary's simulator slice exists and is untested. Claims APL-066, APL-069.
- **NEW (Agello comment)**: `DeviceProfile.keepsConversationAcrossHelpers` documents `availableBytes == 0` as "the Mac", but the iPhone app on the Mac reports 11.3–16.9 GB available (limit = 16 GiB physical). Claim APL-020.
- **NEW (envelope validity)**: three memory-envelope scenarios (memory-google 5.0 GB, memory-host-idle 5.0 GB, memory-sim-a17-test02 4.0 GB) read the early 6 GiB limit, so their effective limits were 7.147, 7.147 and 6.147 GB, not the targets. Claim APL-025.

## Notes for other pages

- **memory.md**: see "Inputs for memory.md" below.
- **litert-lm/build-and-release.md** ("There is no simulator slice: LiteRT-LM has no Metal path in the simulator"): true for our `//agello:CLiteRTLM` only. Google's v0.18.0 `CLiteRTLM.xcframework.zip` (sha256 `d765b995…`) has `ios-arm64` and `ios-arm64-simulator` slices, both minos 15.0 / SDK 26.4, both linking Metal.framework and defining `_LiteRtStaticLinkedAcceleratorGpuDef` and the `ml_drift::ConvAppleMPP`/`ConvWaveMatrix` vtables (`nm -U`, `otool -l`, `otool -L` on the SwiftPM artifact cache copy). Suggest: "Our framework has no simulator slice; Agello doesn't run in the simulator." (APL-066, APL-069). The same page's "Google's own builds fill it internally" is now evidenced for v0.18.0 iOS: `LiteRtStaticLinkedAcceleratorGpuDef` is defined in Google's binary (APL-071).
- **ml-drift/metal-backend.md**, On Apple: "there is no Metal path for LiteRT-LM in it" — same correction as above. Its open question "Argument buffers on device" can close: the delegate drops them only on `MTLArgumentBuffersTier1`, Apple's feature set tables put tier 2 at Apple6+, so every Apple6+ GPU, and every iPhone Agello supports, uses them (APL-051, APL-052).
- **ml-drift/architecture.md**: the family table is right, but "named after `MTLGPUFamily`" could note that the labels coincide with Apple's for A14–A19 and M1–M5 and differ on the M6 (Apple's `supportsFamily` answers 11 on Mac18,5) (APL-033, APL-034). Its open question on simulating a newer family: Apple states TensorOps run on M1–M5 and fall back to shader code without Neural Accelerators, and Metal 4 covers A14/M1+, so the MSL gate would pass; still untested (APL-039, APL-060).
- **versions.md**: Google's v0.18.0 iOS binary declares minos 15.0 and was built with the iOS 26.4 SDK (LC_BUILD_VERSION), so ML Drift's MSL 4.0 branch can be compiled in (APL-073).
- **tools.md**: harness plumbing for the simulation switch is `TEST_RUNNER_AGELLO_SIMULATE_APPLE_GPU` (APL-056); a value without the "Apple " prefix silently selects Apple1 and raises `gpu_family_unknown` (APL-057).
- **troubleshooting.md**: the "Limits and failure modes" table of apple-platform.md (early limit reading, stepper capped at 16,384, envelope ballast too small, `ENOMEM` on mmap).
- **Agello (not this repo; for the user)**: `DeviceProfile.maxContextTokens`' 6,500,000,000-byte threshold sits just above the early 6 GiB reading (APL-013, APL-014); the envelope test should re-read the limit or check `realLimitGB` ≈ 8.59 (APL-025); `CalibrationScenarioTests` reads `appLimitGB` once at start (APL-024).
- **No disagreement found** with the Run A claims reused here (MD-ARC-003/004/006/007/016/017/019, MD-MTL-015/027/028/029/035; MD-MTL-017 is "wrong" in the ledger (argument buffers are on), which this page agrees with, MD-KRN-001..004/016/079/081, LM-BLD-012/013/015/031).

## Open questions

- Why footprint + available reads 6 GiB in the first seconds after a launch and 8 GiB after: lifecycle state, test runner, or time. An app-only probe logging the reading every 100 ms from launch would settle it.
- The unentitled limit on the iPhone 17 Pro, and the limits (with and without the entitlement) on an 8 GB iPhone: no records.
- Which iPhones Apple admits under `iphone-performance-gaming-tier` (the support page's table is behind a filter); the page assumes A17 Pro and newer.
- Kernel names on the iPhone: not recorded in any report (APL-074).
- Device names of the A19 (non-Pro), A18, M5: not observed.
- Whether Google's simulator slice can run E4B in the simulator (untested by rule).
- Effect of residency sets on the iPhone: unmeasured.
- Continued processing with background GPU on the iPhone 17 Pro (ENGINEERING.md says unsupported there): no results file.

## Inputs for memory.md

Apple facts (all with claim IDs; numbers exact):

- **App memory limit, iPhone 17 Pro (12 GB, iOS 27.0.1, with increased-memory-limit + extended-virtual-addressing):** 8,589,934,592 bytes = 8 GiB, steady (APL-001); early-launch reading 6,442,450,944 bytes = 6 GiB (APL-002, APL-003); C1 resolution (APL-004). Without the entitlement: unmeasured (APL-005).
- **Physical memory** iPhone 17 Pro: 12,291,735,552 bytes; limit = 69.9% of it (APL-016). **GPU working set** (`recommendedMaxWorkingSetSize`): 8,589,950,976 bytes, a performance hint, not a limit (APL-017).
- **Mac (iPhone app on Mac mini M6, 16 GiB):** footprint + available = 17,179,869,184 bytes = physical memory; working set 12,713,115,648 bytes (APL-018, APL-019).
- **APIs:** `os_proc_available_memory()` = current limit − footprint, 0 if not an app or over the limit, limits can change during the app life cycle, don't cache (APL-007, APL-008); jetsam reason `per-process-limit` (APL-021); Agello's limit = `phys_footprint` (`TASK_VM_INFO`) + available, nil when available is 0 (APL-012).
- **Entitlements:** increased-memory-limit, iOS 15+, only some models (APL-009); extended-virtual-addressing, iOS 14+, for large mappings (APL-010); Agello declares both (APL-011). Model sections are `mmap(MAP_PRIVATE)` + `MADV_DONTNEED` on Apple; a failed map reports `strerror(errno)` ("Cannot allocate memory" for ENOMEM) (APL-028); E4B file 3,660,726,272 bytes (APL-029).
- **Unified memory:** Metal allocations count in the footprint: M6 engine run 886.8 of 1,556.1 MB was graphics (57.0%) (APL-022); neural ledger 0 on the iPhone with an engine loaded (APL-044); shared/private storage definitions (APL-045, APL-046).
- **Agello policy using the limit:** default KV 8,192 when physical ≥ 7·10⁹ (APL-023); max context 32,768 when limit ≥ 6.5·10⁹, else 16,384 (APL-013, APL-014); stop a reply below 400,000,000 bytes available; keep the chat KV across a helper when available ≥ 28,672 × maxNumTokens + 400,000,000 (869,762,048 at 16K) (APL-015).
- **8 GB envelope (ballast) results:** effective-limit caveat for three scenarios (APL-025); 3.5 GB outcomes (APL-026); 4.0/5.0 GB outcomes (APL-027). Footprints per step are in the same reports (`steps[].footprintGB` includes the ballast).
- **Residency sets** are not requested in Agello's configuration (APL-048); Apple6+, 32 per queue, no hazard tracking (APL-049).
