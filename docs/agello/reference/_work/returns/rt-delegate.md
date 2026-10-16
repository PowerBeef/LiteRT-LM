# Returns: rt-delegate (area A4), page litert/ml-drift-delegate.md

## Settings rows (for configuration.md)

| Name | Layer | Default | Where read | Effect on Apple | Version added | Agello sets it? | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ExperimentalFlags.enableBenchmark` → `litert_lm_engine_settings_enable_benchmark` → `AdvancedSettings.is_benchmark` | Swift / C / `EngineSettings` | false | [`swift/Engine.swift:151-155`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Engine.swift#L151-L155), [`runtime/engine/engine_settings.cc:512-515`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L512-L515), [`runtime/executor/llm_executor_settings_utils.cc:220-235`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_executor_settings_utils.cc#L220-L235) | GPU waits become `kActive` (busy-spin a CPU core per wait) | before v0.18.0 (present in v0.18.0 source) | yes, true at every load (`ChatEngine.swift` 141–142, Agello `93122b6`) | Outside benchmark mode agello-next requests `kPassive` (ours `645aaf65`); upstream left the delegate default `kActive`. RT-DLG-067 |
| `gpu_enable_metal_residency_set` | `AdvancedSettings`; C `litert_lm_engine_settings_set_gpu_enable_metal_residency_set`; Swift `ExperimentalFlags.gpuEnableMetalResidencySet`; runtime `litert_lm_experimental_engine_update_gpu_enable_metal_residency_set` | false | [`runtime/executor/llm_executor_settings_utils.cc:91-103`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_executor_settings_utils.cc#L91-L103), [`runtime/executor/llm_litert_compiled_model_executor.cc:1874-1882`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_litert_compiled_model_executor.cc#L1874-L1882), [`runtime/executor/llm_executor_settings.h:208-217`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/llm_executor_settings.h#L208-L217) | When true: residency set attached to the queue, requested, re-requested every 100 ms by a heartbeat thread (macOS 15 / iOS 18+) | unknown | no | Passed both at compile time and in every run's GPU options; the delegate re-reads it per `Invoke`. RT-DLG-069 |
| `cache_dir` (`":nocache"` disables) | `EngineSettings` / Swift `EngineConfig.cacheDir` | empty: caches next to the model | [`runtime/executor/executor_settings_base.cc:384-430`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/executor_settings_base.cc#L384-L430), [`runtime/executor/litert_compiled_model_executor_utils.cc:1073-1131`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/litert_compiled_model_executor_utils.cc#L1073-L1131) | Program cache and weight cache files `<model>_<mtime>_<size>_mldrift_{program,weight}_cache.bin` | — | yes (`EngineCache` folder per build/size; `:nocache` for the benchmark engine) | RT-DLG-044, RT-DLG-062 |
| `cache_compiled_shaders_only` | `AdvancedSettings` | false | [`runtime/executor/litert_compiled_model_executor_utils.cc:1073-1131`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/litert_compiled_model_executor_utils.cc#L1073-L1131), [`ml_drift_delegate/delegate/delegate_kernel.cc:767-788`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/ml_drift_delegate/delegate/delegate_kernel.cc#L767-L788) | On Metal: no program cache at all (only OpenCL/WebGPU read the option) | — | no | RT-DLG-084 |
| `share_constant_tensors` | `AdvancedSettings` | true | [`runtime/executor/llm_executor_settings_utils.cc:204-219`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_executor_settings_utils.cc#L204-L219), [`runtime/executor/llm_executor_settings.h:252-260`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/llm_executor_settings.h#L252-L260) | Weights shared across subgraphs; required for the weight cache path | — | no | |
| `convert_weights_on_gpu` | `AdvancedSettings` | true | [`runtime/executor/llm_executor_settings_utils.cc:204-219`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_executor_settings_utils.cc#L204-L219), [`runtime/executor/llm_executor_settings.h:235-238`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/llm_executor_settings.h#L235-L238) | Batched GPU weight preparation; on Apple, cache-first reload from the mapped weight cache | — | no | Forced false for FunctionGemma models (`engine_settings.cc`). RT-DLG-057 |
| `gpu_madvise_original_shared_tensors` | `AdvancedSettings` | true | [`runtime/executor/llm_executor_settings_utils.cc:204-219`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_executor_settings_utils.cc#L204-L219), [`runtime/executor/llm_executor_settings.h:208-217`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/llm_executor_settings.h#L208-L217) | `madvise` on original weight pages after upload | — | no | |
| `allow_src_quantized_fc_conv_ops` | `AdvancedSettings` (optional) | unset = on | [`runtime/executor/llm_executor_settings_utils.cc:204-219`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_executor_settings_utils.cc#L204-L219) | When false: `ModelHints::kDisallow8bitConvs` (no INT8-activation convolutions) | — | no | Part of both cache keys |
| `hint_waiting_for_completion` | `AdvancedSettings` (optional) | unset = false | [`runtime/executor/llm_executor_settings_utils.cc:204-219`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_executor_settings_utils.cc#L204-L219) | None on Metal (OpenCL only) | — | no | |
| `hint_kernel_batch_size` | `AdvancedSettings` (optional) | unset (4 for generic models) | [`runtime/executor/llm_executor_settings_utils.cc:204-219`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_executor_settings_utils.cc#L204-L219) | None on Metal: `kernel_batch_size` isn't read by the Metal delegate files | — | no | Same for `num_steps_of_command_buffer_preparations`, `gpu_context_low_priority`, `optimize_shader_compilation`, `num_threads_to_upload/compile` |
| `GpuConfig.external_tensor_mode` / CLI `--gpu_external_tensor_mode` | executor settings | false | [`runtime/executor/llm_executor_settings.h:119`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/llm_executor_settings.h#L119), [`runtime/executor/llm_executor_settings_utils.cc:166-202`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_executor_settings_utils.cc#L166-L202) | false: no-external-tensors mode (converters; the run waits in `PostConvert`); true: all I/O bound directly, asynchronous runs would not wait | — | no | RT-DLG-018, RT-DLG-085 |
| Activation data type / mixed precision | `EngineSettings` / executor | model-dependent | [`runtime/executor/litert_compiled_model_executor_utils.cc:1009-1041`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/litert_compiled_model_executor_utils.cc#L1009-L1041) | FP32 → `kFp32`, else `kFp16` → calculation precision `kF16` | — | no (nil) | RT-DLG-005 |
| GPU option wait type (`SyncExecutionModeWaitType`) | LiteRT GPU option | Default (delegate keeps `kActive`) | [`litert/runtime/accelerators/gpu/ml_drift_delegate_create.cc:258-276`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/accelerators/gpu/ml_drift_delegate_create.cc#L258-L276) | Active / Passive / DoNotWait in `WaitForCompletion` | — | indirectly (benchmark mode) | RT-DLG-065 |
| GPU option `use_metal_argument_buffers` | LiteRT GPU option (`__APPLE__`) | false; LiteRT-LM sets true | [`litert/runtime/accelerators/gpu/ml_drift_delegate_create.cc:199-202`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/accelerators/gpu/ml_drift_delegate_create.cc#L199-L202), [`runtime/executor/litert_compiled_model_executor_utils.cc:1009-1041`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/litert_compiled_model_executor_utils.cc#L1009-L1041) | Argument buffers (dropped on Tier 1 devices) | — | no | Part of the program cache key |
| GPU option `use_ir_model`; Bazel `--define=litert_gpu_use_ir_model=true|false` | LiteRT GPU option / Bazel | unset = false (GraphFloat32) | [`litert/c/options/litert_gpu_options.cc:1267-1283`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/c/options/litert_gpu_options.cc#L1267-L1283), [`litert/c/options/BUILD:232-252`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/c/options/BUILD#L232-L252) | IrModel pipeline; drops Agello's bounded attention | — | no | RT-DLG-015, RT-DLG-031 |
| `enable_qkv_norm_rope_fusion`, `enable_short_conv_step_fusion` | delegate option only | false | [`ml_drift_delegate/delegate/delegate_options.h:323-336`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/ml_drift_delegate/delegate/delegate_options.h#L323-L336) | Not reachable from LiteRT GPU options at the pin | — | no | RT-DLG-029 |
| `AGELLO_BOUNDED_ATTENTION` | environment variable (ours) | on (off if value starts with `0`) | [`ml_drift_delegate/delegate/composite/bounded_attention.cc:293-307`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/ml_drift_delegate/delegate/composite/bounded_attention.cc#L293-L307) | Bounded attention pass; program cache key marker | LiteRT `ede28c64` | no (A/B only) | Read once per process. RT-DLG-038 |
| `AGELLO_BOUNDED_WINDOW` | environment variable (ours) | on | [`ml_drift_delegate/delegate/composite/bounded_attention.cc:293-307`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/ml_drift_delegate/delegate/composite/bounded_attention.cc#L293-L307) | Sliding window for local layers | LiteRT `31386317` | no | `-nowindow` key marker; log line still names a window when off (RT-DLG-039) |
| `AGELLO_SHARED_WEIGHT_CACHE` | environment variable (ours) | on (off only for `0`) | [`ml_drift_delegate/delegate/delegate_kernel.cc:110-125`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/ml_drift_delegate/delegate/delegate_kernel.cc#L110-L125) | Content weight cache key; `0` restores upstream's per-size key | LiteRT `97e06d36` | no | RT-DLG-055 |
| `AGELLO_SIMULATE_APPLE_GPU` | environment variable (ours) | unset | [`ml_drift_delegate/delegate/delegate_kernel.cc:110-125`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/ml_drift_delegate/delegate/delegate_kernel.cc#L110-L125) | Joins both cache keys (the simulation itself is ML Drift's `gpu_info.cc`) | LiteRT `36275983`, ML Drift `ef08a67` | test harness only | |
| `AGELLO_VERIFY_DIRECT_COPIES` | environment variable (ours) | off (`1` enables) | [`ml_drift_delegate/delegate/buffer_handler_metal.mm:253-261`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/ml_drift_delegate/delegate/buffer_handler_metal.mm#L253-L261) | Re-runs ML Drift's conversion after each direct copy; error anomaly on mismatch | LiteRT `584f3461` | test harness only (`TEST_RUNNER_AGELLO_VERIFY_DIRECT_COPIES`) | RT-DLG-078 |
| `LITERT_GPU_FORCE_COMPLETION` | compile-time define | off | [`ml_drift_delegate/delegate/delegate_kernel.cc:104-108`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/ml_drift_delegate/delegate/delegate_kernel.cc#L104-L108) | Wait after every dispatch (tracing) | — | no | |

## Terms (for glossary.md)

| Term | Meaning | Prefer / avoid |
| --- | --- | --- |
| ML Drift delegate | The TFLite delegate in LiteRT's `ml_drift_delegate/` that runs graphs on ML Drift | Prefer "ML Drift delegate"; avoid "Metal delegate" for the whole thing (Metal is one backend) |
| GPU Metal accelerator | LiteRT's accelerator plug-in (`ml_drift_metal_accelerator.cc`) that creates the delegate on Apple and registers Metal buffer handlers | Prefer the code's name `GPU Metal` |
| Partition | Connected supported nodes replaced by one delegate kernel (`LITERT_METAL`); at most one per subgraph | |
| Composite (`odml.*`) | A StableHLO composite op the delegate lowers itself instead of its decomposition subgraph | Avoid "custom op" (TFLite custom ops are different) |
| Custom transformation | A `GraphFloat32` rewrite applied by the delegate after ML Drift's own | |
| Bounded attention | Our pass that bounds the attention select and softmax to the filled KV length (`agello_bounded_mask`, `agello_bounded_softmax_reduce`) | Avoid "flash attention" |
| Sliding window (W) | Keys a local layer attends to, `(p − W, p]`; the delegate reads W from the graph's mask constants | Distinguish from the context window |
| External tensor | A delegate I/O tensor bound directly to a LiteRT buffer, no conversion | |
| No-external-tensors mode | LiteRT-LM's default: only pattern-matched tensors are external, others are converted each run | |
| `param_tensor` | `INT32[1, 1, 1, 7]` runtime parameters `{start, end, end, 0...}` of the Gemma graphs | |
| Program cache | `..._mldrift_program_cache.bin`: serialized inference contexts | Avoid "shader cache" (Metal shaders are still compiled at load) |
| Weight cache | `..._mldrift_weight_cache.bin`: GPU-format weights | Avoid "model cache" |
| Model token | The caches' namespace: `<model file name>_<mtime>_<size>` | |
| Content key | Our weight cache subgraph key, from delegated nodes and constant inputs instead of tensor sizes | |
| Active wait / passive wait | Busy-spinning on a completion flag vs `waitUntilCompleted` | Prefer "busy-wait" for active in user-facing text |
| Direct copy | Our `memcpy` path between a shared linear Metal buffer and its host shadow | |
| Host shadow | The aligned CPU copy a Metal buffer lock returns | |
| Residency set | `MTLResidencySet` the backend fills; only requested when enabled | |

## Contradictions

- **C12** (window-aware kernels: FC, softmax, ConvAppleMPP per T4 vs also ConvWaveMatrix per ML Drift
  `15fa9e2`): the delegate passes the window only through `ConvRuntimeCheckDesc` (both
  `runtime_bmm` products) and `SoftmaxRuntimeCheckDesc` (the reduction) and doesn't pick kernels; at
  ML Drift `15fa9e2` FC, softmax/softmax1x1, ConvAppleMPP and ConvWaveMatrix all read it, so T4's list
  describes `2184a7f`, and kernels without it stay correct because keys below the window are masked.
  Kernel detail is md-kernels'. Claims: RT-DLG-037 (with RT-DLG-034, RT-DLG-035).
- **NEW: weight cache header version.** Google's `serialization_weight_cache/GEMINI.md` says
  `kVersion` was incremented to 3; `cache_builder.h` at LiteRT `362759838` defines `kVersion = 2`.
  The code wins; the note is stale or describes an unlanded change. Claim: RT-DLG-054.

## Notes for other pages

- **apple-platform.md / Agello side:** Agello sets `ExperimentalFlags.enableBenchmark = true` at every
  load, so its GPU waits busy-spin a CPU core in our build too; our passive-wait commit `645aaf65`
  only applies outside benchmark mode. Worth a decision on the Agello side (statistics without
  benchmark mode). RT-DLG-067, RT-DLG-065.
- **apple-platform.md:** residency sets are created and filled on iOS 18 / macOS 15+ but never
  requested in Agello (`gpu_enable_metal_residency_set` unset). RT-DLG-069.
- **memory.md:** on Apple the GPU weights are file-backed pages of the weight cache
  (`newBufferWithBytesNoCopy` over the mapped file), after a first load that writes the cache,
  flushes it and reloads every tensor from it. RT-DLG-057, RT-DLG-058, RT-DLG-059. The KV caches are
  external LiteRT buffers written in place by `odml.cache_update`, one byte per element even for
  sub-byte cache types. RT-DLG-023.
- **memory.md / troubleshooting.md:** program cache files only grow (re-inserted keys append, the old
  blob isn't reclaimed) and hold one entry per context size; weight caches with the upstream key hold
  one copy per size. RT-DLG-047, RT-DLG-050, RT-DLG-056.
- **troubleshooting.md:** "Some ops are not accelerated. Add kLiteRtHwAcceleratorCpu ..." means the
  delegate didn't take a node (no CPU fallback with GPU-only); look for the delegate's "Following
  operations are not supported by GPU delegate" log. RT-DLG-011, RT-DLG-010.
- **troubleshooting.md / maintenance.md:** program cache keys have no GPU or build identity
  (RT-DLG-048); weight cache identity covers only ML Drift's FC/weights sources (RT-DLG-053). Agello's
  per-build folders are the guard.
- **ml-drift/metal-backend.md:** the delegate calls `EncodeWithCommandQueue(queue, 256)`; per run
  `1 + ⌈nodes/256⌉ + 1` command buffers in no-external-tensors mode, which reconciles the engine run's
  108 (RT-DLG-064). Argument buffers arrive as a `CreateGpuModelInfo` hint.
- **ml-drift/kernels.md:** the window start is `max(0, params[0] − (W−1))` rounded down to 32 keys
  (RT-DLG-035); `allow_src_quantized_fc_conv_ops` (on by default in LiteRT-LM) is what permits the
  `quantize_float16_to_int8c4` + `convolution_int8(conv_apple_mpp)` prefill path on Apple10.
- **litert-lm/executors.md:** `param_tensor` = `{start, start+n, start+n, 0, 0, 0, 0}`
  (RT-DLG-024); `logits` is external whenever the sampler backend resolves to GPU, even when the GPU
  sampler then falls back to the CPU (RT-DLG-019); LiteRT-LM never calls `LiteRtCopyKvCacheMetal`
  (RT-DLG-080).
- **litert/runtime.md:** `CompiledModel` applies accelerators in registration order and skips
  JIT-responsible ones whose hardware wasn't requested; XNNPACK is JIT-responsible outside the web
  (RT-DLG-011). LiteRT's magic-number rewrite also updates `odml.cache_update` attributes (RT-DLG-025).
- **tools.md / TELEMETRY:** engine-run.md's counters are reset when the case starts, so load-time
  counters (cache hits, delegation coverage) read 0 there; a load-telemetry capture would be needed
  to show them in generated evidence.

## Open questions

- Restoring a program cache entry written on a different Apple GPU family: does it fail or run the
  other family's kernels? The key would match; restore checks only tensor ids. Needs a device test.
- No generated evidence at the pin for load-time delegation coverage and cache hits/misses
  (engine-run counters reset after load).
- Which LiteRT commit Google's prebuilt v0.18.0 Metal accelerator was built from, and how its
  dispatch and waits behaved, can't be read from source.
- The CPU cost of busy-waiting in Agello's configuration on the iPhone hasn't been measured with a
  results file at this pin (TELEMETRY.md has a Mac reading, a lead).
