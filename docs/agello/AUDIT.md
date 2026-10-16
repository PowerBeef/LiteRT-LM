# LiteRT-LM on Apple silicon: architecture audit

Date: 2026-10-14. Audit by Agello for its fork `PowerBeef/agello-litert-lm`, scope Apple platforms
only (iOS 27, macOS 27, Metal 4). Model: Gemma 4 E4B (`.litertlm`).

> **Dated audit.** Line numbers and code descriptions are LiteRT-LM v0.18.0 (`agello` branch), the
> source of what Agello ships. For the documented build (`agello-next` `7b872544`) see
> [reference/](reference/README.md); statements that changed since are listed in
> `reference/_work/contradictions.md` (C9, C13, C17, N11, N24-N26, N28, N31, N36). What it says of
> our E4B file describes the 2048-token-block revision `13d5aa4c`; Agello now uses `a1693e22`
> (512-token blocks, same weights).

## Sources and versions

| What | Version |
| --- | --- |
| LiteRT-LM | v0.18.0 (branch `agello`; the version Agello ships); upstream `main` as of 2026-10-14 |
| LiteRT | `26895c9f` (v0.18.0's `LITERT_REF`, 2026-09-25); `9b8c7f77` (upstream main's, 2026-10-13) |
| ML Drift | `google-ai-edge/ml-drift` (open source since early October 2026, forks from 2026-10-09; Apache 2.0), commit pinned by LiteRT `9b8c7f77` |
| Release binaries | `CLiteRTLM.xcframework` (iOS) and `CLiteRTLM_mac.xcframework` from the v0.18.0 GitHub release, inspected with `otool`, `nm` and `strings` |
| Model | `litert-community/gemma-4-E4B-it-litert-lm` rev `2eee7ac3` (sha256 `0b2a8980…`) and our split-RoPE copy `PowerBeef02/gemma-4-E4B-it-litert-lm-fp16-rope` rev `13d5aa4c` (sha256 `4a691f03…`) |
| Hardware | iPhone 17 Pro (A19 Pro, Apple10, 12 GB); Mac mini M6 (Apple11 as Metal reports it: 12-core GPU with Neural Accelerators, 16 GB, ~153 GB/s, 12.7 GB Metal working set) |

Line numbers below are from `git show agello:<path>` unless noted.

## 1. Summary

1. **The binaries Agello ships are not built from the open tree.** Google builds the release
   xcframeworks in its internal monorepo (`//depot/branches/odml.litert_lm_release_branch/…`). They
   statically link LiteRT core, XNNPACK and KleidiAI, ML Drift (Metal on iOS, WebGPU over Dawn on
   macOS), the TopK GPU samplers, the Gemma constraint provider, and engines absent from open
   source (`GPU_ARTISAN`, `TENSOR_LLM`).
2. **ML Drift, the GPU engine, became open source in early October 2026.** LiteRT `main` fetches it by
   commit and sha256, and its Metal backend, kernels (including `ConvAppleMPP`, which reaches the
   GPU Neural Accelerators through Metal 4 TensorOps on Apple10), and LiteRT's Metal accelerator
   are all public. v0.18.0's LiteRT pin declares `ml_drift` without a URL, so only upstream
   `main` can build the GPU path from source.
3. **Decode is bandwidth-bound and leaves headroom at long context.** One decode step reads all
   2.198 GB of weights. LiteRT's measured decode on the M6 fits a constant ~96 GB/s (63% of peak)
   from 0.7K to 30K, which only works if every layer, including the 35 sliding-window layers,
   reads the whole KV range. Windowed attention would give about 1.85× at 32K.
4. **Prefill is compute-bound and pays for the allocated context, not the filled one.** The same
   2K prompt prefills at 790 tok/s with an 8K window and 606 tok/s with a 32K one (Mac), because
   the mask and softmax span the full allocation. Matmuls already stop at the filled length.
5. **Host-side costs are large for Gemma 4 E-models:** CPU embedding and per-layer-embedding
   lookups per token (≈2,048 single-token CPU interpreter runs per 1,024-token prefill chunk), a
   full 262,144-logit GPU→CPU→GPU round trip per token whenever any constraint is active (the
   thinking budget is one), synchronous prefill and `waitUntilCompleted` per decode step on
   Metal, buffer locks that always copy through a host shadow on unified memory, JSON per
   streamed chunk, and the whole chat history rendered twice per turn.
6. **Memory:** all 24 INT8 KV caches are allocated at the full context length, including the 20
   sliding-window caches (918 MB at 32K; a 512-entry ring would need ~273 MB). The GPU weight
   cache keeps a full 2.2 GB copy of the weights per context size ever used (probe 14).
7. **#3913 (float16 rotary positions):** the loss is in the stock graph's CAST(INT32→float) → MUL
   → SIN/COS chain. ML Drift's own RoPE kernel computes angles in FP32 and is immune. A new report
   shows a second loss below position 2048 on M1-class GPUs (Apple7, shared with A14/A15), which
   the 2048-block split doesn't fix.

## 2. Architecture

### 2.1 Layers

| Layer | Where | Notes |
| --- | --- | --- |
| Swift | `swift/` | `Engine.swift:50-154` builds settings and calls `litert_lm_engine_create` (blocking, ~10 s cold). `Conversation.swift` speaks JSON: `sendMessage` blocks a thread (121-166, 286-292); `sendMessageStream` feeds an `AsyncThrowingStream` from a C callback (397-592, 819-903); automatic tool calling loops up to 25 times. `apple_fm/LiteRTLanguageModel.swift` adapts to Foundation Models and rebuilds the conversation from the whole transcript every turn. |
| C API | `c/` | `engine.cc` (settings, sessions, plain-text callbacks 57-99); `conversation.cc` serializes each streamed chunk with `message->dump()` (44-68). |
| Conversation | `runtime/conversation/` | Template rendering (`conversation.cc:229-340`), decode and constraint configuration (342-417), `SendMessageAsync` (564-844); `internal_callback_util.cc` parses channels and tool calls per chunk; per-model processors in `model_data_processor/`. |
| Core | `runtime/core/` | `engine_advanced_impl.cc` (`kAdvancedLiteRTCompiledModel`), `session_advanced.cc`, `tasks.cc` (Prefill, Decode, Score), `cached_session.cc` (prefix reuse, not exposed by the C API). |
| Framework | `runtime/framework/` | `ThreadedExecutionManager`: one execution thread and one callback thread; `ResourceManager` owns one `LlmExecutor` behind a mutex and swaps contexts between sessions (`resource_manager.cc:692-797`). |
| Executors | `runtime/executor/` | `LlmLiteRtCompiledModelExecutorStatic`/`Dynamic` (`llm_litert_compiled_model_executor.cc`), `LlmLiteRtMtpDrafter`, vision, audio, embedding executors; KV state in `litert/state.cc`. |
| Components | `runtime/components/` | Model resources and loader, sampler factory and CPU sampler, constrained decoding (llguidance, Gemma provider, penalties, thinking budget), embedding lookup, prompt templates (minijinja), stop tokens, tool-call parsers (ANTLR → Rust). |
| LiteRT | `@litert` | Environment, `CompiledModel`, `TensorBuffer`, TFLite interpreter, XNNPACK, accelerator registry, magic-number rewriting (`litert/runtime/magic_number_utils.cc`). |
| ML Drift | `@ml_drift` | GPU graph compiler, kernels, Metal/WebGPU/OpenCL backends, `LlmMetal` (artisan runner). |

### 2.2 Loading

- `.litertlm` (`schema/core/litertlm_header_schema.fbs`): a flatbuffer header, then sections
  aligned to 16 KB: TFLite models tagged by role (`TF_LITE_PREFILL_DECODE`, `EMBEDDER`,
  `PER_LAYER_EMBEDDER`, `VISION_*`, `AUDIO_*`, `MTP_DRAFTER`, `ARTISAN_TEXT_DECODER`), external
  weights, the SentencePiece or HF tokenizer, `LlmMetadataProto`, `ExecutorMetadataProto`.
- Sections are mapped lazily (`mmap` `MAP_PRIVATE`, `MADV_DONTNEED` on Apple,
  `support/util/memory_mapped_file_posix.cc:132-145`).
- Defaults (`runtime/engine/engine_settings.cc`): on Apple GPU, `maxNumTokens` defaults to the
  model maximum (not clamped to 4096 as elsewhere, 325-340); generic heuristics
  (`hint_kernel_batch_size=4`, `hint_waiting_for_completion`, 434-479); a file with an artisan
  section switches the backend to `GPU_ARTISAN` (187-211), an engine present only in Google's
  binaries.
- The tokenizer is built concurrently with the executor (`engine_advanced_impl.cc:345-393`), for files
  with an `llm_model_type` and with `parallel_file_section_loading` true (the default).
- The LiteRT environment receives only magic-number options and log severity
  (`runtime/util/litert_util.cc:130-170`): no Metal device or command queue, no library dir (CPU and GPU backends).

### 2.3 Prefill and decode

- **Signatures** (`litert_compiled_model_executor_utils.cc:83-104, 386-468`): `decode`, `verify`
  (MTP), and every `prefill*` signature, ordered by length.
- **Magic numbers** (`runtime/executor/magic_number_configs_helper.cc`): exporters write prime
  dimensions (> 10). At load, the context prime becomes `maxNumTokens` (or the largest multiple
  of 256 below the prime), prefill primes become `prefill_batch_sizes` or `floor(n/256)*256`.
  E4B's context prime is 32003, so "32K" runs at 32,000. The C API doesn't expose
  `prefill_batch_sizes` or `selected_signatures`.
- **Chunking** (`GetOptimizedPrefillWorkGroups`, 470-600): "cautious greedy" on GPU (a remainder of
  at least half the current length is padded up to it: 700 tokens → one padded `prefill_1024`).
- **Prefill** (`PrefillInternal`, 517-774): holds back the last token for the first decode step,
  pads, looks up embeddings and per-layer embeddings on the CPU, fills the attention mask and
  `param_tensor` on the CPU. **On Metal, prefill always runs synchronously** (1825-1833, TODO
  b/494284915).
- **Decode step** (`Decode` 1196-1344 → `DecodeLogits` 1361-1400): CPU embedding and PLE lookup
  for the pending token (`GetTokenToDecode` 839-930); input fill and a full mask re-init and
  refill with a copy of the whole token history (`DecodeInternal` 959-1060); new hash maps and
  duplicated handles for every buffer (`BindTensorsAndRunDecode` 1062-1123); `RunAsync("decode")`;
  constraints; `SampleLogits` (1618-1643); a new host ids buffer per step (1221-1227). The session
  loop (`tasks.cc:593-852`) then runs the stop filter, the detokenizer and the streaming callback
  inline on the execution thread.

### 2.4 Sampling

- The sampler backend defaults to the main backend. The Metal TopK sampler is used only when the
  environment carries a Metal device and no WebGPU device option is set; otherwise WebGPU, OpenCL, then the CPU with a warning
  (`sampler_factory.cc:642-781`). The GPU samplers are closed (dlopen or static pointers).
- FP16 logits only with FP16 precision and the sampler backend set to GPU (the default on a GPU
  engine, whether or not the GPU sampler loads) (executor 1982-1998).
- "Sampler handles input" (the sampler writes the next token, position and mask and launches the
  next decode, 1538-1550) needs in-graph embeddings and no local mask: Gemma 4 E-models can't use
  it.

### 2.5 KV cache

- `LitertState` (`runtime/executor/litert/state.{h,cc}`): policies `kInplace` (CPU), `kPingPong`
  (GPU without `param_tensor`: two banks, 2× memory), `kGpuOptimizedInplace` (GPU with a
  single-buffer cache).
- With a `param_tensor` input, ML Drift's `add_values_to_cache` and `runtime_bmm` kernels read
  `{start, end, end, 0, 0, 0, 0}` (`litert_compiled_model_executor_utils.cc:641-660`; the comment
  says local-attention optimization is enabled for WebGPU only).
- Rewinds and checkpoints reset the step counter without copying KV (`session_advanced.cc:494-558`).
- **Clones and session switches deep-copy the KV, into host memory on Metal**
  (`state.cc:255-289`, `resource_manager.cc:733-766`, `runtime/util/tensor_buffer_util.cc:54-62`,
  a workaround for leak b/505373949).

### 2.6 Speculative decoding (MTP)

- The drafter section loads lazily (1164-1189); `verify` checks the drafts; drafts per step =
  `verify.input_pos` length − 1 (3 for E4B).
- Each draft step: CPU embedding lookup; a lock and memcpy concatenating the embedding with the
  projected activations (GPU readback and re-upload, `llm_litert_mtp_drafter.cc:122-173`); run the
  drafter; sample; read the id back (577-646). Then one verify run, masked by constraints.

### 2.7 Multimodal and embeddings

- Vision: encoder on GPU or CPU, adapter forced to CPU; outputs go to host memory and are copied
  into prefill embeddings. Audio: CPU conformer (miniaudio and pffft front end).
- EmbeddingGemma: `runtime/core/embedding_engine_impl.cc`.

### 2.8 Threads

- One execution thread and one callback thread; per-token callbacks run on the execution thread;
  only the final callback goes to the callback pool (`threaded_execution_manager.cc`).
- No QoS on any thread (`runtime/framework/worker_thread_pthread.cc:76-120`, Linux-only nice and
  affinity).

## 3. Open source and binaries

### 3.1 Prebuilt files in the repository (`prebuilt/`, Git LFS)

| Library | iOS / macOS size | What it is | Source |
| --- | --- | --- | --- |
| `libLiteRt.dylib` | 8.2 / 11.5 MB | LiteRT C API | Open |
| `libLiteRtMetalAccelerator.dylib` | 12.7 / 16.5 MB | "GPU Metal" accelerator | Open since early October 2026 (ML Drift) |
| `libLiteRtWebGpuAccelerator.dylib`, `libwebgpu_dawn.dylib` | – / 16.7, 15.3 MB | ML Drift WebGPU, Dawn | ML Drift open; Dawn open |
| `libLiteRtTopKMetalSampler.dylib` | 6.8 / 9.4 MB | GPU TopK sampler | **Closed** |
| `libGemmaModelConstraintProvider.dylib` | 21.1 / 9.4 MB | Gemma function-call constraints | **Closed** |

### 3.2 Release xcframeworks

- iOS: device slice 61.6 MB (min iOS 15, SDK 26.4), statically linked, exports `GpuMetalAccelerator`
  and a Metal TopK sampler. Also carries dead weight on Apple: OpenGL ES, Dawn's Metal backend,
  OpenCL sampler code, NPU code.
- macOS: universal 137.8 MB, `GpuWebGpuAccelerator` (Dawn on Metal), not the Metal accelerator.
- Neither has Core ML, MPSGraph or `MTLBinaryArchive`; shaders compile from MSL source at runtime.

### 3.3 Static linking hooks

- LiteRT-LM: `default_static_gpu_accelerator` (`runtime/executor/BUILD`) and
  `default_static_gpu_samplers` (`runtime/components/BUILD`) are empty in open source.
- LiteRT `9b8c7f77`: `//litert/runtime/accelerators/gpu:ml_drift_metal_accelerator` registers
  itself when linked statically. LiteRT-LM must declare `ml_drift` in its own WORKSPACE.

### 3.4 Constrained decoding without the closed provider

With `LITERT_LM_FST_CONSTRAINTS_DISABLED`, `Gemma4DataProcessor::Create` returns
`FailedPrecondition` whenever constrained decoding is requested
(`runtime/conversation/model_data_processor/gemma4_data_processor.cc:62-70`). Agello's JSON helpers
request it, so a source build needs a patch that keeps llguidance working without the provider.
(v0.18.0; fixed in `agello-next` by `d1488ce7`.)

## 4. Build

- Bazel 7.6.1 (`.bazelversion`), WORKSPACE mode, C++20; rules_apple 3.22.0, rules_swift 2.9.0,
  apple_support 1.23.1, rules_rust 0.61 (llguidance 1.3.0, minijinja 2.14.0, tokenizers 0.21.0,
  antlr4rust, cxx). Configs `ios_arm64`, `ios_sim_arm64`, `macos_arm64` in `.bazelrc`.
- Open targets: `//swift:CLiteRTLM` (`apple_xcframework`, iOS device and simulator) and
  `//swift:CLiteRTLM_dylib` → `:CLiteRTLM_mac` (a zip genrule, not a real xcframework).
- CI (`.github/workflows/ci-build-mac.yml`) builds only `//runtime/engine:litert_lm_main` on macOS in
  dynamic link mode; the iOS step is commented out. No one upstream builds the Apple frameworks in
  the open.
- The CMake build targets Linux and Android only and stubs out the GPU.

## 5. The Gemma 4 E4B model

### 5.1 Sections (`gemma-4-E4B-it-fp16-rope.litertlm`, header v1.5.0)

| # | Section | Bytes | Contents |
| --- | --- | --- | --- |
| 0 | LlmMetadataProto | 12 KB | Gemma4 type, delimiters, tool tokens, Jinja template, `thought` channel; no `max_num_tokens` |
| 1 | SP_Tokenizer | 4.7 MB | SentencePiece |
| 2 | `tf_lite_embedder` | 171 MB | INT2 [262144, 2560] table with per-row scales |
| 3 | `tf_lite_per_layer_embedder` | 837 MB | 42 INT2 lookups → [1,1,42,256] |
| 4–6 | audio encoder (CPU), adapter, end-of-audio | 110 MB | Conformer, INT2/INT4 |
| 7–9 | vision encoder (FP16 activations), adapter, end-of-vision | 232 MB | ViT, INT8; 70/140/280 tokens |
| 10 | `tf_lite_prefill_decode` | 2,260 MB | 62.5 MB flatbuffer + 2,197.9 MB external weights; 4 signatures |
| 11 | `tf_lite_mtp_drafter` | 45 MB | 4 blocks, width 256, INT4 head [262144, 256] |

### 5.2 Main section

- Signatures: `decode` (2,749 ops), `prefill_1024` and `prefill_128` (1,810 ops each), `verify`
  (2,838 ops). Inputs: `embeddings` FP32, `per_layer_embeddings` FP32 [1,N,42,256], `input_pos`
  INT32, `mask` BOOL [1,1,N,32003], `param_tensor` INT32 [1,1,1,7]; outputs: `logits` FP32
  [1,N,262144] (decode, verify), `activations` FP32 (for MTP), and 24 K + 24 V caches.
- Weights: W4A8 with static per-tensor activation scales (QUANTIZE → INT8×INT4 FC → DEQUANTIZE),
  per-channel, no blockwise quantization. INT4 matmuls 1,945 MB; INT8 PLE projections 83 MB; INT2
  LM head 168 MB (a separate copy of the tied embedding).
- Architecture: 42 layers, hidden 2560, GQA 8 query / 2 KV heads, head dim 256 (local) and 512
  (global), 5 local layers per global (global layers 5, 11, 17, 23, 29, 35, 41), window 512, RoPE θ
  1e4 local and 1e6 global with p = 0.25, KV sharing (layers 24–41 reuse caches 22 and 23),
  per-layer embeddings (256 per layer per token), sandwich norms, QK and V norms, logit softcap 30.
- KV: 24 INT8 caches with static scales, V stored transposed, all sized to the full context
  (28,672 B per token): 117 / 235 / 470 / 918 MB at 4K / 8K / 16K / 32K.
- Composites: `odml.rms_norm`, `odml.runtime_bmm` {`is_global`, `is_src`, `scale`} for QKᵀ and PV
  (no window attribute; SOFTMAX over the full row), `odml.cache_update` (QUANTIZE +
  DYNAMIC_UPDATE_SLICE; supports `is_ring_buffer`).
- Masks: local layers build their 512-token window in the graph from `input_pos` and ignore the
  `mask` input, which only the 7 global layers read, so results are exact; the cost is that
  QKᵀ, the select and the softmax still cover every filled key. Details, the per-layer cache map
  and the drafter: `E4B-GRAPH.md`.
- Prefill runs layers 0–22 and the key/value projection of layer 23 (the KV producers), and skips
  the shared layers and the LM head; the first token comes from `decode`.

## 6. Performance

### 6.1 Measured

| Device, path | Prefill (tok/s) | Decode (tok/s) | Notes |
| --- | --- | --- | --- |
| iPhone 17 Pro, Metal, FP16, T=32768 | 512 / 385 / 288 at 2K / 8K / 30K prompts | 22.2 / 16.1 / 11.2 (no MTP); 38.6 / 40.5 / 34.7 with MTP | probe 12 |
| iPhone 17 Pro, Agello calibrate (MTP) | TTFT 2.1 / 6.0 / 18.9 s at 2.3K / 4.4K / 8.6K inputs | 43 / 38 / 34 / 27 at 4K / 8K / 16K / 32K windows | footprint ~75 KB per token of capacity |
| Mac M6, WebGPU, T=32768 | 443–606 | 41.2 (0.7K) → 24.7 (30K) | ~96 GB/s effective |
| Mac M6, WebGPU, T=8192 | 777–805 for a ~2K prompt | – | same prompt: 606 at T=32768 |
| Mac M6, WebGPU, MTP | – | 111–124 on copy tasks | |
| Mac M6, Metal (Agello as an iPhone app on the Mac, iOS v0.18.0 binary) | 3,211 | 49.8 | Agello's benchmark: 1,024 prefill / 256 decode, T=1,312, no MTP. The macOS WebGPU binary gives 570–800 prefill and 44–47 decode on 0.7K–4K prompts (agello-bench, T=8,192) |

### 6.2 Ceilings on the M6 (practical bandwidth 122 GB/s)

| Context | LiteRT-like (all layers read full KV) | Window-aware |
| --- | --- | --- |
| 0.7K | 53 tok/s | 55 tok/s |
| 8K | 45 | 52 |
| 32K | 31 | 46 |

Prefill compute ceilings at 2K: ~4,600 tok/s on the Neural Accelerators at FP16, ~9,200 at INT8,
~2,200 on SIMD FP16 (estimates). Realistic prefill headroom over today: 1.7–3.5×.

## 7. Inefficiencies in the open code

| Area | Evidence | Cost |
| --- | --- | --- |
| Embedding lookups | `embedding_lookup_text.cc:59-91, 160-258`: one CPU interpreter run per token, twice (embedder and PLE) | ≈2,048 CPU runs per 1,024-token prefill chunk; one pair per decoded token |
| Constraint round trip | `constrained_decoder.cc:31-73`; `thinking_budget_constraint.cc:119-137` returns an all-allowed mask, not null; `tasks.cc:645-656` | 512 KB–1 MB copied each way per token whenever thinking or any constraint is on |
| Synchronous Metal | executor 1825-1833 (prefill); `gpu_backend_metal.mm` `waitUntilCompleted`; Metal events unimplemented | No CPU/GPU overlap |
| Host shadows | `buffer_handler_metal.mm`: `Lock` always allocates a host buffer and copies, even for shared storage | Unified memory never used zero-copy |
| Per-step allocation | executor 959-1123, 1196-1344, 1655-1664; `GetCopyOfTokens()` per step; full mask re-init | O(layers + context) work per token |
| Per-turn work | `tasks.cc:640` builds 262,144 strings to get the vocabulary size (fixed in `agello-next` `5e946049`); `conversation.cc:257-340` renders the full history twice; `minijinja_template.rs:220-232` rebuilds the environment per apply | Grows with chat length |
| Streaming | `c/conversation.cc:62` JSON per chunk; `swift/Conversation.swift:839-855, 704-709` parses it twice | Per token, on the decode thread |
| KV copies | `tensor_buffer_util.cc:54-62` | Full KV to host memory on clones and session switches |
| Cache keys | `llm_executor_settings_utils.cc:125-156`: `<basename>_<mtime>_<size>` | Ignores precision, context size, build, OS and GPU family; 2.2 GB weight copy per context size |
| Threads | `worker_thread_pthread.cc` | No QoS |

## 8. Metal 4 and Apple silicon

| Feature | Availability | ML Drift today | Opportunity |
| --- | --- | --- | --- |
| MTL4 queues, allocators, argument tables, barriers | Apple7+, OS 26 | Classic queues; re-encodes every node each dispatch; new command buffer per 256 kernels | Lower encode cost; concurrent dispatch |
| Residency sets | Apple6+ | Supported, off by default (`gpu_enable_metal_residency_set`) | A/B with no code change |
| Indirect command buffers | Metal 3/4 | Coded but hard-disabled (`inference_context.mm`) | Replay the decode graph |
| Next-step pre-encoding | – | WebGPU only (`delegate_options.h`) | Port to Metal |
| TensorOps (MPP `matmul2d`) | All Apple GPUs; Neural Accelerators on Apple10 and later (A19, M5; the M6 reports Apple11) | `ConvAppleMPP` on Apple10 + MSL 4, int4 dequantized to threadgroup memory | Native int4/int8 inputs (26.4), cooperative-tensor dequant (26.3), Morton order, tuned tiles |
| TensorOps FlashAttention | OS 27 (matmul → matmul cooperative chaining) | – | Long-context prefill |
| FP8, FP4, INT2, MX formats | OS 27 | – | Future model variants |
| Binary archives, `MTL4Archive` | iOS 14+ / OS 26 | Runtime `newLibraryWithSource` | Cold start |
| Shared events | – | Unimplemented on Metal | Overlap sampling, MTP and the next encode |
| Persistent kernels | OS 27 (names only) | – | Experimental |

A17 Pro and A18 (Apple9) have no Neural Accelerators; their path stays on simdgroup matrices.

## 9. #3913

- Cause: `input_pos` (INT32) → RESHAPE → CAST (float) → MUL (inverse frequencies) → SIN/COS. In
  FP16, positions above 2048 aren't exact, so neighbouring tokens share rotary phases.
- ML Drift's RoPE kernel (`ml_drift/common/kernels/rope.cc`) computes angles in FP32 from INT32
  positions; the `odml.rope` composite supports partial rotary. Only the stock graph's chain loses
  precision.
- Our model fix (split `2048·h + l`) is exact on the iPhone 17 Pro and the M6. A report on an M1
  Max (Apple7) shows a second loss below 2048 (MUL output or SIN/COS argument above ~1650 rad);
  a 512-block split fixes it. Robust fixes: compute the angles in FP32 in the delegate (a fusion
  pass), or gather sin/cos by INT32 index in the graph.

## 10. Where this leads

The re-engineering plan (tracks T1–T13) follows the measured costs above: the #3913 root fix,
host quick wins, conversation snapshots across helpers, bounded and windowed attention,
unified-memory zero-copy and events, encoding, an open GPU sampler, batched embeddings, Neural
Accelerator prefill, ring-buffer local KV, cache deduplication, MTP on the GPU, and GPU encoders.
Each lands with before and after numbers in `docs/agello/tracks/`.
