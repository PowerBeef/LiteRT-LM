# Other platforms

> **Scope:** what LiteRT-LM, LiteRT and ML Drift do off Apple's Metal path (Android, NPUs, WebGPU,
> Linux, Windows, Python, JavaScript, omni), mapped only far enough to read shared code and upstream
> reports. **Out of scope:** the Apple path, which every other page covers in depth: start at
> [stack.md](stack.md) for the layers and [apple-platform.md](apple-platform.md) for iOS and macOS.
> **Pinned to:** LiteRT-LM `7b872544`, LiteRT `362759838`, ML Drift `15fa9e2`.
> **Applies to:** non-Apple platforms. **Last verified:** 2026-10-15

## Summary

- Off Apple, LiteRT's generic GPU accelerator picks OpenCL or WebGPU at run time
  <!-- c:RT-OTH-003 -->, and LiteRT reaches NPUs through a separate dispatch API. <!-- c:RT-OTH-006 -->
  LiteRT-LM's NPU LLM executor is compiled out on iOS but not on macOS. <!-- c:LM-OTH-001 -->
- Google's v0.18.0 macOS binary (`CLiteRTLM_mac`) carries the WebGPU accelerator, ML Drift through
  Dawn, and no "GPU Metal". <!-- c:RT-OTH-005 --> Our build links the Metal accelerator on iOS and
  macOS ([litert-lm/build-and-release.md](litert-lm/build-and-release.md#on-apple)).
- ML Drift's README lists five GPU APIs <!-- c:MD-OTH-001 -->; the open tree has four backends
  (OpenCL, OpenGL, Metal, WebGPU) and no Vulkan runtime. <!-- c:MD-OTH-002 -->
- Omni (speech recognition, text-to-speech, text-to-image) is a C++ surface that neither the C API
  nor the Swift wrapper exposes ([Omni](#omni-asr-tts-text-to-image)).
- The page has three parts, one per repository: [LiteRT-LM](#litert-lm), [LiteRT](#litert),
  [ML Drift](#ml-drift).

## LiteRT-LM

### Summary

LiteRT-LM is one C++ runtime (`runtime/`) behind several front ends. Swift and Python talk to it only
through the C API (`c/`); Kotlin's JNI layer uses the C API and the C++ headers directly; the web SDK
is a WebAssembly build with its own TypeScript API. GPU support outside Apple goes through
accelerators LiteRT loads at run time (OpenCL and WebGPU, from `prebuilt/`); the NPU LLM executor
exists only off iOS. Omni (speech and image generation) is a separate C++ surface no Apple API exposes.

### Platforms

| Platform | Front end | GPU / NPU path | Build notes | What it shares with Agello's build |
| --- | --- | --- | --- | --- |
| Android | Kotlin (`com.google.ai.edge.litertlm`), JNI library `litertlm_jni` | GPU: if no accelerator was registered from the handle option or linked statically, LiteRT tries to load `libLiteRtGpuAccelerator.so`, then (with OpenCL support) `libLiteRtClGlAccelerator.so` and `libLiteRtOpenClAccelerator.so`, then (with WebGPU support) `libLiteRtWebGpuAccelerator.so`, then (with Vulkan support) `libLiteRtVulkanAccelerator.so`, stopping at the first that registers ([`litert/runtime/accelerators/gpu_registry.cc:47-85`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/accelerators/gpu_registry.cc#L47-L85), [`litert/runtime/accelerators/gpu_registry.cc:111-128`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/accelerators/gpu_registry.cc#L111-L128)); NPU: `Backend.NPU(nativeLibraryDir)`; Google Tensor: `Backend.GOOGLE_TENSOR` | `build:android` links everything statically and re-enables OpenCL ([`.bazelrc:145-151`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/.bazelrc#L145-L151)) | The runtime, executors, conversation layer |
| Linux | C++ CLI (`litert_lm_main`), Python wheels | WebGPU accelerator over Dawn, from `prebuilt/linux_*` | OpenCL compiled out on Linux and Windows ([`.bazelrc:43-45`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/.bazelrc#L43-L45)); `linux_arm64` enables YNNPACK ([`.bazelrc:267-268`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/.bazelrc#L267-L268)) | Everything but the GPU backend |
| Windows | C++ CLI, Python wheel | WebGPU accelerator over Dawn (`prebuilt/windows_x86_64`) | Monolithic build ([`.bazelrc:345-346`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/.bazelrc#L345-L346)) | Same |
| macOS (Google's binaries, Python) | Swift (`CLiteRTLM_mac`), Python wheel | Python: the wheel's from-source build copies the WebGPU accelerator and sampler from `prebuilt/macos_arm64`, not its Metal accelerator ([Python](#python)) | Upstream builds the Mac binary as a dylib (`//swift:CLiteRTLM_mac`) | Our Mac slice uses Metal instead ([litert-lm/build-and-release.md](litert-lm/build-and-release.md)) |
| Web | `@litert-lm/core` (TypeScript), WebAssembly | WebGPU in the browser | npm package with its own version, 0.12.1 at the pin | The C++ runtime compiled to WebAssembly |

#### Android: Kotlin and NPU

- The Kotlin `Backend` mirrors C++'s `litert::lm::Backend`: `CPU(threadCount)`, `GPU`,
  `NPU(nativeLibraryDir)` (the folder holding the vendor NPU libraries, such as the app's
  `nativeLibraryDir`), and `GOOGLE_TENSOR` ([`kotlin/java/com/google/ai/edge/litertlm/Config.kt:137-161`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/kotlin/java/com/google/ai/edge/litertlm/Config.kt#L137-L161)).
  <!-- c:LM-OTH-002 -->
- The JNI library `litertlm_jni` ([`kotlin/java/com/google/ai/edge/litertlm/jni/BUILD:24-45`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/kotlin/java/com/google/ai/edge/litertlm/jni/BUILD#L24-L45))
  includes `c/engine.h` and also the C++ runtime's own headers (`runtime/engine/engine.h`,
  `runtime/conversation/conversation.h`, settings, model resources)
  ([`kotlin/java/com/google/ai/edge/litertlm/jni/litertlm.cc:39-66`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/kotlin/java/com/google/ai/edge/litertlm/jni/litertlm.cc#L39-L66)). A Kotlin report can
  therefore involve C++ paths Swift never reaches; Swift only calls the C API
  ([litert-lm/swift-api.md](litert-lm/swift-api.md)). <!-- c:LM-OTH-003 -->
- The NPU LLM executor is compiled out on iOS: `//build_config:disable_npu_backend` matches
  `@platforms//os:ios` ([`build_config/BUILD:323-332`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/build_config/BUILD#L323-L332)), and the executor factory then gets
  `LITERT_DISABLE_NPU` and drops `llm_litert_npu_compiled_model_executor`
  ([`runtime/executor/BUILD:452-490`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/BUILD#L452-L490)). macOS isn't in that group. <!-- c:LM-OTH-001 --> The vision, audio and embedding executors don't get
  the define and still compile their NPU option code on iOS
  ([litert-lm/build-and-release.md](litert-lm/build-and-release.md#build-time-feature-flags)).
- The Google Tensor backend can only be turned on for Android: both of its `config_setting`s carry
  `@platforms//os:android` ([`build_config/BUILD:385-410`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/build_config/BUILD#L385-L410)).
- NPU-specific code lives in `runtime/executor/npu/` and `llm_litert_npu_compiled_model_executor.*`
  (dynamic KV growth, the NPU embedder); see [litert-lm/executors.md](litert-lm/executors.md).

#### Linux and Windows

The CLI and the Python wheels are the main artifacts. GPU runs through the WebGPU accelerator and
Dawn from `prebuilt/`, loaded at run time with `--define=litert_runtime_link_mode=dynamic`
([`docs/getting-started/build-and-run.md:90-97`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/docs/getting-started/build-and-run.md#L90-L97)). Upstream CI builds them with Bazel and
CMake (the CMake super-build fetches LiteRT `main`; see
[litert-lm/build-and-release.md](litert-lm/build-and-release.md#cmake-mapped)).

#### Python

- `litert_lm` loads `liblitert-lm.{so,dylib}` or `litert-lm.dll` with `ctypes` and calls the C API
  ([`python/litert_lm/_ffi.py:155-181`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm/_ffi.py#L155-L181)), so it is a C API client like Swift. <!-- c:LM-OTH-005 -->
- Without `--define=use_prebuilt_c_api=true`, `python/litert_lm/BUILD`'s `default_libs` copies
  `libGemmaModelConstraintProvider`, `libLiteRtWebGpuAccelerator`, `libLiteRtTopKWebGpuSampler` and
  `libLiteRt` from `prebuilt/linux_x86_64` or `linux_arm64` (Linux) or `prebuilt/macos_arm64` (macOS)
  next to `liblitert-lm`; `prebuilt/macos_arm64`'s `libLiteRtMetalAccelerator.dylib` is not copied.
  With `use_prebuilt_c_api` the macOS package gets only the prebuilt `liblitert-lm.dylib`. At
  agello-next a from-source macOS `liblitert-lm` also links the Metal accelerator statically (ours:
  `43f0f5c3`) ([`python/litert_lm/BUILD:127-133`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm/BUILD#L127-L133), [`python/litert_lm/BUILD:232-237`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm/BUILD#L232-L237), [`python/litert_lm/BUILD:265-270`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm/BUILD#L265-L270), [`python/litert_lm/BUILD:369-388`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm/BUILD#L369-L388), [`python/litert_lm/BUILD:397-413`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm/BUILD#L397-L413), [`runtime/executor/BUILD:252-279`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/BUILD#L252-L279)). <!-- c:LM-OTH-006 -->
- `litert_lm_builder` (the `.litertlm` writer and `litertlm_peek`), `litert_lm_cli` and
  `litert_lm_eval` are separate packages ([tools.md](tools.md)).

#### JavaScript and the web

`@litert-lm/core` is versioned on its own (0.12.1 at the pin) and downloads its WebAssembly files
from jsDelivr at that same package version ([`js/packages/core/package.json:1-5`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/js/packages/core/package.json#L1-L5),
[`js/packages/core/scripts/download-wasm.js:34-36`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/js/packages/core/scripts/download-wasm.js#L34-L36)). Upstream builds separate JSPI and
Asyncify variants (`aa2401bf`) and, since `35b09b18`, streams model loads and defaults to the GPU
backend. Apps in `js/apps` (chat, embedding search, model tester) back Google's
web demos.

#### Omni: ASR, TTS, text-to-image

Omni is a multimodal streaming runtime inside LiteRT-LM: an `OmniEngine` that owns compiled models and
accelerators, `OmniSession`s that stream inputs and outputs, and `MultiStagedSession` pipelines of
stages, for speech recognition, text-to-speech and text-to-image ([`omni/README.md:3-38`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/omni/README.md#L3-L38)).
Its model parameters travel in `.litertlm` metadata (`AsrMetadata`, `TtsMetadata`,
`Text2ImageMetadata`), which is why the container version moved to 1.8 and 1.9
([versions.md](versions.md#container-versions)). <!-- c:LM-OTH-008 --> Nothing in `c/` or `swift/` refers to
Omni, so neither our framework nor Google's Swift package exposes it.

### What this means for reading upstream reports

- A GPU bug reported on Android, Linux, Windows, the web or Google's macOS binary runs ML Drift's
  OpenCL or WebGPU backends, not Metal. The shared parts are the LiteRT-LM runtime, LiteRT's
  compiled model and ML Drift's graph transformations and kernel selection; the Metal backend is
  Apple-only ([ml-drift/metal-backend.md](ml-drift/metal-backend.md)).
- A report against the closed parts (`libGemmaModelConstraintProvider`, the TopK samplers, the
  artisan GPU executor) can't be reproduced with our build, which links none of them.
- Python built from source on Linux and macOS ships the closed constraint provider; our build doesn't, so a JSON or
  tool-call constraint report from Python may take a different path ([litert-lm/conversation.md](litert-lm/conversation.md)).


## LiteRT

### Summary

The same `CompiledModel` code runs everywhere; what changes is which accelerators register, which buffer types and events exist, and how the GPU accelerator is found. On Apple our build links one GPU accelerator ("GPU Metal") statically. Elsewhere Google ships GPU accelerators as shared libraries that the GPU registry opens by name, picks OpenCL or WebGPU at run time, and reaches NPUs through a separate dispatch API. Google's macOS release uses the WebGPU accelerator (ML Drift over Dawn), not Metal.

### Platform defaults

The platform switches live in `litert_common.h` ([`litert/c/litert_common.h:134-175`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/c/litert_common.h#L134-L175)): <!-- c:RT-OTH-001 -->

| Platform | OpenCL | OpenGL | AHWB | ION | DMA-BUF | FastRPC | Vulkan | Metal | WebGPU | Sync fences |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Android | on | on | on | on | on | on | on | off | on | on |
| Linux arm64 | on | off | off | on | off | on | on | off | on | on |
| Linux x86-64, other non-Apple | on | off | off | off | off | off | on | off | on | on (Linux) |
| Windows | off | off | off | off | off | off | on | off | on | off |
| iOS, macOS | off | off | off | off | off | off | off | on | on | off |

"On" is the default; `LITERT_DISABLE_*` defines turn each off ([`litert/c/litert_common.h:161-173`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/c/litert_common.h#L161-L173), [`litert/c/litert_common.h:128-132`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/c/litert_common.h#L128-L132)). Vulkan support is marked experimental in the buffer-type enum ([`litert/c/litert_tensor_buffer_types.h:51-104`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/c/litert_tensor_buffer_types.h#L51-L104)).

### GPU accelerators and how they are found

The GPU registry opens these names, in order, when no accelerator is linked statically or passed by handle ([`litert/runtime/accelerators/gpu_registry.cc:47-87`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/accelerators/gpu_registry.cc#L47-L87)): <!-- c:RT-OTH-002 -->

| Platform | Libraries tried |
| --- | --- |
| Android | `libLiteRtGpuAccelerator.so`, then (with OpenCL) `libLiteRtClGlAccelerator.so`, `libLiteRtOpenClAccelerator.so`, then (with WebGPU) `libLiteRtWebGpuAccelerator.so` |
| iOS | `libLiteRtGpuAccelerator.dylib`, `libLiteRtMetalAccelerator.dylib`, `LiteRtMetalAccelerator`, then WebGPU |
| macOS, Linux, Windows | `libLiteRtGpuAccelerator`, then WebGPU, then OpenCL, then (macOS) Metal; Vulkan last everywhere it is compiled in |

The accelerators behind those names ([`litert/runtime/accelerators/gpu/ml_drift_gpu_accelerator.cc:170-217`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/accelerators/gpu/ml_drift_gpu_accelerator.cc#L170-L217), [`litert/runtime/accelerators/gpu/ml_drift_webgpu_accelerator.cc:47-52`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/accelerators/gpu/ml_drift_webgpu_accelerator.cc#L47-L52), [`litert/runtime/accelerators/gpu/ml_drift_webgpu_accelerator.cc:111-152`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/accelerators/gpu/ml_drift_webgpu_accelerator.cc#L111-L152)):

| Accelerator | Name it reports | Backend |
| --- | --- | --- |
| `ml_drift_gpu_accelerator` (`libLiteRtGpuAccelerator`) | "LiteRT GPU" | OpenCL; WebGPU when the options ask for it, or when the backend is automatic and OpenCL is not supported |
| `ml_drift_allgpu_accelerator`, `ml_drift_cl_gl_accelerator` | "LiteRT GPU" | WebGPU, OpenGL or OpenCL / OpenGL or OpenCL |
| `ml_drift_cl_accelerator` | "GPU OpenCL" | OpenCL |
| `ml_drift_opengl_accelerator` | "GPU OpenGL" | OpenGL |
| `ml_drift_webgpu_accelerator` | "GPU WebGPU" | WebGPU (Dawn); buffer types 20–26 |
| `ml_drift_vulkan_accelerator` | "GPU Vulkan" | Vulkan |
| `ml_drift_metal_accelerator` | "GPU Metal" | Metal ([litert/runtime.md](litert/runtime.md)) |

The generic accelerator chooses at delegate creation: WebGPU if the GPU options name it, or if they leave the backend automatic and OpenCL is not supported on the device; OpenCL otherwise ([`litert/runtime/accelerators/gpu/ml_drift_gpu_accelerator.cc:170-217`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/accelerators/gpu/ml_drift_gpu_accelerator.cc#L170-L217)). <!-- c:RT-OTH-003 --> The WebGPU accelerator registers handlers for the seven WebGPU buffer types and takes its device and queue from `kLiteRtEnvOptionTagWebGpuDevice` and `kLiteRtEnvOptionTagWebGpuQueue` ([`litert/runtime/accelerators/gpu/ml_drift_webgpu_accelerator.cc:111-152`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/accelerators/gpu/ml_drift_webgpu_accelerator.cc#L111-L152)).

Google's prebuilt libraries in LiteRT-LM's `prebuilt/` folder follow this split. The Android folders hold `libLiteRtGpuAccelerator`, `libLiteRtOpenClAccelerator` and `libLiteRtWebGpuAccelerator` with Dawn; the Linux and Windows folders hold `libLiteRt` and the WebGPU accelerator with Dawn; `ios_arm64` holds `libLiteRt` and the Metal accelerator; `macos_arm64` holds both the Metal and the WebGPU accelerators and Dawn (GitHub listing of `prebuilt/` at `df6df5b8`). <!-- c:RT-OTH-004 -->

### Google's macOS release

LiteRT-LM's `gpu_accelerator` build flag documents the intended defaults: "iOS uses Metal ("metal") and macOS uses WebGPU ("webgpu")" ([`runtime/executor/BUILD:179-194`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/BUILD#L179-L194)). Google's v0.18.0 `CLiteRTLM_mac.xcframework` (one universal `libCLiteRTLM_mac.dylib`, 137.8 MB) is built that way: its strings contain the `GpuWebGpuAccelerator` class, the name "GPU WebGPU" and the "Statically linked GPU accelerator registered." message, while Metal appears only as the file name `libLiteRtMetalAccelerator.dylib`, with no "GPU Metal" name (strings of the release asset, sha256 `5f6ee68d…`, inspected 2026-10-15). <!-- c:RT-OTH-005 --> So on a Mac, Google's native macOS binary runs ML Drift through Dawn on top of Metal, while Agello's iPhone app on the Mac uses the iOS binary and Metal directly, and our own build links the Metal accelerator on macOS too ([litert/runtime.md](litert/runtime.md), RT-RUN-005).

### NPUs

NPUs go through the dispatch API, not a TFLite GPU-style delegate: a compiler plugin compiles parts of the graph ahead of time or at load, and the dispatch library runs the result. The NPU accelerator registers only if a dispatch API is linked statically or `kLiteRtEnvOptionTagDispatchLibraryDir` is set ([`litert/runtime/accelerators/dispatch/dispatch_accelerator.cc:147-160`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/accelerators/dispatch/dispatch_accelerator.cc#L147-L160)). LiteRT-LM sets the dispatch and compiler-plugin directories only when an NPU backend is in use ([`runtime/util/litert_util.cc:134-174`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_util.cc#L134-L174)). Vendor code lives under `litert/vendors/` (Qualcomm, MediaTek, Google Tensor, Samsung, Intel OpenVINO, and others). `CompiledModel` applies the NPU accelerator only to models that contain dispatch ops, and tries compiler plugins (with an on-disk compilation cache when `kLiteRtEnvOptionTagCompilerCacheDir` is set) before building the interpreter ([`litert/runtime/compiled_model.cc:1044-1071`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/litert/runtime/compiled_model.cc#L1044-L1071), [`litert/runtime/compiled_model.cc:836-865`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/litert/runtime/compiled_model.cc#L836-L865)). <!-- c:RT-OTH-006 --> LiteRT-LM's embedders ask for NPU plus CPU on Android and CPU alone elsewhere ([`runtime/components/embedding_lookup/embedding_lookup_text.cc:292-298`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/embedding_lookup/embedding_lookup_text.cc#L292-L298)).

### Android buffers and events

Android adds buffer types that the CPU can also use without a copy: hardware buffers (AHWB), DMA-BUF and FastRPC buffers are treated as CPU-compatible when a tensor needs host memory (on Android and on Linux arm64) ([`litert/runtime/compiled_model.cc:1930-1934`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/litert/runtime/compiled_model.cc#L1930-L1934), [`litert/runtime/compiled_model.cc:1972-1987`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/litert/runtime/compiled_model.cc#L1972-L1987)). Sync-fence events exist on Linux and Android only: `Wait` polls the fence file descriptor with a timeout ([`litert/runtime/event.cc:46-102`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/event.cc#L46-L102)). A locked AHWB waits on its own fence inside the lock ([`litert/runtime/tensor_buffer.cc:1048-1060`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/tensor_buffer.cc#L1048-L1060)).

### CPU differences

YNNPACK, a second CPU delegate that runs before XNNPACK when present, is enabled in LiteRT-LM's `.bazelrc` only for `linux_arm64` ([`.bazelrc:266-268`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/.bazelrc#L266-L268), [`litert/runtime/accelerators/cpu_registry.cc:44-83`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/accelerators/cpu_registry.cc#L44-L83)). <!-- c:RT-OTH-007 --> On the web (Emscripten), XNNPACK does not claim JIT responsibility, so the hardware-set check never skips it, and external weights are restored for the CPU only when the CPU is requested and neither the GPU nor the NPU is, so weights streamed to the GPU are not loaded twice ([`litert/runtime/accelerators/xnnpack/xnnpack_accelerator.cc:123-135`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/accelerators/xnnpack/xnnpack_accelerator.cc#L123-L135), [`litert/runtime/compiled_model.cc:998-1017`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/litert/runtime/compiled_model.cc#L998-L1017)). On every platform, the WebNN accelerator registers only through the statically linked `LiteRtRegisterStaticLinkedAcceleratorWebNn` hook ([`litert/runtime/accelerators/webnn_registry.cc:31-40`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/accelerators/webnn_registry.cc#L31-L40), [`litert/runtime/accelerators/auto_registration.cc:63-67`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/accelerators/auto_registration.cc#L63-L67)).

## ML Drift

### Summary

All backends share ML Drift's core (`GraphFloat32` → `GpuModel`, selectors, kernels written once in
ML Drift's portable kernel language): only the runtime that compiles and runs a `GpuModel` differs.
ML Drift's README lists five GPU APIs: Vulkan, OpenCL (primary on Android), Metal, WebGPU (through
Dawn) and OpenGL ES 3.1+. <!-- c:MD-OTH-001 --> [`README.md:68-74`](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/README.md#L68-L74) The open tree at
the pin has backend directories for four of them, `cl/`, `gl/`, `metal/` and `webgpu/`; the
Vulkan runtime isn't in it. <!-- c:MD-OTH-002 --> [`README.md:90-96`](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/README.md#L90-L96)

### The backends

| API | Directory | Kernel language | Program caching | Work-group tuning | LiteRT delegate |
| --- | --- | --- | --- | --- | --- |
| OpenCL | `ml_drift/cl/` | OpenCL C | Compiled program binaries (`ProgramCache`) | Exhaustive, measured on a profiling queue, or fast | `delegate_opencl.cc` |
| WebGPU | `ml_drift/webgpu/` | WGSL (through Dawn) | Compute pipeline cache | `kExhaustive` unless the fast-tuning hint is set | `delegate_webgpu.cc` |
| OpenGL ES | `ml_drift/gl/` | GLSL ES 3.10 or 3.20 | — | — | `delegate_opengl.cc` |
| Vulkan | not in the open tree (`ml_drift/syrtis`, referenced by LiteRT) | — | — | — | `delegate_vulkan.cc` |
| Metal | `ml_drift/metal/` | MSL | Source only, recompiled on restore | First candidate, never measured | `delegate_metal.mm` |

**OpenCL** (Android's main path). `ProgramCache` keys programs by a fingerprint of their source and
options and can export and import compiled binaries (`AddProgramBinary`, `GetProgramBinary`), read
from the driver with `CL_PROGRAM_BINARIES`: unlike Metal, a cache hit skips compilation.
[`ml_drift/cl/program_cache.h:60-64`](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/cl/program_cache.h#L60-L64)
[`ml_drift/cl/cl_program.cc:194-205`](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/cl/cl_program.cc#L194-L205) `ClOperation::Tune` measures every candidate
work-group size on a profiling queue when there is more than one (`GetBestWorkGroupIndex`).
[`ml_drift/cl/cl_operation.cc:182-210`](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/cl/cl_operation.cc#L182-L210) The CL backend can also
record into `cl_khr_command_buffer` command buffers (`AddToCommandBuffer`), allowed by default by
the `allow_cl_khr_command_buffer` hint. [`ml_drift/cl/inference_context.h:105-108`](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/cl/inference_context.h#L105-L108)
[`ml_drift/common/model_hints.h:83-84`](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/common/model_hints.h#L83-L84) Qualcomm-specific paths (`qcom_command_buffer.h`,
`qcom_thin_filter.*`) live here too.

**WebGPU.** Kernels are translated to WGSL and compiled into compute pipelines through a
`ComputePipelineCache`; tuning happens between the two steps, exhaustively unless the fast-tuning hint is set. [`ml_drift/webgpu/compute_task.cc:295-302`](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/webgpu/compute_task.cc#L295-L302)
[`ml_drift/webgpu/inference_context.cc:283-286`](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/webgpu/inference_context.cc#L283-L286)
`WebGpuInfo` carries WebGPU's limits (by default 256 invocations per workgroup, 128 MB storage
binding, 256 MB buffers) and feature flags (subgroups, FP16, subgroup matrices).
[`ml_drift/common/gpu_info.h:823-875`](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/common/gpu_info.h#L823-L875) On Apple platforms WebGPU runs on Metal, and
ML Drift disables dp4a there as it does for Metal. [`ml_drift/common/gpu_info.cc:1657-1667`](https://github.com/PowerBeef/ml-drift/blob/15fa9e2ba632145992299e3f04786dc1f575373c/ml_drift/common/gpu_info.cc#L1657-L1667)
Google's own macOS builds of LiteRT-LM reach the GPU through a WebGPU accelerator
(this page); Agello's builds use Metal.

**OpenGL ES.** An EGL-based backend (`egl_environment`, `gl_inference_context`) that emits GLSL with
`#version 320 es` on OpenGL ES 3.2 or later and `#version 310 es` otherwise.
[`ml_drift/gl/gl_operation.cc:103-110`](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/gl/gl_operation.cc#L103-L110)

**Vulkan.** LiteRT's delegate has a Vulkan backend whose BUILD depends on
`@ml_drift//ml_drift/syrtis:command_buffer`, `:environment` and `:vulkan_wrapper`; no `syrtis`
package exists in the open ML Drift tree at the pin, so that delegate can't be built from the open
sources. <!-- c:MD-OTH-006 --> [`ml_drift_delegate/delegate/BUILD:381-393`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/ml_drift_delegate/delegate/BUILD#L381-L393)

### GPU detection beyond Apple

`GpuApi` has `kOpenCl`, `kMetal`, `kVulkan`, `kOpenGl`, `kWebGpu` and `kCuda`.
[`ml_drift/common/gpu_info.h:46-54`](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/common/gpu_info.h#L46-L54) The vendor comes from substrings of the device
description: Adreno and Qualcomm, Apple, Mali, PowerVR, AMD (also Radeon and Samsung's Xclipse),
Intel, NVIDIA (also Quadro and Tesla), Huawei's Maleoon, Broadcom, and the llvmpipe and lavapipe
software renderers. [`ml_drift/common/gpu_info.cc:34-61`](https://github.com/PowerBeef/ml-drift/blob/15fa9e2ba632145992299e3f04786dc1f575373c/ml_drift/common/gpu_info.cc#L34-L61)
Each vendor has its own info struct and family or generation tables (Adreno 6xx–8xx with compute
units, Mali generations, PowerVR, AMD and NVIDIA architectures, Intel generations), which steer
kernel choices the way Apple families do ([ml-drift/architecture.md](ml-drift/architecture.md)).
Selectors branch per vendor: for example, a convolution with external weights goes to the Apple
selector on Apple GPUs, to `ConvConstants`, `ConvWaveMatrix` or `ConvGeneric` on OpenGL, Vulkan and
WebGPU, and to Adreno-, Mali- and PowerVR-specific selectors elsewhere.
[`ml_drift/common/selectors/convolution_selector.cc:251-331`](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/common/selectors/convolution_selector.cc#L251-L331)

Precision on other APIs, with the default precision option: LiteRT's OpenCL delegate picks F32
(`kF32`) on PowerVR GPUs older than Rogue GM9xxx, or of an unrecognized PowerVR version, and
otherwise F16 (`kF16`) when the CL environment supports it, else F32; the Vulkan delegate picks F16
when explicit FP16 is supported, else F32. Both switch to `kF32F16` when `use_f32_accum_for_fp16`
is set. [`ml_drift_delegate/delegate/delegate_opencl.cc:664-693`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/ml_drift_delegate/delegate/delegate_opencl.cc#L664-L693)
[`ml_drift_delegate/delegate/delegate_vulkan.cc:356-375`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/ml_drift_delegate/delegate/delegate_vulkan.cc#L356-L375)

## Where in the code

| Concept | Symbol | Citation |
| --- | --- | --- |
| NPU executor off on iOS | `disable_npu_backend` | [`build_config/BUILD:323-332`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/build_config/BUILD#L323-L332) |
| GPU accelerator libraries opened by name | GPU registry | [`litert/runtime/accelerators/gpu_registry.cc:47-85`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/accelerators/gpu_registry.cc#L47-L85) |
| OpenCL or WebGPU choice | `ml_drift_gpu_accelerator` | [`litert/runtime/accelerators/gpu/ml_drift_gpu_accelerator.cc:170-217`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/accelerators/gpu/ml_drift_gpu_accelerator.cc#L170-L217) |
| GPU accelerator per Apple OS | `gpu_accelerator` flag | [`runtime/executor/BUILD:179-194`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/BUILD#L179-L194) |
| NPU registration | dispatch accelerator | [`litert/runtime/accelerators/dispatch/dispatch_accelerator.cc:147-160`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/accelerators/dispatch/dispatch_accelerator.cc#L147-L160) |
| Sync-fence events | `Event::Wait` | [`litert/runtime/event.cc:46-102`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/event.cc#L46-L102) |
| CPU delegates (YNNPACK, XNNPACK) | CPU registry | [`litert/runtime/accelerators/cpu_registry.cc:44-83`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/accelerators/cpu_registry.cc#L44-L83) |
| Python front end | `_ffi.py` | [`python/litert_lm/_ffi.py:155-181`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm/_ffi.py#L155-L181) |
| Kotlin backends | `Backend` | [`kotlin/java/com/google/ai/edge/litertlm/Config.kt:137-161`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/kotlin/java/com/google/ai/edge/litertlm/Config.kt#L137-L161) |
| Omni | `OmniEngine`, `OmniSession` | [`omni/README.md:3-38`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/omni/README.md#L3-L38) |
| ML Drift backends | README | [`README.md:68-74`](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/README.md#L68-L74) |
| OpenCL precision | `delegate_opencl.cc` | [`ml_drift_delegate/delegate/delegate_opencl.cc:664-693`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/ml_drift_delegate/delegate/delegate_opencl.cc#L664-L693) |

## Open questions

- Which GPU accelerator Python actually runs on a Mac (the copied WebGPU accelerator, or Metal
  linked into a from-source `liblitert-lm`, or what Google's prebuilt `liblitert-lm.dylib`
  contains) isn't shown by the build files and wasn't checked at run time.
- Whether Google's prebuilt Android and Linux LiteRT-LM binaries include the closed Vulkan
  (`syrtis`) backend is not verified here.
