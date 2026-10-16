# Glossary

> **Scope:** the terms this reference uses, what each means here, the wording to prefer and the
> wording to avoid, and the page that defines it. **Out of scope:** the values themselves ([facts.md](facts.md)); how things work
> (the linked pages).
> **Pinned to:** LiteRT-LM `7b872544`, LiteRT `362759838`, ML Drift `15fa9e2`; shipped: LiteRT-LM
> `85b1a776` with Google's prebuilt v0.18.0 binary.
> **Applies to:** all platforms; Apple terms marked. **Last verified:** 2026-10-15

## Summary

- Most mistakes in older notes come from one word covering two things: "context" (trained, exported,
  allocated, filled), "Apple10" (Apple's family or ML Drift's), "our LiteRT-LM" (what ships or what
  we build), "the GPU sampler" (the setting or the closed library), "Neural Engine" (it isn't used).
  The [first table](#words-that-mean-two-things) settles those; the rest are by area.
- In user docs prefer the plain wording in the "Prefer" column; in commit messages and upstream
  text, name the build and commit as well.
- Each row links the page that defines the term. Terms with a number point to the fact row that
  carries it.

## Concepts

How to read the tables: **Term** is the word as this reference uses it (code names in backticks, as
the code spells them). **Prefer / avoid** gives the wording to use and the wording that misleads.
**Why** says what goes wrong with the avoided wording. **Defined in** links the owning page.

## Words that mean two things

| Term | Meaning | Prefer / avoid | Why | Defined in |
| --- | --- | --- | --- | --- |
| trained context | Positions the model was trained for: 131,072 for E2B/E4B ("128K"), 262,144 for the larger Gemma 4 models ([F-048](facts.md#f-048)) | prefer "trained context"; avoid "the model's context" when the file or runtime is meant | the phone never gets 128K with these files | [models/gemma4-e4b.md](models/gemma4-e4b.md#trained-context-export-window-allocated-context) |
| export window | The `.litertlm` file's context dimension as written: the magic number 32003 ([F-049](facts.md#f-049)) | prefer "export window"; avoid "max context" | 32003 is a placeholder rewritten at load, not a usable size | [litert-lm/executors.md](litert-lm/executors.md#magic-numbers-and-the-allocated-context) |
| allocated context | Positions the KV cache gets at load: `maxNumTokens` below 32,003, else 32,000; 4,096 if unset ([F-050](facts.md#f-050)) | prefer "allocated context" (or "KV capacity"); avoid "window" when the KV size is meant | "32K" allocates 32,000, which is why 918 MB and 940 MB both circulate | [litert-lm/executors.md](litert-lm/executors.md#magic-numbers-and-the-allocated-context) |
| filled length | Positions that hold tokens now (the session's current step minus the pending token) | prefer "filled length"; avoid "context" for it | attention and conversions are bounded by it, memory by the allocated context | [litert-lm/executors.md](litert-lm/executors.md#the-attention-mask) |
| allocated context vs output limit | `max_num_tokens` (everything held in the KV cache) vs `max_output_tokens` (decode steps per request) ([F-055](facts.md#f-055)) | never "max tokens" alone | they fail differently: KV full ends the session's room, the output limit ends one reply | [litert-lm/engine-and-sessions.md](litert-lm/engine-and-sessions.md#what-ends-a-decode) |
| documented build | Our source build: LiteRT-LM `7b872544` + LiteRT `362759838` + ML Drift `15fa9e2` (`agello-next`) | prefer "our build at `<commit>`" or the build id; avoid "our LiteRT-LM" without a commit | its runtime patches don't reach users yet ([F-018](facts.md#f-018)) | [versions.md](versions.md#what-ships-and-what-is-documented) |
| shipped wrapper, Google's v0.18.0 binary | What Agello ships: the Swift wrapper `agello` `85b1a776` (v0.18.0 plus two Swift patches) with Google's prebuilt `CLiteRTLM.xcframework` ([F-003](facts.md#f-003)) | prefer "Google's v0.18.0 binary" in user docs, "the shipped wrapper" for the Swift side; avoid "our fork" or "Agello's fork" for it | the fork has much more than the wrapper Agello vendors | [versions.md](versions.md#what-ships-and-what-is-documented), [patches.md](patches.md#what-agello-ships-c3) |
| Apple family (Apple's) | `MTLGPUFamily.appleN`, what `MTLDevice.supportsFamily` answers | prefer "Apple's `MTLGPUFamily` AppleN" when it matters which | the M6 is Apple11 here and Apple10 in ML Drift ([F-087](facts.md#f-087)) | [apple-platform.md](apple-platform.md#gpu-families-apples-and-ml-drifts) |
| ML Drift family | `AppleInfo::Family` (Apple1-Apple10), derived from the Metal device *name* by ML Drift's table | prefer "ML Drift family AppleN"; avoid presenting it as `MTLGPUFamily` | it is a name lookup, not a capability query; unknown names give Apple1 | [ml-drift/architecture.md](ml-drift/architecture.md#apple-gpu-families) |
| GPU Neural Accelerators | Apple's matrix units in each GPU core of M5/A19-class GPUs, which ML Drift reaches through `ConvAppleMPP` (Metal Performance Primitives TensorOps) ([F-091](facts.md#f-091)) | prefer "GPU Neural Accelerators"; avoid "Neural Engine", "ANE" | the Apple Neural Engine is a different unit, and this stack doesn't use it ([F-084](facts.md#f-084)) | [apple-platform.md](apple-platform.md#concepts), [ml-drift/kernels.md](ml-drift/kernels.md#convapplempp-the-gpu-neural-accelerators) |
| configured sampler backend | What `GetSamplerBackend` returns (GPU on a GPU engine by default), as opposed to the sampler that actually loads | prefer "sampler backend setting" or "the closed TopK Metal sampler" by name; avoid "the GPU sampler" without saying which | FP16 logits follow the setting even when the CPU sampler runs ([F-097](facts.md#f-097)) | [litert-lm/components.md](litert-lm/components.md#fp16-logits-and-the-greedy-half-path) |
| file template | The `jinja_prompt_template` inside a `.litertlm` file | prefer "the file's template"; avoid "the chat template" without saying which | four templates circulate and the file's is what runs ([F-032](facts.md#f-032)) | [models/tokenizer-and-chat-format.md](models/tokenizer-and-chat-format.md#four-templates) |
| draft (upstream) | A text in `docs/agello/upstream/`, not posted | avoid "filed", "reported", "submitted" for drafts | only #3913 is posted ([F-126](facts.md#f-126)) | [contributing.md](contributing.md#our-drafts) |

## Context, memory and limits

| Term | Meaning | Prefer / avoid | Why | Defined in |
| --- | --- | --- | --- | --- |
| magic number | A prime greater than 10 the exporter writes for a dimension chosen at load (32003 in the Gemma 4 files) | prefer "magic number (placeholder)"; avoid "dynamic shape" | the shape is static once rewritten | [litert/runtime.md](litert/runtime.md#magic-numbers) |
| KV cache, bytes per position | The INT8 key and value caches; E4B 28,672 bytes, E2B 9,216 bytes per allocated position ([F-057](facts.md#f-057), [F-060](facts.md#f-060)) | prefer "per allocated position"; avoid "per token" when the allocation is meant | the cost is paid for every allocated position, filled or not | [models/gemma4-e4b.md](models/gemma4-e4b.md#kv-caches) |
| KV sharing, KV-shared layer | A layer that computes queries only and reads the cache of the last non-shared layer of its type (E4B 24-41 read 22 and 23; E2B 15-34 read 13 and 14) | prefer "KV-shared layer"; avoid "cross-layer attention" | — | [models/gemma4-e4b.md](models/gemma4-e4b.md#kv-sharing) |
| local / global layer | Sliding-window attention (512 positions in E2B/E4B, head 256) / full attention (head 512); config names `sliding_attention` / `full_attention` | prefer "local", "global" | — | [models/gemma4-e4b.md](models/gemma4-e4b.md#local-and-global-layers) |
| sliding window (W) | The keys a local layer attends to, `(p − W, p]` | distinguish from the context window | — | [litert/ml-drift-delegate.md](litert/ml-drift-delegate.md#bounded-attention-ours-ede28c64-31386317) |
| window start | `max(0, p − W + 1)` rounded down to 32 keys, where window-aware kernels start reading | — | — | [ml-drift/kernels.md](ml-drift/kernels.md#the-sliding-window-ours) |
| app memory limit | The footprint at which iOS may terminate the app, read as `phys_footprint` + `os_proc_available_memory()` ([F-071](facts.md#f-071)) | prefer "app memory limit"; avoid "available RAM", "app RAM" | the limit is not the device's RAM | [apple-platform.md](apple-platform.md#the-limit-and-how-to-read-it) |
| early limit reading | The 6 GiB the same reading returns early in some runs on an iPhone 17 Pro, before the first engine load ([F-072](facts.md#f-072)) | avoid quoting it as the limit | it produced the "6.4 GB" in VALIDATION.md | [apple-platform.md](apple-platform.md#c1-86-gb-or-64-gb) |
| footprint | `phys_footprint` from `TASK_VM_INFO`, the quantity the limit applies to | prefer "footprint"; avoid "resident size", "RSS" | RSS misses compressed and GPU memory | [apple-platform.md](apple-platform.md#unified-memory-and-the-footprint) |
| jetsam | iOS terminating a process for memory (`per-process-limit` and other reasons) | prefer "jetsam" or "terminated for memory"; avoid "OOM crash" | it isn't a crash in the app's code | [apple-platform.md](apple-platform.md#memory-on-ios) |
| working set (`recommendedMaxWorkingSetSize`) | Metal's estimate of GPU allocation before performance suffers | avoid using it as the app limit | on the iPhone 17 Pro it happens to be 16 KiB above the limit ([F-074](facts.md#f-074)) | [apple-platform.md](apple-platform.md#concepts) |
| unified memory | CPU and GPU share system memory; Metal allocations count in the footprint | — | GPU buffers aren't "free" memory on iOS | [apple-platform.md](apple-platform.md#unified-memory-and-the-footprint) |
| ballast, envelope | Agello's memory suite: incompressible memory allocated so a 12 GB phone behaves like a smaller limit ([F-077](facts.md#f-077)) | say "simulated limit of N GB" and give the effective limit | three runs read the early limit and tested more memory than intended | [apple-platform.md](apple-platform.md#the-8-gb-envelope-and-why-three-runs-didnt-test-it) |
| kept conversation | Agello's chat conversation kept alive (with a KV snapshot) while a helper conversation runs | — | how long the snapshot lives differs between builds ([F-079](facts.md#f-079)) | [litert-lm/engine-and-sessions.md](litert-lm/engine-and-sessions.md#the-kept-conversation-after-a-helper-c11) |
| KV snapshot | Host-memory copy of a session's KV state taken at a context switch or clone | prefer "snapshot"; "raw snapshot" for ours, "lock-path snapshot" for Google's | — | [litert-lm/engine-and-sessions.md](litert-lm/engine-and-sessions.md#context-switches-what-is-copied-and-where) |
| raw snapshot | Our KV snapshot that copies a shared `MTLBuffer`'s bytes as they are (`aec81cb7`) | avoid "zero-copy" | it copies | [patches.md](patches.md#memory) |
| context switch | Loading another session's context into the executor: snapshot out, restore in | avoid "session swap" | — | [litert-lm/engine-and-sessions.md](litert-lm/engine-and-sessions.md#context-switches-what-is-copied-and-where) |
| orphaned handler | A released session's context handler held only by the resource manager; ours skips its snapshot (`ed998d80`) | — | — | [litert-lm/engine-and-sessions.md](litert-lm/engine-and-sessions.md#the-resource-manager) |
| host shadow | The aligned CPU copy a Metal buffer lock returns | avoid "mapping", "zero-copy" for Metal locks | the data is copied or converted | [litert/runtime.md](litert/runtime.md#tensor-buffers) |
| direct copy | Our `memcpy` between a shared linear Metal buffer and its host shadow, skipping ML Drift's conversion | — | — | [litert/ml-drift-delegate.md](litert/ml-drift-delegate.md#the-metal-buffer-handler) |
| zero-copy registration | Binding a buffer whose type the delegate accepts, without a runtime copy | only for `RegisterBuffer` | not for host locks of GPU buffers | [litert/runtime.md](litert/runtime.md#tensor-buffers) |

## Builds, versions and the fork

| Term | Meaning | Prefer / avoid | Why | Defined in |
| --- | --- | --- | --- | --- |
| `agello-next` | Our fork branch with the source build (LiteRT-LM `7b872544`) | name the commit | — | [versions.md](versions.md#ours-on-top) |
| `agello` (branch) | The fork's default branch: v0.18.0 plus the two Swift patches Agello vendors, plus docs and tools | — | — | [patches.md](patches.md#where-our-commits-live) |
| `research/3913` | An annotated tag (not a branch) for the #3913 investigation ([F-015](facts.md#f-015)) | avoid "branch" | the landing page and agello-bench README still call it one | [patches.md](patches.md#the-fork-landing-page-c4) |
| build id | 12 hex digits of SHA-256 over the three source descriptions and the flags; `dev` outside `--config=agello` ([F-012](facts.md#f-012)) | prefer "build id"; avoid "build number" | — | [litert-lm/build-and-release.md](litert-lm/build-and-release.md#build-stamping) |
| build info, build stamp | The JSON `litert_lm_get_build_info()` returns / the mechanism that writes it | — | — | [litert-lm/build-and-release.md](litert-lm/build-and-release.md#build-stamping) |
| working-tree build | A measured build whose build info shows `+<hash>` (uncommitted changes) | say "commit X plus local changes" | not reproducible from commit X | [patches.md](patches.md#evidence-and-its-limits) |
| `BUILD-INFO.txt` | Text file beside the framework in `dist/` and in releases; not inside the binary | — | — | [litert-lm/build-and-release.md](litert-lm/build-and-release.md#the-xcframework) |
| `--config=agello` | Our Bazel configuration block in `.bazelrc` | prefer the exact name; avoid "agello build flags" | — | [litert-lm/build-and-release.md](litert-lm/build-and-release.md#the-agello-config) |
| pin / override | A commit plus archive SHA-256 in `WORKSPACE` / a `--override_repository` line in `.bazelrc.user` | avoid "lock" / "local pin" | — | [litert-lm/build-and-release.md](litert-lm/build-and-release.md#external-repositories-and-pins) |
| static accelerator | The GPU accelerator linked into `CLiteRTLM` and registered through `LiteRtStaticLinkedAcceleratorGpuDef` | avoid "built-in delegate", "built-in GPU" | — | [litert-lm/build-and-release.md](litert-lm/build-and-release.md#linking-the-metal-accelerator-statically) |
| nightly tag | `<VERSION>-nightly.<YYYYMMDD>` on upstream's `nightly` branch, force-pushed | prefer the commit | tags move | [versions.md](versions.md#how-upstream-versions-and-tags) |
| release branch | `release/vX.Y.Z`, where upstream release tags live | — | v0.18.0 is not on main | [versions.md](versions.md#how-upstream-versions-and-tags) |
| C API version | `LITERT_LM_C_API_VERSION_*`, independent of the runtime version: 0.2.0 in v0.18.0, 1.0.0 at the pin ([F-006](facts.md#f-006)) | prefer "C API 1.0.0"; avoid "API level" or calling it the LiteRT-LM version | — | [litert-lm/c-api.md](litert-lm/c-api.md#version-macros) |
| container version | Major.minor.patch in a `.litertlm` header; the writer's is 1.9.0 at the pin ([F-016](facts.md#f-016)) | say "file" or "writer" version | files are 1.5-1.7 | [litert-lm/file-format.md](litert-lm/file-format.md#reading-the-header) |
| `prebuilt/` | Google's binaries in Git LFS (accelerators, closed samplers, constraint provider, `libLiteRt`) | — | — | [litert-lm/build-and-release.md](litert-lm/build-and-release.md#the-prebuilt-binaries) |
| overlapped upstream | Google landed a change in the same place after our base (e.g. LiteRT `bd326784`); ours needs a rebase | avoid "merged upstream" unless Google's code makes ours unnecessary | — | [patches.md](patches.md#upstream-status-the-terms) |
| fixed upstream | Google's code makes our patch unnecessary from a named commit on (XNNPACK via `c05e67a2`, [F-129](facts.md#f-129)) | — | — | [patches.md](patches.md#upstream-status-the-terms) |
| track tag | The plan item in our commit subjects (`[T4]`, `[M1.2]`, `[P6]`, `[SIM]`, `[CPU]`) | explain or drop in upstream text | readers upstream don't know our plan | [patches.md](patches.md#tags-in-commit-subjects) |
| copybara-service | Google's internal mirror; it authors almost every merged LiteRT-LM pull request | avoid "the LiteRT-LM bot" | — | [contributing.md](contributing.md#the-three-upstreams) |
| CLA | Google's Contributor License Agreement, needed before LiteRT or ML Drift accepts a pull request | — | — | [contributing.md](contributing.md#what-signing-the-cla-involves) |
| modification notice | The header comment a modified Apache-2.0 file carries; only `swift/Conversation.swift` has one ([F-131](facts.md#f-131)) | — | — | [contributing.md](contributing.md#license-notices-in-modified-files) |

## Models and files

| Term | Meaning | Prefer / avoid | Why | Defined in |
| --- | --- | --- | --- | --- |
| E2B / E4B | Gemma 4 sizes counted in effective parameters | prefer "E4B"; avoid "4B" | the file holds 8B parameters | [models/gemma4-family.md](models/gemma4-family.md#the-family-at-a-glance) |
| effective parameters | Google's headline count for E2B/E4B, without the per-layer embedding tables: "4.5B effective (8B total)" ([F-037](facts.md#f-037)) | say "effective (Google's count)"; avoid an unqualified "4B model" | — | [models/gemma4-e4b.md](models/gemma4-e4b.md#parameter-counts-and-effective-parameters) |
| A4B | "Active 4B": the 26B mixture-of-experts model uses about 3.8B parameters per token | prefer "26B-A4B" | — | [models/gemma4-family.md](models/gemma4-family.md#the-mixture-of-experts-model-26b-a4b) |
| unified (12B) | Gemma 4 12B: encoder-free, image patches and raw audio projected into the decoder | — | — | [models/gemma4-family.md](models/gemma4-family.md#the-family-at-a-glance) |
| per-layer embeddings (PLE) | A second token embedding in E2B/E4B giving each token a 256-value vector per layer, added after each layer's MLP through a gated bottleneck | prefer "per-layer embeddings"; expand "PLE" once per page | — | [models/gemma4-e4b.md](models/gemma4-e4b.md#per-layer-embeddings) |
| projection-only per-layer embeddings | EmbeddingGemma 2's per-layer inputs computed from the token embeddings by one linear map, with no per-layer table | avoid "per-layer embedder" for it | the E4B section of that name is a table lookup | [models/embeddinggemma.md](models/embeddinggemma.md#the-model-as-trained) |
| double-wide MLP | `use_double_wide_mlp`: KV-shared layers get twice the feed-forward width (E2B 12,288); E4B doesn't use it | prefer "double-wide feed-forward"; avoid "2x MLP" without the layer range | — | [models/gemma4-e2b.md](models/gemma4-e2b.md#double-wide-mlp-in-the-shared-layers) |
| p-RoPE (proportional RoPE) | RoPE on a fraction of the head: E4B global layers rotate 128 of 512 dimensions, θ 10⁶ | spell out once | — | [models/gemma4-e4b.md](models/gemma4-e4b.md#rotary-positions-per-layer-type) |
| rotary block (512-token block) | Our file's split of each position as 512·h + l, so the float16 math sees at most 511 ([F-101](facts.md#f-101)) | prefer "512-token rotary blocks"; avoid "RoPE fix" alone | the 2048-block version fixed fewer GPUs | [models/gemma4-e4b.md](models/gemma4-e4b.md#our-file-512-token-rotary-blocks) |
| vision token budget | Soft tokens per image: 70, 140 or 280 in the E4B file ([F-066](facts.md#f-066)) | avoid quoting 560 or 1,120 for the phone | the runtime rejects them with this file | [models/gemma4-e4b.md](models/gemma4-e4b.md#vision-sections) |
| soft token | One pooled vision (or audio) embedding row fed to the decoder | — | — | [models/gemma4-e4b.md](models/gemma4-e4b.md#vision-encoder) |
| placeholder | `<\|image\|>` / `<\|audio\|>` in rendered text, replaced by media embeddings in LiteRT-LM | — | the model never sees the placeholder token | [models/tokenizer-and-chat-format.md](models/tokenizer-and-chat-format.md#how-litert-lm-applies-the-template) |
| control token / user-defined piece | SentencePiece piece types: `<pad>`, `<eos>`, `<bos>` vs the chat markers (`<\|turn>`, …) | — | marker text typed by a user still becomes the token | [models/tokenizer-and-chat-format.md](models/tokenizer-and-chat-format.md#vocabulary-layout-and-special-tokens) |
| MTP drafter | The 4-block speculative-decoding head (`tf_lite_mtp_drafter`) that reads the main model's KV caches | prefer "MTP drafter"; avoid "draft model" | it has no KV cache of its own | [models/gemma4-e4b.md](models/gemma4-e4b.md#the-mtp-drafter-section) |
| assistant model | Google's Hugging Face name (`-assistant`) for a Gemma 4 MTP drafter | prefer "MTP drafter"; avoid "assistant" alone in user text | confused with the chat role | [models/gemma4-family.md](models/gemma4-family.md#the-assistant-models-mtp-drafters) |
| MTP round, priming step, bonus token | One drafting loop (3 drafts) plus a verify run / the plain decode step that starts MTP / the verifier's own pick, always emitted ([F-047](facts.md#f-047)) | — | — | [litert-lm/executors.md](litert-lm/executors.md#mtp-speculative-decoding) |
| artisan file (`-gpu`) | A `.litertlm` with one `tf_lite_artisan_text_decoder` section for Google's closed `GPU_ARTISAN` engine ([F-028](facts.md#f-028)) | prefer "artisan `-gpu` file"; avoid "GPU version of the model" | the regular file runs on the GPU too | [models/gemma4-e2b.md](models/gemma4-e2b.md#the-artisan-gpu-variants) |
| inline / external weights | Weights inside the TFLite flatbuffer (every E2B section) / constant data after the graph or in a `TFLiteWeights` section (E4B main graph) | — | — | [litert-lm/file-format.md](litert-lm/file-format.md#external-weights) |
| `.litertlm` container | LiteRT-LM's model file: 32-byte preamble, FlatBuffers header, 16 KiB-aligned typed sections | prefer "container"; avoid "bundle" | `.task` is the bundle | [litert-lm/file-format.md](litert-lm/file-format.md#layout) |
| section, `model_type` | One typed byte range of a `.litertlm` file / the string naming a TFLite section's role (`tf_lite_prefill_decode`, …) | use the exact string | — | [litert-lm/file-format.md](litert-lm/file-format.md#modeltype-strings) |
| section hint | The `backend_constraint` and `prefer_activation_type` items of a section | avoid "selector" | hints never select among sections | [litert-lm/file-format.md](litert-lm/file-format.md#what-the-hints-do) |
| signature | A named entry point of a `.tflite` graph (`prefill_1024`, `decode`, `verify`, `encoder_1x128`) | avoid "subgraph" unless the index is meant; for EmbeddingGemma avoid "context size" | — | [litert-lm/executors.md](litert-lm/executors.md#signatures-in-the-gemma-4-files) |
| `param_tensor` | INT32 `[1, 1, 1, 7]` graph input: update start, end, end, then zeros | — | — | [litert-lm/executors.md](litert-lm/executors.md#paramtensor) |
| EmbeddingGemma 2 | Google DeepMind's multimodal embedding model (768 dimensions, 740M parameters; text part 270M) ([F-119](facts.md#f-119)) | prefer "EmbeddingGemma 2 text 270m" for Agello's file; avoid "EmbeddingGemma" alone | it also names the first, Gemma 3 based model | [models/embeddinggemma.md](models/embeddinggemma.md#the-variants-google-publishes) |
| task prompt | The prefix telling the embedder what a vector is for (`task: search result \| query: `, `title: none \| text: `) | prefer "task prompt" or "prefix"; avoid "instruction" | — | [models/embeddinggemma.md](models/embeddinggemma.md#task-prompts) |
| Matryoshka dimensions (MRL) | Leading sub-vectors (128, 256, 512 of 768) trained to work alone; truncate, then re-normalize | prefer "truncate to N dimensions"; avoid "compress", "quantize" | different operations | [models/embeddinggemma.md](models/embeddinggemma.md#matryoshka-dimensions) |
| `max_input_length` (embedding) | The bound that decides which encoder signatures load (1,024 from the file by default) ([F-056](facts.md#f-056)) | distinguish from the trained context (8,192) and the longest signature (8,192) | — | [models/embeddinggemma.md](models/embeddinggemma.md#how-litert-lm-runs-it) |
| mean pooling | Averaging the final token states over real tokens | — | — | [models/embeddinggemma.md](models/embeddinggemma.md#the-text-encoder-graph) |

## LiteRT-LM runtime

| Term | Meaning | Prefer / avoid | Why | Defined in |
| --- | --- | --- | --- | --- |
| executor | The object that runs a model's graphs (`LlmExecutor`); one per engine | avoid "runner" for it | — | [litert-lm/executors.md](litert-lm/executors.md#the-executor-interfaces) |
| static / dynamic executor | `LlmLiteRtCompiledModelExecutorStatic` (fixed shapes; every Gemma 4 file) / `…Dynamic` (dynamic KV, CPU only) | — | — | [litert-lm/executors.md](litert-lm/executors.md#creating-an-executor) |
| work group, cautious greedy | One prefill run (signature + its tokens) / the GPU split rule: a remainder of at least half a signature gets a padded run of it ([F-054](facts.md#f-054)) | — | — | [litert-lm/executors.md](litert-lm/executors.md#prefill-and-prompt-chunking) |
| pending token | The last token of each prefill or decode call, processed by the next call | — | — | [litert-lm/executors.md](litert-lm/executors.md#concepts) |
| `LitertState` | The KV caches of one context; policies `kInplace`, `kPingPong`, `kGpuOptimizedInplace` | — | — | [litert-lm/executors.md](litert-lm/executors.md#kv-state) |
| internal / external sampling | The executor samples (`Decode`) / the session samples logits (`DecodeLogits`) | avoid "session sampler" for the executor's own | — | [litert-lm/executors.md](litert-lm/executors.md#sampling-hand-off) |
| `TopPSampler` | The open CPU sampler (top-k, top-p, weighted draw; k = 1 is argmax) | — | — | [litert-lm/components.md](litert-lm/components.md#samplers) |
| TopK Metal sampler | Google's closed GPU sampler (`libLiteRtTopKMetalSampler`); in Google's binary, not in ours | avoid "the GPU sampler" without naming the build | — | [litert-lm/components.md](litert-lm/components.md#samplers) |
| greedy half path | Our argmax over FP16 logits compared as integers (`fdd5bac7`) | — | — | [litert-lm/components.md](litert-lm/components.md#fp16-logits-and-the-greedy-half-path) |
| benchmark mode | An engine created with benchmark parameters (`is_benchmark`); it also selects the busy GPU wait ([F-019](facts.md#f-019)) | — | Agello turns it on to get stats | [litert-lm/executors.md](litert-lm/executors.md#concepts) |
| active / passive wait | Busy-spinning on a completion flag / `waitUntilCompleted` | prefer "busy-wait" for active in user text | — | [litert/ml-drift-delegate.md](litert/ml-drift-delegate.md#gpubackendmetal-and-its-litert-glue) |
| execution thread / callback thread | The two single-thread pools of `ThreadedExecutionManager`: every task of an engine runs on the first (`execution_threa` on Apple); final callbacks on the second | prefer these names; avoid "decode thread", "worker thread" | the execution thread runs prefill too | [litert-lm/engine-and-sessions.md](litert-lm/engine-and-sessions.md#execution-managers-and-threads) |
| task, task chain | One unit of session work (prefill, decode, clone, score) / a session's tasks, each depending on the previous | avoid "job", "request" | — | [litert-lm/engine-and-sessions.md](litert-lm/engine-and-sessions.md#task-lifecycle) |
| poisoned session | A session whose chain ends in a cancelled (or KV-full) task, so new tasks are born ended | say "cancelled session" in user docs | — | [litert-lm/engine-and-sessions.md](litert-lm/engine-and-sessions.md#cancellation) |
| context handler | A session's view of the shared executor (`ContextHandler`): config, state, shared processed context | — | — | [litert-lm/engine-and-sessions.md](litert-lm/engine-and-sessions.md#the-resource-manager) |
| preface / history | The fixed start of a conversation, rendered into the first prefill / messages sent and completed replies | avoid "history" for the preface | the preface isn't in `GetHistory()` | [litert-lm/conversation.md](litert-lm/conversation.md#preface-and-history) |
| channel, reasoning channel | A named span of model output (`thought` for Gemma 4) streamed and stored apart from the answer | prefer "thinking channel" for `thought` | — | [litert-lm/conversation.md](litert-lm/conversation.md#channels) |
| channel filter, refill | Removing channel text from the KV cache by rewinding to a checkpoint and prefilling the reply without it / that prefill | avoid "deleting" | nothing is erased; positions are overwritten | [litert-lm/conversation.md](litert-lm/conversation.md#channels-and-thinking) |
| prefill text (turn difference) | `render(history + new)` minus `render(history)`: what a later turn sends | avoid "the prompt" for it | — | [litert-lm/conversation.md](litert-lm/conversation.md#what-is-prefilled-for-a-turn) |
| model data processor | The model-specific converter between messages and session inputs (`Gemma4DataProcessor`) | — | — | [litert-lm/conversation.md](litert-lm/conversation.md#model-data-processors-and-multimodal-input) |
| llguidance, response format | The open constraint library (regex, JSON schema, Lark) / Swift's per-message JSON-schema or regex constraint | prefer "response format" when naming the API; avoid "structured output" | — | [litert-lm/conversation.md](litert-lm/conversation.md#constraints) |
| FST provider (closed Gemma constraint provider) | Google's closed prebuilt library that builds tool-call constraints for Gemma processors | prefer "the closed Gemma FST provider" | it isn't a `ConstraintProvider`, and our build lacks it ([F-009](facts.md#f-009)) | [litert-lm/conversation.md](litert-lm/conversation.md#constraints) |
| logit mask, mask runner, constrained decoder | `BitmapLogitMask` / `SparseLogitMask`; `LiteRtLogitMaskRunner` (host or device); per-row constraint states plus a runner | — | — | [litert-lm/components.md](litert-lm/components.md#logit-masks) |
| thinking budget | `ThinkingBudgetConstraint`: forces the channel's end tokens after N thinking tokens | — | — | [litert-lm/components.md](litert-lm/components.md#constraints) |
| end-of-media embedding | Constant embedding from `tf_lite_end_of_vision` / `tf_lite_end_of_audio` | — | — | [litert-lm/components.md](litert-lm/components.md#embedding-lookups) |
| status code, last error | `LiteRtLmStatusCode` (17 canonical codes, 0 = OK) / the thread-local message of the last failed C call | avoid "errno" except as an analogy | — | [litert-lm/c-api.md](litert-lm/c-api.md#status-codes-out-parameters-and-the-last-error) |
| stream chunk, final chunk | `LiteRtLmStreamChunk` (text, final flag, error), valid only during the callback / the one with `is_final` | — | — | [litert-lm/c-api.md](litert-lm/c-api.md#threads-and-callbacks) |
| session API / conversation API | The two C generation layers: plain inputs and text out / JSON messages with templates, tools and constraints | — | — | [litert-lm/c-api.md](litert-lm/c-api.md#two-ways-to-generate) |
| `StreamContext` | The Swift object passed as `callback_data` that bridges C callbacks to an `AsyncThrowingStream`; ours holds its conversation weakly | — | — | [litert-lm/swift-api.md](litert-lm/swift-api.md#our-fix-a-stream-holds-its-conversation-weakly) |
| automatic tool calling | The Swift wrapper running tool calls itself, up to 25 rounds | — | — | [litert-lm/swift-api.md](litert-lm/swift-api.md#tools) |
| experimental flags | Process-wide `ExperimentalFlags` static vars, settable after opting in | — | read at different times; see the page | [litert-lm/swift-api.md](litert-lm/swift-api.md#experimentalflags) |

## LiteRT and ML Drift

| Term | Meaning | Prefer / avoid | Why | Defined in |
| --- | --- | --- | --- | --- |
| LiteRT environment | `LiteRtEnvironmentT`: options, accelerator registry, buffer handlers, GPU environment; one per LiteRT-LM engine | avoid "context" | — | [litert/runtime.md](litert/runtime.md#the-environment) |
| accelerator / delegate | A registered `LiteRtAcceleratorDef` ("GPU Metal", "CpuAccelerator") / the TFLite delegate it creates per compiled model | — | — | [litert/runtime.md](litert/runtime.md#the-accelerator-registry) |
| GPU Metal accelerator | LiteRT's ML Drift Metal accelerator (`ml_drift_metal_accelerator`) | prefer the code's name "GPU Metal"; avoid "Metal delegate" for the accelerator | — | [litert/ml-drift-delegate.md](litert/ml-drift-delegate.md#linking-and-registration) |
| ML Drift delegate | The TFLite delegate in LiteRT's `ml_drift_delegate/` that runs graphs on ML Drift | avoid "Metal delegate" for the whole thing | Metal is one backend of it | [litert/ml-drift-delegate.md](litert/ml-drift-delegate.md#the-ml-drift-delegate-in-litert) |
| compiled model, hardware set | `CompiledModel`: an interpreter plus delegates for one hardware set / the CPU, GPU, NPU bits passed to compilation | prefer "compiled model" over "interpreter"; "hardware set" over "backend" in LiteRT | "backend" is LiteRT-LM's word | [litert/runtime.md](litert/runtime.md#creating-a-compiled-model) |
| partition | Connected supported nodes replaced by one delegate kernel | avoid "cluster" | — | [litert/ml-drift-delegate.md](litert/ml-drift-delegate.md#partitioning-which-nodes-run-on-the-gpu) |
| composite (`odml.*`) | A StableHLO composite op the delegate lowers itself instead of its decomposition | avoid "custom op" | TFLite custom ops are different | [litert/ml-drift-delegate.md](litert/ml-drift-delegate.md#composites-in-the-gemma-graphs) |
| bounded attention | Our pass bounding the attention select and softmax to the filled KV length ([F-110](facts.md#f-110)) | mark "(ours)"; avoid "flash attention" | — | [litert/ml-drift-delegate.md](litert/ml-drift-delegate.md#bounded-attention-ours-ede28c64-31386317) |
| program cache | `…_mldrift_program_cache.bin`: serialized inference contexts; Metal shaders are still compiled from source at load | avoid "shader cache", "compiled shader cache" | — | [litert/ml-drift-delegate.md](litert/ml-drift-delegate.md#program-cache) |
| weight cache, content key | `…_mldrift_weight_cache.bin`: GPU-format weights / our key from delegated nodes and constants, shared across context sizes ([F-081](facts.md#f-081)) | avoid "model cache" | — | [litert/ml-drift-delegate.md](litert/ml-drift-delegate.md#weight-cache) |
| XNNPACK weight cache | A file of XNNPACK-packed weights for the CPU path | distinct from ML Drift's weight cache | — | [litert/cpu-xnnpack.md](litert/cpu-xnnpack.md#the-xnnpack-weight-cache) |
| KleidiAI | Arm's micro-kernel library XNNPACK uses on Arm64 | avoid calling it a delegate | — | [litert/cpu-xnnpack.md](litert/cpu-xnnpack.md#kleidiai) |
| dequant-BMM rewrite | XNNPACK's rewrite of `dequantize(INT8) → batch matmul` into a channelwise convert; the source of our XNNPACK patch ([F-085](facts.md#f-085)) | — | — | [litert/cpu-xnnpack.md](litert/cpu-xnnpack.md#our-xnnpack-patch) |
| ML Drift | Google's GPU inference engine behind LiteRT's GPU accelerator | avoid "the GPU delegate" for the engine itself | — | [ml-drift/architecture.md](ml-drift/architecture.md#where-ml-drift-sits) |
| `GraphFloat32`, `GpuModel`, `GPUOperation` | ML Drift's input graph / its compiled backend-neutral graph / one generated kernel | avoid "op" when the TFLite operation is meant | — | [ml-drift/architecture.md](ml-drift/architecture.md#concepts) |
| calculation precision | `kF32`, `kF16`, `kF32F16`; LiteRT-LM's default on the GPU is FP16 activations (`kF16`) | distinguish from a tensor's data type | the file's FLOAT32 tensors run as FP16 on the GPU ([F-098](facts.md#f-098)) | [ml-drift/architecture.md](ml-drift/architecture.md#precision) |
| `ConvAppleMPP` | ML Drift's convolution on `mpp::tensor_ops::matmul2d`; ML Drift family Apple10 and MSL 4.0 only | "the Neural Accelerator convolution" in prose | — | [ml-drift/kernels.md](ml-drift/kernels.md#convapplempp-the-gpu-neural-accelerators) |
| `ConvWaveMatrix` | ML Drift's convolution on 8 × 8 `simdgroup_matrix` tiles; Apple7-Apple9 in practice | "simdgroup-matrix convolution"; avoid "tensor cores" | — | [ml-drift/kernels.md](ml-drift/kernels.md#convwavematrix-simdgroup-matrices) |
| `FullyConnected` (`fc1x1_*`) | ML Drift's fully connected kernel for up to 16 rows on Apple at FP16, dequantizing INT2/INT4/INT8 weights in its loop | — | — | [ml-drift/kernels.md](ml-drift/kernels.md#the-quantized-fully-connected-kernel) |
| offset storage | INT2/INT4/INT8 values stored unsigned with +2/+8/+128 and dequantized with the offset | avoid calling the KV cache "uint8 data" | it is INT8 stored with an offset | [ml-drift/kernels.md](ml-drift/kernels.md#quantized-weights-and-the-kv-cache-bytes) |
| INT8 path (Apple10 prefill) | Weights converted to INT8, activations quantized per token to INT8, INT32 accumulation | avoid bare "W8A8" | the activation quantization is per token and dynamic | [ml-drift/kernels.md](ml-drift/kernels.md#weight-conversion-and-activation-quantization) |
| runtime check | `ConvRuntimeCheckDesc` / `SoftmaxRuntimeCheckDesc`: bounds read from `param_tensor` at run time | — | — | [ml-drift/kernels.md](ml-drift/kernels.md#the-runtime-parameters) |
| flush period | Kernels per command buffer (256 in LiteRT's dispatch) ([F-094](facts.md#f-094)) | — | — | [ml-drift/metal-backend.md](ml-drift/metal-backend.md#encoding-and-command-buffers) |
| residency set | `MTLResidencySet` the backend fills; only requested when enabled (not in Agello) ([F-093](facts.md#f-093)) | — | — | [ml-drift/metal-backend.md](ml-drift/metal-backend.md#residency-sets) |
| ICB | Indirect command buffer; present in ML Drift's Metal code but disabled | spell out on first use | — | [ml-drift/metal-backend.md](ml-drift/metal-backend.md#indirect-command-buffers) |

## Apple platform and testing

| Term | Meaning | Prefer / avoid | Why | Defined in |
| --- | --- | --- | --- | --- |
| the iPhone app on the Mac | The unmodified iOS app on an Apple silicon Mac ("Designed for iPad/iPhone", `isiOSAppOnMac`), using the iOS slice and the same Metal path | avoid "Catalyst" | Catalyst is a different build | [apple-platform.md](apple-platform.md#the-iphone-app-on-a-mac) |
| simulation switch, simulated family | `AGELLO_SIMULATE_APPLE_GPU` (ours): another Apple GPU's kernel choices on this device ([F-095](facts.md#f-095)) | prefer "simulated family"; avoid "emulated GPU", "emulation" | speed, memory and arithmetic stay the real GPU's | [apple-platform.md](apple-platform.md#the-gpu-simulation-switch) |
| storage mode | `MTLStorageMode.shared` (CPU and GPU), `.private` (GPU only) | — | — | [apple-platform.md](apple-platform.md#concepts) |
| argument buffer | A Metal buffer holding references to a kernel's resources; tier 2 from Apple6 | — | — | [apple-platform.md](apple-platform.md#argument-buffers) |
| no simulator path | Agello's rule: no builds or tests in the simulator; our framework has no simulator slice ([F-010](facts.md#f-010)) | say "our framework has no simulator slice"; avoid "the simulator has no Metal path for LiteRT-LM" | Google's binary has a simulator slice with the Metal accelerator | [apple-platform.md](apple-platform.md#no-simulator-path) |

## Where in the code

Terms named after code symbols are cited on their owning pages; this page carries no code citations.

## Open questions

- "Filled length" and "allocated context" have no single field in Agello's UI; the app shows a token
  count and a KV setting. Whether user docs need both words is a call for the user guide.
