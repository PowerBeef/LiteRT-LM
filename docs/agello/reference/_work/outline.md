# Outline of docs/agello/reference

Each page lists its owner (agent ID from the plan), its sections, the questions it must answer, and
the contradictions (C1–C17, at the end) it must resolve. Pages follow `_work/style.md`.

## Inputs every writer gets

- `_work/style.md`, `tools/agello-docs/pins.json`, `tools/agello-docs/cite.py`.
- Generated evidence in `generated/`:
  - `c-api-inventory.md` (220 functions), `swift-api-inventory.md` (504 symbols), `proto-fields.md`
  - `cli-flags.md` (94 flags), `patch-log.md` (our commits per fork), `version-matrix.md`
  - `binaries.md` (Google's v0.18.0 CLiteRTLM vs ours)
  - `models/*.md|json` (E4B ours and Google's, E2B, E2B/E4B `-gpu`, EmbeddingGemma 2 text 270m: sections, signatures, ops, composites, weights by type, per-layer attention)
  - `hf-configs.md` and `hf/` (official configs: E2B, E4B, 12B, 26B-A4B, 31B, the 12B and 31B "assistant" models, EmbeddingGemma 2)
  - `kernels-by-family.md` (kernels ML Drift picks for E4B on Apple10 / 9 / 8 / 7)
  - `engine-run.md` (one engine run: facts, counters, engine log lines)
- Existing notes to mine (leads, not sources; check against code): `docs/agello/AUDIT.md` (line
  numbers are v0.18.0), `E4B-GRAPH.md`, `TELEMETRY.md`, `BUILD.md`, `PARITY.md`, `VALIDATION.md`,
  `tracks/*.md`, `upstream/*.md`; Google's docs in each repo (listed per area below); Agello's
  `docs/*.md` and `.claude/skills/*` in `/Users/patricedery/Coding projects/iRT`.

## Pages

### README.md (C1, written in Run C)
Reading paths (app developer, fork developer, upstream reporter, writer of user docs); the page map;
conventions (pins, citations, claim tags, fact IDs); how to regenerate and re-verify.

### stack.md (C1)
Sections: the layers (Agello → Swift `LiteRTLM` → C API → `Conversation` → `SessionAdvanced` →
execution manager → `Tasks` → executor → LiteRT `CompiledModel` → ML Drift delegate → Metal); one chat
turn traced hop by hop (from A1–A5's verified call paths, re-verified); threads (execution and callback
pools, Swift actors, the Metal queue); who owns what (engine, sessions, KV state, buffers); the open and
closed parts (`generated/binaries.md`: what the shipped binary has that ours doesn't).
Questions: Which thread runs the decode loop and which delivers chunks to Swift? What is copied
between layers per token? What does the shipped binary run that the documented build doesn't (closed
sampler, FST provider, artisan executors)?

### memory.md (C1, Apple inputs from B3)
Sections: where every byte lives for E4B and E2B at 4K/8K/16K/32K (weights mapped vs resident;
embedder and per-layer embedder; vision/audio encoders; KV formula; activations; MTP drafter);
engine caches on disk (program cache, content-keyed weight cache); iOS limits (jetsam, footprint,
`os_proc_available_memory`, the increased-memory-limit and extended-virtual-addressing entitlements);
what Agello measured (with conditions).
Questions: KV bytes per token for E4B and E2B from the graphs (24 INT8 caches for E4B; E2B's map)?
Why do 918 MB and 940 MB both appear for E4B at 32K? What does "app limit" mean on an iPhone 17 Pro
with and without the entitlement? Contradictions: C1, C9.

### configuration.md (C1, rows proposed by every writer)
One table per layer: Swift (`EngineConfig`, `ConversationConfig`, `SamplerConfig`, `ExperimentalFlags`),
C setters, `EngineSettings`/`SessionConfig`/`AdvancedSettings`, CLI flags, Bazel configs and defines
(`build_config/BUILD`, `--config=agello`), environment variables (`AGELLO_*`, LiteRT/ML Drift ones),
LiteRT environment and GPU options. Columns: name, default, where read (citation), effect on Apple,
version added, notes. Mark which ones Agello sets.

### versions.md (A3 narrative, generated matrix)
What ships vs what's documented; v0.18.0 → agello-next changes that matter (C API 0.2 → 1.0 with
status codes, logit mask runner, noop targets, ...); container versions (1.5 / 1.6 / 1.7 / writer 1.9);
how upstream versions and nightlies are tagged. Contradiction: C16.

### troubleshooting.md (C1)
`LiteRtLmStatusCode` and `LiteRTLMError` cases → causes and fixes; the failure modes Agello's
`agello-inference-rules` skill guards against (KV overflow, cancelled conversations, MTP with JSON,
validation of a loaded engine, ...), each traced to code.

### tools.md (C1)
`litertlm_peek`, `tools/model-surgery/inspect_e4b.py` and `patch_rope.py`, `model_info`, the CLI tools
and their flags, agello-bench, telemetry (levels, env vars, graph dumps), the GPU simulation switch,
`tools/agello-docs`. Exact commands.

### litert-lm/engine-and-sessions.md (A1)
Sections: `Engine`/`EngineFactory`/`EngineSettings` (creation flow in `EngineAdvancedImpl::Create`,
parallel loading, tokenizer future, environment); sessions (`SessionAdvanced`, `AudioSessionAdvanced`,
`SessionConfig`); execution managers (threaded vs serial, pools, task lifecycle, callbacks, cancellation,
timeouts); `ResourceManager` (locked executors, context handlers, KV context switching, snapshots,
clone); prefix cache and `cached_session`; checkpoints and rewind; `Tasks::Prefill/Decode/Score` and
`DecodeOneStep` (stop tokens, detokenizer, constraints, thinking budget, KV-full); benchmark info.
Questions: What exactly happens when two conversations share one engine? What does a context switch
copy, and where (GPU vs host)? What ends a decode (stop token, max tokens, KV full, cancel)? Where are
timeouts? What did our T2/T3 commits change? Contradiction: C11.

### litert-lm/executors.md (A1)
Sections: `LlmExecutor` interfaces; `LlmLiteRtCompiledModelExecutorStatic` vs `Dynamic` (and NPU,
mapped); signatures (prefill_1024/prefill_128/decode/verify) and chunking; magic numbers (32003 →
window) and how they're resolved; inputs (embeddings, per-layer embeddings, input_pos, mask,
param_tensor) and who fills them; KV state (`LitertState` policies, ping-pong, in-place, raw Metal
snapshots); sampling hand-off (internal vs external sampler, GPU sampler input handling); MTP
(drafter loading, draft/verify loop, acceptance, stats); vision and audio executors (encoder, adapter,
end-of-media, CPU constraints); embedding executor (EmbeddingGemma path); executor settings.
Questions: How is a prompt longer than 1024 tokens split? What does `param_tensor` carry? How many
tokens does MTP draft and how is acceptance decided? Why do the vision adapter and audio encoder run on
the CPU? What runs on the CPU for every token (embedding lookups)?

### litert-lm/conversation.md (A2)
Sections: `Conversation` and `ConversationConfig`; messages (JSON format, roles, content parts);
templates (Jinja via minijinja, preface, `PromptTemplate`, Gemma 4 E2B/E4B template); model data
processors (registry, `Gemma4DataProcessor`, multimodal assembly); channels and thinking (filtering
from the KV cache, `ThinkingConfig`); tool calling (declarations, parsers, streaming, the Swift
automatic loop); constraints (llguidance JSON/regex/Lark, external, FST and the closed Gemma provider,
what `LITERT_LM_FST_CONSTRAINTS_DISABLED` changes, ours); history and rewind.
Questions: What exactly is prefilled for a new user turn (whole history or the new turn)? How does
thinking content get removed from the KV cache? Which constraint path does a JSON response format take
in our build vs the shipped one?

### litert-lm/components.md (A2)
Sections: tokenizers (SentencePiece vs HF, streaming detokenizer); samplers (CPU `TopPSampler`, greedy
FP16 path, the closed Metal TopK sampler and how it's loaded, fallback); logit masks (bitmap/sparse,
device runner); stop tokens; constraints internals (composite, thinking budget, repetition penalty,
no-repeat n-gram, suppress tokens); model resources (`ModelResourcesLitertLm`, loader, sections);
LoRA; embedding lookup (text, multimodal, end-of-media); preprocessors (image, audio).
Contradictions: C10, C13.

### litert-lm/file-format.md (A2)
Sections: the container (magic, version, header flatbuffer, 16 KiB alignment, sections and items);
section types and model_type strings; metadata protos (`LlmMetadata`, `ExecutorMetadata`,
`EmbeddingMetadata`; link `generated/proto-fields.md`); how the loader maps sections and picks one by
backend constraint/activation hint; external weights (after the graph, or `TFLiteWeights` sections);
`ModelInfo`; the builder and peek tools; the variants (`-gpu` artisan files, web `.task`).
Questions: How does the runtime choose between duplicate sections? Which container versions does the
reader accept?

### litert-lm/c-api.md (A3)
Conventions (status codes, out-params, ownership, threading, callbacks, JSON strings), object
lifecycles, the API by area with the important semantics (not a copy of the inventory: link
`generated/c-api-inventory.md`), error reporting, version macros, changes from 0.2 (v0.18.0).

### litert-lm/swift-api.md (A3)
Actors and threading (`Engine` actor, `Conversation` Sendable class), lifecycle, streaming
(`AsyncThrowingStream`, `StreamContext`, our weak-reference fix), tools (`Tool`, `@ToolParam`,
automatic calling, our error-response patch), response formats, `ExperimentalFlags` (process-wide,
when they're read), errors, `ModelInfo`, `EmbeddingEngine`, `BuildInfo`/`Telemetry` (ours), the
`apple_fm` Foundation Models adapter (and that it doesn't build against the macOS 27 SDK at the pin).
Differences between the shipped wrapper (`85b1a776`) and agello-next.

### litert-lm/build-and-release.md (A3)
Bazel (version, configs, feature flags, `--config=agello`, static GPU accelerator linking, WORKSPACE
pins and overrides, local LiteRT/ML Drift), our xcframework (`agello/build-xcframework.sh`, minimum
OS 27, slices, build info stamping), release script, third-party notices, privacy manifest, CMake (mapped),
upstream's release artifacts and nightlies, `prebuilt/` binaries. Contradiction: C5, C6.

### litert/runtime.md (A4)
Environment and options; `LiteRtModelT` loading; `CompiledModel` creation (magic numbers, delegates,
partitioning, CPU fallback) and runs (buffer registration, async, events); tensor buffers (types,
Metal 30–34, custom handlers, locking, registry); events; options (CPU/GPU/runtime); accelerator
registry (static vs dlopen, auto-registration); delegation metrics; the C/C++ API layers.

### litert/ml-drift-delegate.md (A4)
Delegate creation from GPU options; `DelegatePrepare` and partitioning; `DelegateKernel` init (TFLite →
GraphFloat32, custom transformations including ours: bounded attention and the window, SDPA/QKV
fusions), composites (`odml.*` parsers), op selector; program cache and weight cache (keys, files,
our content key); `GpuBackendMetal`/`...Litert` (dispatch, flush period, waits, residency sets,
argument buffers); Metal buffer handler (our direct copies), KV cache copies; per-run `Invoke` steps.
Questions: What decides whether an op runs on the GPU or falls back to the CPU? What is cached on
disk, keyed by what? Contradiction: C12 (with A5).

### litert/cpu-xnnpack.md (A4)
The CPU path for LLMs (when it's used, CPU executor settings), XNNPACK and KleidiAI, the weight cache,
our XNNPACK patch (qs8→qc8 scales), which E4B sections always run on the CPU and why.

### ml-drift/architecture.md (A5)
Concepts (GpuModel, tensors, storage types, precision), GPU detection (`GpuInfo`, Apple families
table, our simulation switch), TFLite → GraphFloat32 → transformations → GpuModel (builder, merge,
memory planning) → backend; selectors.

### ml-drift/kernels.md (A5)
Only the kernels the Gemma graphs use (`generated/kernels-by-family.md`): quantized FC (int2/4/8,
weight conversion), ConvAppleMPP (Neural Accelerators, Apple10), ConvWaveMatrix (simdgroup matrix,
Apple9 and below in practice), ConvGeneric, runtime batched matmul (attention), softmax, RMS norm,
RoPE (FP32 angles from INT32), embedding lookup, cache update, ours (bounded attention, window). How
selection depends on the GPU family, precision and shapes. Contradiction: C12.

### ml-drift/metal-backend.md (A5)
`metal::Environment`, `InferenceContext` (init, memory, compile, tune, serialization), `ComputeTask`,
runtime MSL compilation (no binary archives), argument buffers, encoding and command buffers, ICBs
(disabled), residency sets, our timing/graph-dump hooks.

### models/gemma4-family.md (B2)
The family from the official configs: E2B, E4B, 12B, 26B-A4B (MoE), 31B, the 12B/31B "assistant"
models (what they are), base vs instruction-tuned; shared design (local/global attention, head dims,
RoPE, per-layer embeddings on E models, KV sharing); modalities; trained context; which ones have
LiteRT-LM files and which run on a phone.

### models/gemma4-e4b.md (B1)
Architecture (config): layers, widths, heads, head dims, local/global pattern and window, RoPE, norms,
softcaps, per-layer embeddings, KV sharing, MatFormer/"effective parameters" if documented. The export
(file): sections, signatures and their ops, the per-layer attention table, KV caches and sizes,
quantization by section, the MTP drafter graph, vision (encoder, 70/140/280 signatures, adapter on
CPU, patch metadata), audio (encoder on CPU, adapter, front end), the embedder and per-layer embedder.
Trained vs exported context. Rotary positions and #3913 (our 512-token block file and why).
Contradictions: C2, C9, C14.

### models/gemma4-e2b.md (B1)
Same structure for E2B (35 layers, 1 KV head, KV sharing from layer 15, INT2 in prefill/decode,
inline weights), and the differences from E4B; the `-gpu` artisan variants (what's in them, why Agello
doesn't use them).

### models/embeddinggemma.md (B2)
EmbeddingGemma 2 (the variants; text 270m in depth): architecture, the file (embedder, text encoder,
signatures 128–8192), pooling/normalization, Matryoshka dimensions, how Agello uses it.

### models/tokenizer-and-chat-format.md (B2)
The tokenizer (SentencePiece, vocabulary 262,144, special tokens and IDs from the files), the chat
template (turns, system, thinking channel, tool calls and responses, images and audio markers), stop
tokens, how LiteRT-LM applies it.

### apple-platform.md (B3)
Devices → GPU families → what ML Drift does on each (with `kernels-by-family.md`); memory limits and
entitlements; unified memory and storage modes; residency sets; Metal 4 / tensor ops status in this
stack; the GPU simulation switch and its limits; "Designed for iPad/iPhone" on the Mac; no simulator path.
Contradiction: C1.

### other-platforms.md (A3 for LiteRT-LM parts, A4 for LiteRT, A5 for ML Drift; merged in Run C)
Android (OpenCL/GL, NPU dispatch, Kotlin), WebGPU (and the macOS WebGPU accelerator Google's macOS
release uses), Linux/Windows, Python, JS, omni (ASR/TTS/text-to-image): one paragraph to one table
each, enough to read shared code and upstream reports.

### patches.md (B4)
Every commit of ours (generated/patch-log.md) grouped by purpose: what, why, evidence (results files),
upstream status (draft, posted, merged), when to drop it. Contradictions: C3, C4, C8.

### contributing.md (B4)
How each upstream takes contributions (LiteRT-LM issues only; LiteRT and ML Drift pull requests with
the CLA), templates (issue, PR, our commit messages), evidence rules for upstream text, the posting
rule (the user's OK), a checklist for user-facing docs. Contradiction: C7 (test counts belong to
Agello's docs; record the rule).

### facts.md, glossary.md (C2), maintenance.md (C1)

## Contradictions to resolve (each becomes a fact with a written resolution)

| ID | Topic | Values seen | Owner |
| --- | --- | --- | --- |
| C1 | App memory limit, iPhone 17 Pro | 8.6 GB (Agello ENGINEERING.md, harness device report) vs 6.4 GB (VALIDATION.md, archived plan) | B3, memory.md |
| C2 | Vision tokens per image | 70/140/280 (AUDIT; file signatures `vision_70/140/280`), 140/280/560 (USER_GUIDE), 70–1120 (Agello `ChatTypes.swift`), max 2520 patches (metadata) | B1 |
| C3 | Patches in the shipped wrapper | one (Agello CLAUDE.md) vs two (Vendor/README, NOTICE, skill) | B4 |
| C4 | Fork landing page `.github/README.md` | one patch, no agello-next, `research/3913` a branch (now a tag), "Metal accelerator closed" (open since 2026-10-08) | B4 |
| C5 | Re-pinned builds | "byte-identical" (agello-litert-update skill) vs "not byte-identical; compare replies" (BUILD.md) | A3 |
| C6 | agello-bench README | "no macOS Metal slice yet" (we have one), suites list incomplete, `research/3913` a branch | A3 |
| C7 | Test counts | 220 unit + 48 device + 23 UI (README, TESTING.md) vs 227 unit, 22 UI (VALIDATION) | B4 (rule), Run D |
| C8 | Upstream drafts | "3 issues, 3 PRs" (PLAN-MAC) vs 4 issues, 3 LiteRT PRs, 1 ML Drift PR (upstream/README) | B4 |
| C9 | E4B KV cache at 32K | 918 MB (AUDIT) vs 940 MB (E4B-GRAPH, T4): 32,000 vs 32,768 positions? MB vs MiB? | B1, memory.md |
| C10 | CPU sampling cost | 0.50 ms (TELEMETRY) vs 0.044 ms greedy (T7, after T5) | A2 |
| C11 | Kept conversation's KV copy | lives until the next message (T3) vs freed when the helper ends (VALIDATION, Agello `6d389e6`) | A1 |
| C12 | Window-aware kernels | FC, softmax, ConvAppleMPP (T4) vs also ConvWaveMatrix (ML Drift `15fa9e2`) | A5 |
| C13 | FP16 logits | "only with FP16 precision and the GPU sampler" (AUDIT, v0.18.0) vs greedy over FP16 logits in the CPU sampler (T7, ours) | A2 |
| C14 | Context length | 128K (archived plan) vs 32K (export) — config says `max_position_embeddings` 131072 | B1 |
| C15 | AUDIT line numbers | v0.18.0 (`agello` branch) while the work is on agello-next | Run D (banner) |
| C16 | Performance figures | README table is Google's v0.18.0 binary; VALIDATION's newer figures are our build | A3 (versions), C2 |
| C17 | Model revision in AUDIT | cites `13d5aa4c` / `4a691f03` (2048-block, deleted) vs current `a1693e22` / `6353a233` (512-block) | B4, Run D |
