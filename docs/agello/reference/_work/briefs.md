# Research briefs (Runs A and B)

Every researcher: read `_work/style.md` and your pages in `_work/outline.md` first; write only your
pages; cite with `tools/agello-docs/cite.py`; return claims, settings rows (for configuration.md),
terms (for glossary.md), contradiction resolutions and open questions. Paths below are relative to
the repository named in the heading of each list.

## Run A: code

### A1: LiteRT-LM engine, sessions, execution, executors
Pages: `litert-lm/engine-and-sessions.md`, `litert-lm/executors.md`.
Read (agello-next worktree): `runtime/engine/`, `runtime/core/` (`engine_advanced_impl.cc`,
`session_advanced.*`, `tasks.cc`, `cached_session.*`, `prefix_cache.*`, `session_utils.*`),
`runtime/framework/` (threadpool, worker threads, `resource_management/`), `runtime/executor/`
(compiled-model executor, factory, MTP drafter, vision, audio, embedding, `litert/state.*`, settings).
Ours: T2/T3 commits in `generated/patch-log.md`; `runtime/util/metal_raw_buffer.*`.
Leads: AUDIT §2.1–2.8, E4B-GRAPH, tracks T2/T3 (T3 superseded on the copy's lifetime: VALIDATION).
Google docs: `docs/api/cpp/conversation.md` (session parts).

### A2: LiteRT-LM conversation, components, file format
Pages: `litert-lm/conversation.md`, `litert-lm/components.md`, `litert-lm/file-format.md`.
Read: `runtime/conversation/`, `runtime/components/` (prompt template, tool_use, constrained_decoding,
samplers, stop tokens, model resources, lora, embedding_lookup, preprocessor), `support/tokenizer/`,
`support/preprocessor/`, `schema/`, `runtime/util/litert_lm_loader.*`, `runtime/proto/`,
`models/gemma4/`, `python/litert_lm_builder/`.
Evidence: `generated/proto-fields.md`, `generated/models/*` (sections, metadata text).
Leads: AUDIT §2.2, §2.4, §3.4; tracks T2, T7; TELEMETRY (sampling cost superseded by T7).
Google docs: `docs/api/cpp/conversation.md`, `constrained-decoding.md`, `tool-use.md`,
`docs/litert_lm_builder.md`, `models/README.md`, `models/gemma4/README.md`.

### A3: LiteRT-LM APIs, build, release, versions
Pages: `litert-lm/c-api.md`, `litert-lm/swift-api.md`, `litert-lm/build-and-release.md`,
`versions.md` (narrative), LiteRT-LM part of `other-platforms.md`.
Read: `c/`, `swift/` (and `swift/apple_fm/`), `Package.swift`, `.bazelrc`, `build_config/BUILD`,
`WORKSPACE`, `agello/`, `CMakeLists.txt` (mapped), `prebuilt/`, `kotlin/`, `python/`, `js/`, `omni/`
(mapped). For the shipped wrapper: `/Users/patricedery/Coding projects/LiteRT-LM` at `85b1a776`
(`git show 85b1a776:swift/...`).
Evidence: `generated/c-api-inventory.md`, `swift-api-inventory.md`, `version-matrix.md`,
`binaries.md`, `cli-flags.md`.
Leads: AUDIT §3–4, BUILD.md, Agello `Vendor/README.md`, the `agello-litert-update` skill (stale on
re-pins: C5), agello-bench README (C6).
Google docs: `docs/getting-started/*`, `samples/ios_and_mac/README.md`, LiteRT `litert/swift/README.md`.

### A4: LiteRT runtime and the ML Drift delegate
Pages: `litert/runtime.md`, `litert/ml-drift-delegate.md`, `litert/cpu-xnnpack.md`, LiteRT part of
`other-platforms.md`.
Read (LiteRT fork): `litert/c/`, `litert/cc/`, `litert/core/`, `litert/runtime/` (compiled_model,
tensor buffers, events, accelerator registry, `accelerators/gpu/`), `ml_drift_delegate/` (delegate,
composite, serialization caches, shared memory manager, tflite model builder), `tflite/delegates/xnnpack/`.
Ours: `generated/patch-log.md` (litert); the XNNPACK patch in the LiteRT-LM worktree (`agello/patches/`).
Leads: AUDIT §2.2, §3.3; tracks T4, T5, T11; TELEMETRY (delegation).
Google docs: `litert/runtime/README.md`, `litert/swift/README.md`, `docs/apis/*`,
`docs/instructions/LARGE_TFLITE_MODELS.md`, `ml_drift_delegate/delegate/serialization_weight_cache/GEMINI.md`,
`tflite/delegates/xnnpack/README.md`.

### A5: ML Drift
Pages: `ml-drift/architecture.md`, `ml-drift/kernels.md`, `ml-drift/metal-backend.md`, ML Drift part
of `other-platforms.md`.
Read (ML Drift fork): `ml_drift/common/` (gpu_info, model, operations, transformations, gpu_model*,
selectors, kernels used by `generated/kernels-by-family.md`, precision, task/), `ml_drift/metal/`.
Ours: `generated/patch-log.md` (ml-drift).
Evidence: `generated/kernels-by-family.md` (the selector narrative must match it), `engine-run.md`.
Leads: AUDIT §8, tracks T4, VALIDATION (A17 paths, ConvWaveMatrix window).
Google docs: `README.md`, `ml_drift/samples/llm/README.md`.

## Run B: models, Apple, our changes

### B1: Gemma 4 E4B and E2B
Pages: `models/gemma4-e4b.md`, `models/gemma4-e2b.md`.
Evidence first: `generated/hf/google__gemma-4-E4B-it/`, `google__gemma-4-E2B-it/` (config, processor,
generation config, chat template), `generated/models/e4b-ours.md`, `e4b-google.md`, `e2b-google.md`,
`e2b-gpu.md`, `e4b-gpu.md`, `generated/hf-configs.md`.
Web: the model cards (google/gemma-4-E4B-it, E2B-it; litert-community/gemma-4-E4B-it-litert-lm and
E2B), Google's Gemma 4 docs on ai.google.dev, the Gemma 3n docs (MatFormer, per-layer embeddings, KV
sharing) and any Gemma 4 technical report; cite versions.
Run A pages (verified claims) for how the runtime uses each section.
Leads: E4B-GRAPH, AUDIT §5, `research/3913` tag, `tools/model-surgery/patch_rope.py`, LiteRT-LM
issue #3913 thread.

### B2: Gemma 4 family, EmbeddingGemma, tokenizer and chat format
Pages: `models/gemma4-family.md`, `models/embeddinggemma.md`, `models/tokenizer-and-chat-format.md`.
Evidence: `generated/hf-configs.md`, `generated/hf/*` (all repos, including the "assistant" ones and
`google__embeddinggemma-2`), `generated/models/embeddinggemma.md`, the metadata text in the model pages.
Web: model cards, EmbeddingGemma paper and docs, Matryoshka representation learning (cite the paper
version), the MTP drafter blog if it explains the "assistant" models.
Code: `models/gemma4/`, `models/embeddinggemma2/`, the tokenizer components (from A2's page).

### B3: Apple platform
Page: `apple-platform.md`; Apple inputs for `memory.md`.
Evidence: `generated/kernels-by-family.md`, `engine-run.md`, `binaries.md`; Agello's
`Agello/Inference/DeviceProfile.swift`, `Agello.entitlements`, `docs/ENGINEERING.md`;
`docs/agello/results/iphone/` reports (device facts: app limit, GPU family).
Apple docs: via the `axiom-apple-docs` skill files or sosumi (Metal feature set tables, `MTLGPUFamily`,
`os_proc_available_memory`, the increased-memory-limit and extended-virtual-addressing entitlements,
residency sets, Metal 4).
Code: ML Drift `gpu_info.cc`, LiteRT Metal accelerator and buffer handler.

### B4: our changes and process
Pages: `patches.md`, `contributing.md`. (The dossier on C1–C17, `_work/contradictions.md`, moved to C2 in Run C, so it can use every run's resolutions.)
Evidence: `generated/patch-log.md`, `docs/agello/tracks/`, `upstream/`, `VALIDATION.md`, `PARITY.md`,
`results/`; CONTRIBUTING.md and GOVERNANCE.md of each upstream; issue #3913 and our posted comments
(`gh api repos/google-ai-edge/LiteRT-LM/issues/3913/comments`, read only); Agello's `Vendor/README.md`,
`NOTICE`, the `agello-litert-update` skill; git logs of the three forks.
