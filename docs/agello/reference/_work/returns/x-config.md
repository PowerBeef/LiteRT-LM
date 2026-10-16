# Returns: x-config (C1a, Run C): configuration.md

## Contradictions

- **Argument buffers for the Gemma decoder (new, not in C1–C17).** APL-051 (corrected) ends with
  "The main LLM executor's CreateCompilationOptions doesn't set the option ... so the Gemma decoder's
  graphs don't use argument buffers." The code says otherwise at both pins: `CreateCompilationOptions`
  calls `SetCommonGpuOptions(executor_settings, gpu_compilation_options, activation_data_type)`
  (`runtime/executor/llm_executor_settings_utils.cc:162-163` at 7b872544, same lines in v0.18.0), and
  `SetCommonGpuOptions` calls `SetUseMetalArgumentBuffers(true)` under `__APPLE__`
  (`litert_compiled_model_executor_utils.cc:1034`, v0.18.0 line 932). Google's 172d6806 (2026-09-09,
  "Disable Metal argument buffers.") removed only the *direct* call from `CreateCompilationOptions`;
  APL-051's verifier read lines 82-103 and missed the later call. RT-DLG-004, RT-DLG-071 and MD-MTL-022
  (and MD-MTL-017 being "wrong") agree with the code. New claim X-CFG-009 (tier A, conflicts_with
  APL-051). configuration.md follows X-CFG-009.

## Notes for other pages

- **apple-platform.md** (if X-CFG-009 verifies): lines 34, 309, 316-333 (Argument buffers section),
  486 and 530 say the decoder doesn't request argument buffers; they should say all executors do
  (dropped only on tier-1 devices), and APL-051's last sentence needs a correction.
- **swift-api.md / LM-SWIFT-057:** "a cache directory per context size" holds only with Google's
  binary. With a build that reports a build id (ours), Agello uses one folder, `kv-all-<build id>`,
  for every size, and keeps two size folders only for Google's binary (X-CFG-008).
- **lm-engine returns row "helpers: greedy":** only the transcription helper passes
  `SamplerConfig(topK: 1, topP: 1, temperature: 0)`; the JSON helpers, text helpers and the validation
  probe pass no sampler config (they get the engine default from the file, top-p k 1, also greedy)
  (X-CFG-007).
- **tools.md:** configuration.md's CLI section names `tools.md` in plain text because the page didn't
  exist when this ran; turn it into a link once it does.
- **troubleshooting.md:** configuration.md's "Limits and failure modes" table lists symptom →
  setting → cause rows with tags that can be reused.
- **facts.md / user docs:** don't describe the effect of Agello's temperature or top-k until a run
  checks LM-ENG-062 against Agello's validation probe (see Open questions).
- **models/embeddinggemma.md:** `embedding_litert_lm_main`'s `--activation_data_type` help says
  "Defaults to float32 on GPU", while the engine's own default on the GPU is FP16 (MOD-EMB-049); the
  CLI help text is the odd one out.
- **Rows left out of configuration.md** (in returns, no verified claim, low value on Apple):
  `LITERT_DISABLE_GPU/NPU/CPU`, `LITERT_NO_BUILTIN_OPS` defines; LiteRT runtime options "selected
  signature keys" and "error reporter mode"; `kLiteRtEnvOptionTagRuntimeLibraryDir` is folded into the
  accelerator-handle row. `litert_lm_engine_settings_set_audio_num_threads` (default 4) is listed
  without a claim.

## Open questions

- Agello's chat sampler under LM-ENG-062: the validation probe samples first on each engine, so plain
  decode steps would use the file default (greedy) rather than the chat's settings. Needs a run.
- Whether the decoder's kernels actually bind through argument buffers on device: no engine run in
  `generated/` records the binding mode.
- "Version added" is untraced for most C and C++ settings.
