# Troubleshooting

> **Scope:** symptoms → causes → fixes for LiteRT-LM on Apple as Agello uses it: the C API's
> `LiteRtLmStatusCode` (1.0.0) and the Swift wrapper's `LiteRTLMError`; the failure modes Agello's
> `agello-inference-rules` skill guards against, each traced to the code that explains it; and the
> failure modes Runs A and B found while documenting the stack. Each entry says which build it
> applies to.
> **Out of scope:** the mechanisms in depth (each entry links the page that owns it); iOS memory
> accounting ([apple-platform.md](apple-platform.md), [memory.md](memory.md)); building the runtime
> ([litert-lm/build-and-release.md](litert-lm/build-and-release.md)).
> **Pinned to:** LiteRT-LM `7b872544`, LiteRT `362759838`, ML Drift `15fa9e2`; what Agello ships:
> LiteRT-LM `85b1a776` with Google's prebuilt v0.18.0 `CLiteRTLM.xcframework`; Agello `93122b647a`.
> **Applies to:** Apple (iOS, macOS). **Last verified:** 2026-10-15

## Summary

- **Read the message, not only the code.** Every failing C call returns a status code and records a
  per-thread message (`"INVALID_ARGUMENT: ..."`); the Swift wrapper folds both into one string,
  `"<function> failed with status <code>: <message>"`. Streams report failures as an error string in
  their last chunk.
- **Two states end a session's chain of work silently:** a cancel and a full KV cache. Later tasks
  chained to the ended one never run. Agello treats both as "start a new conversation from saved
  history".
- **Most "it loaded but nothing works" reports** are a model section that failed to map: the load
  succeeds and every prefill then fails. Agello validates each new engine with a one-token reply.
- **Wrong digits in long replies** with Google's E4B or E2B file at FP16 are #3913; our 512-token
  rotary-block file fixes them.
- **Memory:** 8 GiB is the iPhone 17 Pro's limit (an early reading can say 6 GiB), the first image
  costs about 1.3 GB, and a CPU engine needs our XNNPACK patch in a source build.

## Concepts

| Term | Meaning here |
| --- | --- |
| Status code | `LiteRtLmStatusCode`, 0 on success, 1–16 on failure, numerically equal to `absl::StatusCode`. |
| Last error | The calling thread's message for its most recent failed C call; never cleared by a success. |
| Task state | Where a queued prefill or decode stands: `kDone`, `kMaxNumTokensReached`, `kCancelled`, `kDependentTaskCancelled`, `kFailed`, ... ([litert-lm/engine-and-sessions.md](litert-lm/engine-and-sessions.md#task-lifecycle)). |
| Chained task | A session's next task waits for its previous one; it inherits a cancelled or full end state instead of running. |
| Documented build | agello-next (`7b872544` and the LiteRT and ML Drift pins). |
| Shipped | What Agello runs: the Swift wrapper at `85b1a776` over Google's prebuilt v0.18.0 binary. |

## How it works

### Reading an error

1. **C API 1.0.0.** A failing function returns a non-zero `LiteRtLmStatusCode`; pointer
   out-parameters are set to NULL, scalars left untouched; a NULL out-parameter is
   `kLiteRtLmStatusInvalidArgument`. <!-- c:LM-CAPI-004 --> The message is per thread and survives
   later successes, so read `litert_lm_get_last_error_message()` right after the failing call, on
   the same thread. <!-- c:LM-CAPI-007 --> A failure that comes from C++ is recorded as
   `absl::Status::ToString()` (`"NOT_FOUND: ..."`); one the C layer detects itself records only its
   text. <!-- c:LM-CAPI-008 -->
2. **Swift (documented build).** `LiteRTLMError.check` turns a non-OK status into the caller's error
   with the text `"<function> failed with status <code>: <message>"`, reading and clearing the
   thread's last error. <!-- c:LM-SWIFT-037 -->
3. **Shipped wrapper.** C API 0.2: `litert_lm_engine_create` returns NULL on failure, the stream call
   returns an `int`, and the code is read with `litert_lm_get_last_error_code()`, which the wrapper
   exposes as `LiteRTLMError.getLastErrorCode()`. <!-- c:LM-CAPI-037 --> <!-- c:LM-SWIFT-039 -->
4. **Streams.** The conversation stream turns a failed status into a last chunk whose error is
   `status.ToString()`, and a cancelled task into `"CANCELLED: Task cancelled"`.
   <!-- c:LM-CAPI-025 --> <!-- c:LM-CAPI-049 --> The session stream says `"Max number of tokens
   reached."` for a full KV cache and `"CANCELLED."` for a cancel ([`c/engine.cc:70-92`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/c/engine.cc#L70-L92)).
   <!-- c:LM-CAPI-026 --> The Swift wrapper throws any error chunk as
   `conversation(.invalidResponse(<text>))`. <!-- c:LM-SWIFT-016 -->

### `LiteRtLmStatusCode` (C API 1.0.0)

The enum has 17 values, 0 to 16, numerically identical to `absl::StatusCode`, frozen; treat unknown
values as `kLiteRtLmStatusUnknown` ([`c/error_reporter.h:140-158`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/c/error_reporter.h#L140-L158)).
<!-- c:LM-CAPI-005 --> Any C++ status outside 0–16 becomes `kLiteRtLmStatusUnknown`.
<!-- c:LM-CAPI-006 --> The same enum exists in the shipped v0.18.0 header, but its functions mostly
return pointers and `int`s instead.

| Code | Name | Seen in this stack when | What to do |
| --- | --- | --- | --- |
| 0 | `kLiteRtLmStatusOk` | success | |
| 1 | `kLiteRtLmStatusCancelled` | a task was cancelled, or was chained to a cancelled one ([cancel](#the-next-message-fails-with-cancelled-after-a-cancel)) | create a new conversation |
| 2 | `kLiteRtLmStatusUnknown` | a status outside 0–16 | read the message |
| 3 | `kLiteRtLmStatusInvalidArgument` | NULL arguments; unknown backend strings; unparsable message JSON; prompt longer than the context; visual budget above 280; unknown section type; backend constraint mismatch; text LoRA; a cache directory that isn't writable | see the entries below |
| 4 | `kLiteRtLmStatusDeadlineExceeded` | a wait on a session ran past `Engine::kDefaultTimeout` (10 minutes) | look for a callback that never returned ([stall](#a-new-conversation-takes-about-600-s-to-answer)) |
| 5 | `kLiteRtLmStatusNotFound` | no engine for the backend (`-gpu` files); "No per-layer embeddings found in inputs."; absent model-info values | see the entries below |
| 6 | `kLiteRtLmStatusAlreadyExists` | not raised by LiteRT-LM's own code paths described here | read the message |
| 7 | `kLiteRtLmStatusPermissionDenied` | not raised by LiteRT-LM's own code paths described here | |
| 8 | `kLiteRtLmStatusResourceExhausted` | a full KV cache, only with the conversation option `return_error_on_max_tokens_reached` (not settable from C or Swift) | |
| 9 | `kLiteRtLmStatusFailedPrecondition` | a prompt that doesn't fit the remaining KV cache (executor check); in v0.18.0, constrained decoding in a build without the FST provider | see [KV full](#the-reply-stops-or-the-next-message-answers-nothing) |
| 10 | `kLiteRtLmStatusAborted` | not raised by LiteRT-LM's own code paths described here | |
| 11 | `kLiteRtLmStatusOutOfRange` | a response or batch index past the end ([litert-lm/c-api.md](litert-lm/c-api.md#status-codes-out-parameters-and-the-last-error)) | |
| 12 | `kLiteRtLmStatusUnimplemented` | a container of another major version; `TOP_K` or `GREEDY` on the CPU sampler; an unregistered data processor | use `TOP_P` with k 1 for greedy |
| 13 | `kLiteRtLmStatusInternal` | a template whose new render doesn't start with the previous one; a failed task; executor errors | read the message |
| 14 | `kLiteRtLmStatusUnavailable` | not surfaced: the TopK Metal sampler's `Unavailable` (our build has none) makes the factory fall back to the CPU | |
| 15 | `kLiteRtLmStatusDataLoss` | not surfaced: a program cache entry that fails to deserialize is rebuilt | |
| 16 | `kLiteRtLmStatusUnauthenticated` | not raised by LiteRT-LM's own code paths described here | |

"Not raised by the code paths described here" means no entry on this page leads to it; another
component's status passes through unchanged. Sources for the rows: the entries below and
[litert-lm/c-api.md](litert-lm/c-api.md#status-codes-out-parameters-and-the-last-error).

### `LiteRTLMError` (Swift)

Seven top-level cases, each wrapping a nested enum ([`swift/LiteRTLMError.swift:19-27`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/LiteRTLMError.swift#L19-L27)).
<!-- c:LM-SWIFT-036 --> The full list is in
[generated/swift-api-inventory.md](generated/swift-api-inventory.md); the cases you meet:

| Case | Cause | What to do |
| --- | --- | --- |
| `engine(.alreadyInitialized)` | `initialize()` called twice on one `Engine` <!-- c:LM-SWIFT-001 --> | create a new `Engine` |
| `engine(.notInitialized)` | `createConversation` before `initialize()` | initialize first |
| `engine(.failedToCreateSettings(_))`, `engine(.failedToCreateEngine(_))` | a C call failed during `initialize()`; the string carries the C message | see [Loading an engine](#loading-an-engine) |
| `engine(.failedToCreateConversation(_))`, `.failedToCreateConversationConfig(_)` | the C conversation could not be created (for example a constraint provider failure) | read the message |
| `engine(.notOptedIntoExperimentalAPIs)` | `updateGPUEnableMetalResidencySet` without `ExperimentalFlags.optIntoExperimentalAPIs()` <!-- c:LM-SWIFT-055 --> | opt in first |
| `config(.invalidMaxNumTokens / .invalidTopK / .invalidTopP / .invalidTemperature)` | `maxNumTokens <= 0`; `topK <= 0`, `topP` outside [0, 1] or `temperature < 0` <!-- c:LM-SWIFT-053 --> | fix the value |
| `config(.multipleSystemMessages)` | both `systemMessage` and an initial system message, or two initial ones <!-- c:LM-SWIFT-006 --> | keep one |
| `conversation(.invalidResponse(_))` | a failed native send, an error chunk (including `CANCELLED: Task cancelled`), or a reply without content or channels | see [cancel](#the-next-message-fails-with-cancelled-after-a-cancel) and [KV full](#the-reply-stops-or-the-next-message-answers-nothing) |
| `conversation(.failedToStartStream(status:message:))` | `litert_lm_conversation_send_message_stream` refused to start; the only case that keeps the numeric status <!-- c:LM-SWIFT-014 --> | read the message |
| `conversation(.recurringToolCallLimitExceeded(limit: 25))` | the 25th reply of a `sendMessage` tool loop still calls tools <!-- c:LM-SWIFT-012 --> | answer without tools, or use the stream path |
| `conversation(.responseFormatNotEnabled)` | a `ResponseFormat` on a conversation created without `enableResponseFormat` <!-- c:LM-SWIFT-030 --> | create it with `enableResponseFormat: true` |
| `conversation(.notAlive)` | a stream's tool round ran after its conversation was released (our weak-reference patch) <!-- c:LM-SWIFT-021 --> | keep the conversation while it streams |
| `conversation(.benchmarkNotEnabled / .benchmarkInfoUnavailable)` | benchmark data asked for without `ExperimentalFlags.enableBenchmark` at call time, or at engine creation | set the flag before `initialize()` |
| `conversation(.toolExecutionError(name:error:))` | upstream's wrapper rethrew tool errors; with our patch nothing throws it <!-- c:LM-SWIFT-024 --> | |
| `tool(.notFound(name:))` | the model called an unregistered tool; our patch answers it to the model instead <!-- c:LM-SWIFT-023 --> | |
| `message(.invalidContent / .failedToConvertToJson)` | a JSON reply with no content, channels or tool calls | |
| `modelInfo(.failedToLoadModel(_))` | `ModelInfo(throwingModelPath:)` couldn't read the file | check the path; in the documented build the non-throwing accessors return fallbacks instead <!-- c:LM-SWIFT-040 --> |

### Loading an engine

#### "No available engine for backend" with a `-gpu` file

- **Symptom.** Engine creation fails with `NOT_FOUND: No available engine for backend: ...` (C API),
  or "Unsupported backend" from the executor factory.
- **Cause.** A file with a `tf_lite_artisan_text_decoder` section (the `-gpu` E2B and E4B files)
  makes `EngineSettings::CreateDefault` switch a GPU request to `GPU_ARTISAN`. Its preferred engines
  are the two legacy TFLite engines, and the only engine registered in the open-source build is the
  compiled-model one, so `EngineFactory::CreateDefault` fails. <!-- c:LM-ENG-002 -->
  <!-- c:MOD-E2B-056 --> <!-- c:LM-FMT-017 -->
- **Where.** [`runtime/engine/engine_factory.h:88-110`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_factory.h#L88-L110).
- **Do.** Use the regular `gemma-4-E*-it.litertlm` files; the `-gpu` files also lack vision, audio
  and the MTP drafter. <!-- c:MOD-E2B-058 -->
- **Applies to.** The documented build. Google's v0.18.0 binary contains the artisan executor's
  symbols; whether it runs a `-gpu` file on iOS hasn't been tried.

#### "Audio backend constraint mismatch"

- **Symptom.** Engine creation fails with `INVALID_ARGUMENT: Audio backend constraint mismatch...`.
- **Cause.** The Gemma 4 E files mark `tf_lite_audio_encoder_hw` `backend_constraint=cpu`, and the
  engine validates the main, vision-encoder and audio-encoder sections' constraints against the
  configured backends. <!-- c:LM-EXE-031 --> <!-- c:LM-FMT-012 -->
- **Where.** [`runtime/engine/engine_settings.cc:81-112`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L81-L112).
- **Do.** Give audio the CPU backend, as Agello does. <!-- c:LM-EXE-029 -->
- **Applies to.** Both.

#### "Unknown model type" or a file that won't open

- **Symptom.** `INVALID_ARGUMENT: Unknown model type: ...`, `UNIMPLEMENTED` on the header, or
  "Unsupported or unknown file format.".
- **Cause.** The loader accepts only known `model_type` strings <!-- c:LM-FMT-010 --> and only
  container major version 1 <!-- c:LM-FMT-002 -->; from a C API path the format is taken from the
  file extension, and an LLM engine accepts only `.litertlm` and `.task` files.
- **Do.** Check the file with `litertlm_peek` ([tools.md](tools.md)); a file made by a newer builder
  may need a newer runtime.
- **Applies to.** Both.

#### Engine creation fails with a vision budget above 280

- **Symptom.** Creating an E2B or E4B engine with `max_vision_tokens_per_image` (Swift:
  `ExperimentalFlags.visualTokenBudget`) above 280 fails.
- **Cause.** The files have vision signatures for 70, 140 and 280 tokens only; a caller-set value is
  kept as is, and above 280 no signature fits. <!-- c:MOD-E4B-050 -->
- **Do.** Leave it unset (280 is the default from the metadata) or use 70, 140 or 280.
- **Applies to.** Both.

#### "Some ops are not accelerated"

- **Symptom.** `CompiledModel` creation fails with `kLiteRtStatusErrorCompilation` "Some ops are not
  accelerated. Add kLiteRtHwAcceleratorCpu ...".
- **Cause.** LiteRT-LM compiles the LLM GPU-only; a node the ML Drift delegate doesn't take has no
  CPU fallback. <!-- c:RT-RUN-015 --> <!-- c:RT-DLG-011 --> The delegate's log line "Following
  operations are not supported by GPU delegate" names them.
- **Where.** [`litert/runtime/compiled_model.cc:1185-1198`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/litert/runtime/compiled_model.cc#L1185-L1198).
- **Do.** A model exported with an op or composite ML Drift doesn't know needs a re-export or a
  delegate change ([litert/ml-drift-delegate.md](litert/ml-drift-delegate.md#partitioning-which-nodes-run-on-the-gpu)).
- **Applies to.** The documented build; the same check exists upstream.

#### "Failed to create OpenCL context." in the log

- **Symptom.** The engine log shows this line on every GPU load on Apple.
- **Cause.** LiteRT logs it at INFO level whenever OpenCL support is compiled out, the default on
  Apple; Metal is created next. <!-- c:RT-RUN-010 -->
- **Where.** [`litert/runtime/gpu_environment.cc:436-443`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/gpu_environment.cc#L436-L443).
- **Do.** Nothing; it isn't an error.
- **Applies to.** Both.

#### "Failed to map, error: Cannot allocate memory"

- **Symptom.** A section fails to map when an engine is created, often while a previous engine's
  mappings are still alive.
- **Cause.** Each section is mapped with `mmap(MAP_PRIVATE)`; a failed map reports `strerror(errno)`,
  "Cannot allocate memory" for `ENOMEM`. <!-- c:APL-028 --> The extended-virtual-addressing
  entitlement gives apps that map large files a larger address space. <!-- c:APL-010 -->
- **Where.** [`support/util/memory_mapped_file_posix.cc:132-145`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/support/util/memory_mapped_file_posix.cc#L132-L145).
- **Do.** Declare the entitlement (Agello does <!-- c:APL-011 -->), release the old engine before the
  new one, and validate the new engine (next entry).
- **Applies to.** Both.

#### The engine loads, but every message fails: "No per-layer embeddings found in inputs."

- **Symptom.** Engine creation succeeds; every send then fails with `NOT_FOUND: No per-layer
  embeddings found in inputs.`.
- **Cause.** A section that fails to map is logged ("Failed to map section") and reported as absent;
  `ModelResources` then answers "not found" for the per-layer embedder, and
  `InitializeEmbeddingLookups` creates the per-layer lookup only when that model is found, so the
  failure is skipped and the engine is created without it. Every prefill of an E2B or E4B file then
  looks for per-layer embeddings in the inputs instead, finds none and fails.
  <!-- c:X-TRB-001 -->
- **Where.** [`runtime/util/litert_lm_loader.cc:404-410`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.cc#L404-L410),
  [`runtime/components/model_resources_litert_lm.cc:128-136`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/model_resources_litert_lm.cc#L128-L136),
  [`runtime/executor/litert_compiled_model_executor_utils.cc:1294-1322`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/litert_compiled_model_executor_utils.cc#L1294-L1322),
  [`runtime/executor/llm_litert_compiled_model_executor.cc:739-764`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_litert_compiled_model_executor.cc#L739-L764),
  [`runtime/executor/llm_litert_compiled_model_executor.cc:292-302`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_litert_compiled_model_executor.cc#L292-L302);
  v0.18.0: [`runtime/executor/litert_compiled_model_executor_utils.cc:1193-1220`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/runtime/executor/litert_compiled_model_executor_utils.cc#L1193-L1220),
  [`runtime/executor/llm_litert_compiled_model_executor.cc:285-297`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/runtime/executor/llm_litert_compiled_model_executor.cc#L285-L297).
- **Do.** Validate every new engine with a generation before using it. Agello sends "Hi" with
  `maxOutputTokens` 1 on a throwaway conversation <!-- c:LM-SWIFT-058 -->, and retries the same
  configuration once after 750 ms when that validation fails, to let the previous engine's mappings
  go. <!-- c:X-TRB-001 -->
- **Applies to.** Both.

#### "GPU sampler unavailable. Falling back to CPU sampling."

- **Symptom.** A warning at the first decode; telemetry records `sampler = cpu (GPU sampler
  unavailable)`.
- **Cause.** Our build doesn't link Google's closed Metal TopK sampler, so the sampler factory falls
  back to the CPU `TopPSampler`; the shipped binary contains the sampler's symbols.
  <!-- c:LM-CMP-018 --> <!-- c:LM-EXE-019 -->
- **Where.** [`runtime/components/sampler_factory.cc:764-790`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/components/sampler_factory.cc#L764-L790).
- **Do.** Nothing; each step reads the 524,288 bytes of FP16 logits back to the CPU
  ([litert-lm/components.md](litert-lm/components.md#sampling-cost-on-the-cpu)).
- **Applies to.** The documented build.

#### A file without `LlmMetadataProto` crashes the load

- **Symptom.** Creating an LLM engine from a `.litertlm` without a metadata section aborts instead
  of returning a status.
- **Cause.** `LitertLmLoader::GetLlmMetadata()` dereferences the optional section without checking.
  <!-- c:LM-FMT-038 -->
- **Do.** Check files with `litertlm_peek` before shipping them.
- **Applies to.** Both.

#### The model reports no thinking, no tools and no context

- **Symptom.** `ModelInfo` says `thinking=false tools=false maxContext=0` for E4B.
- **Cause.** The E files don't set `supports_thinking` or `supports_function_calling`, so both read
  false; both work. <!-- c:MOD-E2B-048 --> <!-- c:MOD-E4B-052 --> Without `max_num_tokens` in the
  metadata, `GetModelInfo` reads the whole main section to find the context, but refuses sections
  over 2 GiB: E4B's (2,260,043,376 bytes) reads 0, E2B's (818 MB, read into memory) reads 32,003.
  <!-- c:MOD-E2B-049 -->
- **Do.** Don't gate features on these flags; with E2B, call `ModelInfo` before loading the engine
  or cache its result.
- **Applies to.** Both.

#### An `ExperimentalFlags` setting has no effect

- **Symptom.** A flag set in Swift changes nothing.
- **Cause.** Setters ignore the value and log an error unless `optIntoExperimentalAPIs()` was
  called; the flags are process-wide statics. <!-- c:LM-SWIFT-032 --> Engine-level flags are read
  once, by `Engine.initialize`. <!-- c:LM-SWIFT-004 -->
- **Do.** Opt in, then set engine-level flags before creating the engine (Agello's
  `LiteRTChatEngine.load` does both). <!-- c:LM-SWIFT-056 -->
- **Applies to.** Both.

#### Two loads take turns, or a load is slow

- **Symptom.** Two engines created at once load one after the other; the first load after an
  install or an update takes several seconds, later ones about one.
- **Cause.** LiteRT serializes compiled-model creation behind a process-wide mutex.
  <!-- c:RT-RUN-014 --> The first load converts weights and builds GPU programs and writes the
  engine caches; later loads restore them. Probe 14 on the iPhone 17 Pro, our build: 7.2 s, then
  1.5, 1.7 and 1.0 s for further sizes. <!-- c:OUR-023 -->
- **Do.** Load once, from one place (Agello's `serialized { }`); keep the cache directory between
  launches.
- **Applies to.** Both; the shared-weights speed-up is ours.

#### Engine caches: folders that grow, or caches from another build

- **Symptom.** The cache folder holds several ~2.47 GB copies; or after an update, restored
  programs misbehave; or the log says "Cache needs to be built again".
- **Cause.** Upstream's weight cache key includes the KV cache sizes, so each context size writes
  another copy of the weights; ours keys by content. <!-- c:RT-DLG-055 --> <!-- c:PRC-008 --> The
  program cache key has no GPU or build identity <!-- c:RT-DLG-048 -->; the weight cache rejects a
  file written by another ML Drift build of its fully connected weight code, and only that
  <!-- c:RT-DLG-052 --> <!-- c:RT-DLG-053 -->. A weight cache loaded from disk that lacks some
  tensors leaves them uncached and prepares them on every load (ours reports
  `weight_cache_rebuild`). <!-- c:RT-DLG-060 -->
- **Do.** Give each engine configuration a cache folder of its own and each runtime build its own
  set, as Agello's `EngineCache` does: `kv-all-<build>` for ours, `kv-<n>` with two sizes kept for
  Google's, `:nocache` for one-off engines. <!-- c:X-MEM-014 --> Never hand an engine the cache root.
  Sizes and keys: [memory.md](memory.md#engine-caches-on-disk).
- **Applies to.** Both; the content key and the anomaly are ours.

### Sending and generating

#### The reply stops, or the next message answers nothing

- **Symptom.** A reply ends early; the session stream's last chunk says "Max number of tokens
  reached."; the next message on the same conversation completes at once with an empty reply.
- **Cause.** Decode stops when the executor's step reaches `max_num_tokens` (a full KV cache) and
  ends the task `kMaxNumTokensReached`, not an error. <!-- c:LM-ENG-077 --> <!-- c:LM-ENG-078 -->
  A task the session chains to it afterwards is created in that state and never runs.
  <!-- c:LM-ENG-035 --> In a `Conversation` with the default `return_error_on_max_tokens_reached`
  false <!-- c:LM-CNV-001 -->, the next message's prefill callback gets that end state, the internal
  callback treats it like `kDone`, and the send completes with an empty reply and no error; nothing
  is evicted from the cache, which never drops tokens by itself. <!-- c:X-TRB-003 -->
  <!-- c:LM-EXE-066 -->
- **Where.** [`runtime/core/tasks.cc:96-110`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/core/tasks.cc#L96-L110),
  [`runtime/framework/resource_management/threaded_execution_manager.cc:304-324`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/framework/resource_management/threaded_execution_manager.cc#L304-L324),
  [`runtime/conversation/conversation.cc:800-822`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/conversation.cc#L800-L822),
  [`runtime/conversation/internal_callback_util.cc:318-343`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L318-L343).
- **Do.** Keep every send inside the room left in the cache, and after a full cache start a new
  conversation from saved history. Agello checks on each chat send: `fitContext` throws
  `messageTooLong` when the system preamble, the message's estimate, a reply reserve
  (min(`maxNumTokens` / 3, 512 + the thinking budget)) and a 64-token margin exceed the window
  (otherwise it trims and reseeds the history when the conversation is too full); `runSend` caps the
  reply at the room left minus a 64-token margin and throws `contextFull` below 96 tokens; when any
  tool is offered (on-device tools or `web_search`), `ToolBudget.plan` splits the room (a quarter per
  generation, an eighth per tool result counted as 3 bytes per token, at most two calls); and a
  streaming guard cancels the reply (`replyTruncated`) when emitted bytes plus tool result bytes
  exceed the room (3 bytes per token). <!-- c:X-TRB-002 --> Agello's
  skill records that overflowing the cache corrupted the native heap with the shipped binary; in the
  documented build's code decode stops at the limit and an oversized prefill is refused (next
  entry); the one path that addresses positions past the end is an MTP round
  ([MTP near the end of the cache](#mtp-near-the-end-of-the-cache)), and whether it explains that
  corruption is an open question.
- **Applies to.** The traced chain: the documented build. Agello's guards: both.

#### "Input token ids are too long" or "Prefill input length exceeds available state entries"

- **Symptom.** A send fails with `INVALID_ARGUMENT: Input token ids are too long...` or
  `FAILED_PRECONDITION: ... exceeds available state entries`.
- **Cause.** `Tasks::Prefill` rejects an input whose own length is at least `max_num_tokens`; it
  doesn't add the tokens already in the cache. <!-- c:LM-ENG-075 --> The executor then refuses a
  prefill that doesn't fit the remaining positions. <!-- c:LM-EXE-060 -->
- **Where.** [`runtime/core/tasks.cc:560-576`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/core/tasks.cc#L560-L576),
  [`runtime/executor/litert_compiled_model_executor_utils.cc:511-521`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/litert_compiled_model_executor_utils.cc#L511-L521).
- **Do.** Count the conversation's tokens plus the new input against the allocated context; load a
  larger context for long inputs. A Gemma 4 conversation prefills its whole history with the first
  message when it was created with prior messages. <!-- c:LM-CNV-022 -->
- **Applies to.** Both.

#### A 32K context holds 32,000 positions

- **Symptom.** `maxNumTokens` 32,768 behaves like 32,000.
- **Cause.** For the E files, any `maxNumTokens` of 32,003 or more allocates 32,000 positions, with
  only a log warning. <!-- c:LM-EXE-003 -->
- **Do.** Budget with 32,000 ([memory.md](memory.md#c9-918-mb-or-940-mb)).
- **Applies to.** Both.

#### MTP near the end of the cache

- **Symptom.** None observed with the documented build. Agello's skill records native heap corruption
  after a KV overflow with the shipped binary; this entry lists what in the code can address positions
  past the allocated context N (positions 0 to N - 1).
- **Cause.** Traced in the agello-next and v0.18.0 sources:
  - *Prefill can't.* The executor charges each work group its whole signature length against the
    positions left and refuses what doesn't fit, so no prefill, the channel filter's refill included,
    writes past N; with E4B's `prefill_128` and `prefill_1024`, every prefill fails when fewer than 128
    positions remain, however short it is. <!-- c:X-GAP-008 --> <!-- c:LM-EXE-010 -->
  - *Plain decode can't.* For the E files `max_num_tokens` equals N <!-- c:LM-EXE-003 -->, a plain step
    writes only position `current_step - 1`, and decode stops once `current_step` reaches
    `max_num_tokens`. <!-- c:X-GAP-009 -->
  - *An MTP round can.* Nothing on the MTP path compares the verify positions p..p+3 with N. A primed
    round can reach position N + 1, and a priming round right after a prefill that filled the cache can
    reach N + 3. <!-- c:X-GAP-010 --> On the GPU in our build the key and value writes there are
    dropped: `add_values_to_cache` skips positions at or beyond `cache_size`, which LiteRT rewrites to
    N. <!-- c:RT-DLG-024 --> <!-- c:RT-RUN-012 --> <!-- c:X-GAP-011 --> The verify run's attention mask
    is filled on the CPU without that bound: for a causal mask, row k is filled up to column p + k + 1,
    so the last row ends p + 4 - N bytes past the 4N-byte `BOOL[1, 1, 4, N]` tensor, at most 2 bytes in
    a primed round and 4 in a priming round. <!-- c:X-GAP-012 --> Where those bytes land depends on the
    buffer: when no GPU consumer registers requirements for a BOOL mask, LiteRT's compiled model falls
    back to host memory, which LiteRT allocates with the buffer size plus `XNN_EXTRA_BYTES` and a zeroed
    tail (16 bytes when LiteRT is built without its CPU path, XNNPACK's own value otherwise). Which
    buffer type the verify mask gets on Metal, and what a write past a Metal buffer's tensor size hits,
    were not traced. <!-- c:X-GAP-013 -->
- **Where.** [`runtime/executor/litert_compiled_model_executor_utils.cc:508-637`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/litert_compiled_model_executor_utils.cc#L508-L637),
  [`runtime/executor/llm_litert_compiled_model_executor.cc:2015-2028`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_litert_compiled_model_executor.cc#L2015-L2028),
  [`runtime/executor/llm_litert_compiled_model_executor.cc:99-114`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_litert_compiled_model_executor.cc#L99-L114),
  [`runtime/executor/llm_litert_compiled_model_executor.cc:1053-1056`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_litert_compiled_model_executor.cc#L1053-L1056),
  [`runtime/executor/llm_litert_compiled_model_executor.cc:1281-1306`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_litert_compiled_model_executor.cc#L1281-L1306),
  [`runtime/executor/llm_litert_compiled_model_executor.cc:1377-1432`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_litert_compiled_model_executor.cc#L1377-L1432),
  [`runtime/executor/llm_litert_mtp_drafter.cc:576-607`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_litert_mtp_drafter.cc#L576-L607),
  [`runtime/executor/llm_litert_mtp_drafter.cc:711-760`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_litert_mtp_drafter.cc#L711-L760),
  [`runtime/executor/litert_compiled_model_executor_utils.cc:792-916`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/litert_compiled_model_executor_utils.cc#L792-L916),
  [`runtime/executor/litert_compiled_model_executor_utils.cc:169-180`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/litert_compiled_model_executor_utils.cc#L169-L180),
  [`ml_drift_delegate/delegate/composite/add_values_to_cache_kernel.cc:119-142`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/ml_drift_delegate/delegate/composite/add_values_to_cache_kernel.cc#L119-L142),
  [`litert/runtime/tensor_buffer.cc:276-290`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/litert/runtime/tensor_buffer.cc#L276-L290);
  v0.18.0: [`runtime/executor/llm_litert_mtp_drafter.cc:648-680`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/runtime/executor/llm_litert_mtp_drafter.cc#L648-L680),
  [`runtime/executor/litert_compiled_model_executor_utils.cc:794-811`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/runtime/executor/litert_compiled_model_executor_utils.cc#L794-L811),
  [`runtime/executor/litert_compiled_model_executor_utils.cc:471-600`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/runtime/executor/litert_compiled_model_executor_utils.cc#L471-L600).
- **Do.** Stop each reply a few positions before N and leave at least 128 positions for any prefill.
  Agello (93122b6) starts a chat reply only when `maxNumTokens` minus the tokens in use, the input
  estimate and a 64-token margin leaves at least 96 tokens, and caps the reply at that room (at a
  quarter of it when tools are offered). <!-- c:X-TRB-002 --> When `maxNumTokens` is below 32,003, so
  that it equals E4B's allocated context (the defaults 8,192 and 4,096 are), and with exact counts, a
  send's prefill has at least its input plus 160 positions free, more than the at most input + 127 that
  E4B's `prefill_128` runs need, and a reply without tool calls stops decoding 64 positions before the
  end of the window plus at most one MTP round's tokens. A manual KV-cache size of 32,768 (the
  Stepper's top when the app memory limit reads at least 6.5 GB) allocates only 32,000 positions, and
  the margin no longer holds. <!-- c:X-GAP-024 -->
- **Applies to.** The traced code: both sources. The dropped KV writes: the documented build (the GPU
  delegate in Google's v0.18.0 binary isn't in our sources). Attention reads past N on the GPU (the
  verify `param_tensor` ends at p + 4) weren't traced.

#### "Visual token budget (N) cannot be larger than the engine's max vision tokens per image (280)"

- **Symptom.** A message with an image and a budget above 280 fails with `INVALID_ARGUMENT`.
- **Cause.** `Conversation::SendMessageAsync` rejects budgets above the engine's limit, 280 for the
  E files. <!-- c:MOD-E4B-050 -->
- **Where.** [`runtime/conversation/model_data_processor/model_data_processor_factory.cc:698-722`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/model_data_processor_factory.cc#L698-L722).
- **Do.** Clamp to the model's maximum: Agello lowers its 560-token "Detailed" preset to 280.
  <!-- c:MOD-E4B-051 -->
- **Applies to.** Both.

#### The prompt's media don't match

- **Symptom.** `INVALID_ARGUMENT` when a message carries images or audio.
- **Cause.** The number of images or audio clips must equal the placeholders the template renders.
  <!-- c:LM-CNV-036 --> Multimodal input is accepted only in prefill; a negative (media) token at
  decode fails. <!-- c:LM-EXE-090 -->
- **Applies to.** Both.

#### "... does not start with the previous rendered template string"

- **Symptom.** A later message fails with `INTERNAL`.
- **Cause.** Without single-turn template support (the E files' template), each message prefills the
  difference between the full render with it and the render without it, and fails when the new
  render doesn't start with the old one. <!-- c:LM-CNV-021 -->
- **Where.** [`runtime/conversation/conversation.cc:308-313`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/conversation.cc#L308-L313).
- **Do.** Use the file's template; a custom template must render history the same way whatever
  follows it.
- **Applies to.** Both.

#### A JSON schema keeps constraining later messages

- **Symptom.** After one message with a response format, later messages of the same conversation
  come out as JSON too.
- **Cause.** `Conversation` keeps the constraint it creates and passes it to every later decode.
  <!-- c:LM-CNV-071 -->
- **Where.** [`runtime/conversation/conversation.cc:389-407`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/conversation.cc#L389-L407).
- **Do.** One conversation per constrained request, as Agello's JSON helpers do. <!-- c:LM-CNV-079 -->
- **Applies to.** The documented build (the shipped source wasn't traced for this).

#### JSON-constrained replies break with MTP on

- **Symptom.** With speculative decoding on, JSON-schema helpers return text that breaks the schema.
- **Cause.** Agello's observation on device (its `helperJSON` comment says MTP-drafted tokens
  bypassed the JSON mask), with the shipped binary. <!-- c:LM-SWIFT-059 --> In the documented build,
  every token an MTP round emits is the argmax of the verifier's logits after any constraint mask,
  and the drafter builds mask runners for both its drafter and verifier, so the mechanism there isn't
  established. <!-- c:LM-EXE-024 --> <!-- c:LM-CMP-052 --> Tracing the constraint state across a
  round shows the mechanism, the same in agello-next and the v0.18.0 source:
  1. `LlgConstraint::ComputeNext` commits the token into the llguidance object of the state it is given
     and returns a copy that shares that object; only `Start()` clones a fresh one. All states derived
     from one `Start()` are one mutable matcher. <!-- c:X-GAP-019 -->
  2. The drafting loop masks each draft correctly, then commits it. By verification, the verified state
     and the three draft states all point to the matcher after d1, d2 and d3, so verifier rows 0 to 2
     are masked for the position after d3, not their own; only row 3 is right. <!-- c:X-GAP-020 -->
  3. After acceptance, the verified state becomes draft state n (the same matcher) and the bonus token is
     committed on top: the matcher holds d1, d2, d3 and the bonus whatever n is. That is right only when
     all 3 drafts are accepted; after a rejection, later masks start from a state with rejected drafts in
     it, and the bonus commit can fail, failing the decode with `InternalError` ("Failed to commit
     token"). <!-- c:X-GAP-021 -->
  4. Priming is right: the priming step is masked by `ConstrainedDecoder`'s own state, and `Draft` then
     starts from a fresh `Start()` plus the priming token. `ConstrainedDecoder::UpdateState` is not
     called in MTP rounds at all; it advances only on plain steps, and with MTP on the priming step is
     the only one in a decode. <!-- c:X-GAP-022 -->

  So the drafts are masked from the right states, but a rejection's bonus token is picked under the
  mask of the position after d3, and a rejected draft stays committed in the matcher, so later masks
  come from a wrong state: a constrained reply with MTP on can leave the grammar. This is consistent
  with the symptoms Agello recorded on device (garbled output or "compute_mask() called after stop"),
  though Agello's comment blames the drafted tokens. MTP must stay off for constrained conversations at
  the pin too, and it is a correctness bug to report upstream. <!-- c:X-GAP-023 -->
  ([`runtime/components/constrained_decoding/llg_constraint.cc:66-91`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/constrained_decoding/llg_constraint.cc#L66-L91),
  [`runtime/components/constrained_decoding/llg_constraint.h:41-72`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/constrained_decoding/llg_constraint.h#L41-L72),
  [`runtime/executor/llm_litert_mtp_drafter.cc:501-572`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_litert_mtp_drafter.cc#L501-L572),
  [`runtime/executor/llm_litert_mtp_drafter.cc:663-709`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_litert_mtp_drafter.cc#L663-L709),
  [`runtime/executor/llm_litert_mtp_drafter.cc:761-783`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_litert_mtp_drafter.cc#L761-L783),
  [`runtime/executor/llm_litert_compiled_model_executor.cc:1526-1559`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_litert_compiled_model_executor.cc#L1526-L1559),
  [`runtime/components/constrained_decoding/constrained_decoder.cc:108-122`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/constrained_decoding/constrained_decoder.cc#L108-L122);
  v0.18.0: [`runtime/executor/llm_litert_mtp_drafter.cc:578-645`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/runtime/executor/llm_litert_mtp_drafter.cc#L578-L645),
  [`runtime/executor/llm_litert_mtp_drafter.cc:826-849`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/runtime/executor/llm_litert_mtp_drafter.cc#L826-L849),
  [`runtime/components/constrained_decoding/llg_constraint.cc:79-91`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/runtime/components/constrained_decoding/llg_constraint.cc#L79-L91)).
- **Do.** Turn MTP off for constrained conversations (`enableSpeculativeDecoding: false`), as Agello
  does; plain-text helpers keep it. This holds for agello-next too.
- **Applies to.** Observed with the shipped binary; the traced mechanism is in both sources (not run
  on device with the documented build).

#### Reasoning fills the context

- **Symptom.** Think turns use more of the KV cache than their answers.
- **Cause.** The channel filter removes reasoning from the cache only when
  `filter_channel_content_from_kv_cache` is on; the Swift wrapper passes the flag only when it is set,
  and the C++ default is on. <!-- c:LM-CNV-057 --> In the documented build the filter rewinds right
  after a reply and refills it without its reasoning <!-- c:LM-CNV-052 -->; v0.18.0 does it lazily, at
  the next user message. <!-- c:LM-CNV-054 -->
- **Do.** Set `ExperimentalFlags.filterChannelContentFromKvCache = true` explicitly, as Agello does.
  <!-- c:LM-CNV-058 -->
- **Applies to.** Both, with the timing difference above.

#### Sampling settings are ignored

- **Symptom.** Temperature, top-k or top-p of a conversation seem to have no effect.
- **Cause.** The executor creates its sampler once, from the parameters of the first session that
  samples, and keeps it for every session of the engine. <!-- c:LM-ENG-062 --> With MTP on, every
  token an MTP round emits is greedy; only the priming step uses the session's sampler.
  <!-- c:LM-EXE-024 --> Agello's one-token validation (default parameters) is the first sampling on
  each engine.
- **Do.** Make the first conversation that samples on a new engine carry the wanted `SamplerConfig`.
  In Agello that is the validation probe <!-- c:X-CFG-007 -->, so it should pass the chat's sampler.
  Expect greedy output from MTP rounds. <!-- c:LM-EXE-024 --> The effect on Agello's chat is derived
  from code <!-- c:LM-ENG-062 --> and not yet observed in a run.
- **Applies to.** Both (v0.18.0's sampler has the same early return; its MTP priming differs
  <!-- c:LM-EXE-026 -->).

#### `getTokenCount()` reads 0 before the first message

- **Symptom.** A new conversation reports 0 tokens although it has a system prompt or prior
  messages.
- **Cause.** `GetTokenCount` returns the session's current step; with `prefill_preface_on_init`
  false (the default) nothing is prefilled at creation, and the preface and prior messages go in with
  the first message. <!-- c:X-TRB-004 --> <!-- c:LM-CAPI-034 -->
- **Where.** [`runtime/conversation/conversation.cc:437-438`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/conversation.cc#L437-L438),
  [`runtime/conversation/conversation.cc:889-891`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/conversation.cc#L889-L891).
- **Do.** Estimate the seeded history yourself until the first send (Agello's `pendingSeedTokens`).
- **Applies to.** The documented build.

#### Text LoRA, CPU samplers and long embedding inputs

- **"Lora is not supported."** A session with a text LoRA file fails at creation; no code path loads
  one. <!-- c:LM-CMP-071 -->
- **`TOP_K` or `GREEDY` on the CPU** returns `UNIMPLEMENTED`; greedy is `TOP_P` with k 1.
  <!-- c:LM-CMP-024 -->
- **Embedding input too long** fails from Swift: the overflow strategy is always *error* there.
  <!-- c:LM-SWIFT-042 --> Chunk inputs below the longest signature.
- **Applies to.** Both.

#### Tool calls

- **Tool loop limit.** `sendMessage` stops after 25 rounds that still call tools.
  <!-- c:LM-SWIFT-012 -->
- **Tools run with `automaticToolCalling` false** on the stream path, which never checks the flag.
  <!-- c:LM-SWIFT-017 --> Don't register tools on conversations that mustn't call them.
- **A failing tool ends the reply** with upstream's wrapper; our patch answers every call with a
  result or an error message instead. <!-- c:LM-SWIFT-024 --> <!-- c:LM-SWIFT-023 -->
- **`fatalError` inside a tool**: a required `@ToolParam` read without a value.
  <!-- c:LM-SWIFT-026 --> Give parameters defaults or make them optional.
- **Applies to.** Both wrappers carry our patch.

### Cancellation, helpers and threads

#### The next message fails with "CANCELLED" after a cancel

- **Symptom.** After `conversation.cancel()`, the next message ends at once with
  `conversation(.invalidResponse("CANCELLED: Task cancelled"))`.
- **Cause.** `CancelProcess` cancels every task of the session. <!-- c:LM-ENG-044 --> A task chained
  to a cancelled one after its final callback is created `kDependentTaskCancelled` and never runs.
  <!-- c:LM-ENG-034 --> The conversation reports that state exactly like a cancel, as
  `CancelledError("Task cancelled")`, which the C stream turns into the error chunk.
  <!-- c:X-TRB-005 --> <!-- c:LM-SWIFT-022 --> The cancelled turn's tokens also stay in the KV cache
  while the history drops them <!-- c:LM-CNV-081 -->, and both headers say a cancelled session or
  conversation is not reusable. <!-- c:LM-CNV-082 -->
- **Where.** [`runtime/framework/resource_management/threaded_execution_manager.cc:304-324`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/framework/resource_management/threaded_execution_manager.cc#L304-L324),
  [`runtime/conversation/internal_callback_util.cc:302-308`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L302-L308),
  [`runtime/core/session_advanced.h:192-211`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/session_advanced.h#L192-L211).
- **Do.** Treat a cancelled conversation as spent: create a new one from saved history, as Agello
  does. <!-- c:LM-SWIFT-060 --> While a helper runs, cancel the helper's conversation, never the kept
  chat.
- **Applies to.** The traced chain: the documented build; Agello observed the same with the shipped
  binary.

#### A cancel doesn't stop the work at once

- **Symptom.** After `cancel()`, the engine stays busy for a while; cancelling the Swift `Task` that
  consumes a stream doesn't stop generation at all.
- **Cause.** Cancellation is checked between decode steps and around prefill, not inside a running
  prefill. <!-- c:LM-ENG-045 --> The Swift stream registers no `onTermination`; only
  `Conversation.cancel()` reaches the runtime. <!-- c:LM-SWIFT-013 -->
- **Do.** Call `cancel()` when the consumer stops (Agello checks `Task.isCancelled` on every chunk).
- **Applies to.** Both.

#### A new conversation takes about 600 s to answer

- **Symptom.** After a stream was abandoned, the next conversation's first token arrives after
  about 10 minutes.
- **Cause.** With upstream's wrapper, a stream held its conversation strongly; releasing the last
  reference inside the final callback ran the conversation's deletion on the callback thread, where
  the session destructor waits up to 10 minutes for the task that is still finishing.
  <!-- c:LM-SWIFT-020 --> <!-- c:LM-ENG-027 --> Probe 9 measured 600.2 s before our fix and 0.111 s
  after. <!-- c:OUR-006 -->
- **Where.** [`swift/Conversation.swift:830-840`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L830-L840).
- **Do.** Use our wrapper (the stream holds its conversation weakly) <!-- c:OUR-005 -->; never
  release the last reference to a conversation inside its own callback; cancel, wait for the last
  chunk, then delete.
- **Applies to.** Both of our wrappers carry the fix; upstream's doesn't.

#### A helper wipes the chat, or a kept chat costs memory

- **Symptom.** With Google's runtime, after a title or memory helper the chat must be reseeded; with
  ours, memory rises by about one KV cache while a helper runs.
- **Cause.** One engine has one live KV cache; switching sessions snapshots the outgoing one in our
  build and restores it when the chat next acquires the executor. <!-- c:LM-ENG-064 --> Agello keeps
  the chat only on our runtime and only with one cache of headroom above its 400 MB stop line.
  <!-- c:APL-015 --> Agello `6d389e6` restores it right after the helper with `getTokenCount()`.
  <!-- c:LM-ENG-065 -->
- **Do.** Ask whether the helper kept the conversation (`lastHelperKeptConversation`) and reseed
  only when it didn't. Sizes: [memory.md](memory.md#host-snapshots-of-kv-for-kept-conversations).
- **Applies to.** Kept chats: our build only.

#### Helpers wait, or decode slows with a heavy callback

- **Cause.** All sessions of an engine share one execution thread; a task waits behind tasks
  already queued, and a decode runs to its end before the next task. <!-- c:LM-ENG-061 --> Chunks
  streamed during decoding are delivered on that thread. <!-- c:LM-ENG-039 -->
- **Do.** Keep stream callbacks short; schedule helpers between replies.
- **Applies to.** Both.

#### A CPU core spins while the GPU works

- **Cause.** Agello sets `ExperimentalFlags.enableBenchmark`, which puts the engine in benchmark
  mode, where GPU waits are active (busy-wait) in both builds; our passive-wait change applies only
  outside benchmark mode. <!-- c:OUR-034 --> <!-- c:LM-EXE-038 -->
- **Do.** A decision for Agello: collect statistics without benchmark mode to get passive waits.
- **Applies to.** Both.

### Wrong output

#### Digits repeat or shift in long replies (#3913)

- **Symptom.** With Google's E4B or E2B file on the GPU at default FP16 activations, verbatim
  numbers, IDs and citations stutter once the reply passes position 2,048 ("2026" becomes
  "202026"); on M1-class GPUs errors start below 2,048.
- **Cause.** The stock files compute rotary angles at run time from a float copy of the INT32
  position <!-- c:MOD-E4B-058 -->; at FP16 on the GPU those tensors hold integers exactly only up to
  2,048 <!-- c:MD-KRN-075 -->, so near position 30,000 up to 16 positions share one phase
  <!-- c:MOD-E4B-059 -->. Measured: 15/15 exact below 2,048, 0/12 above, 9/9 at FLOAT32
  <!-- c:MOD-E4B-061 -->; shifting the position in INT32 before the cast fixed it, after the cast
  didn't <!-- c:MOD-E4B-062 -->. An M1 Max loses precision below 2,048 too, once the angle passes
  about 1,650 rad. <!-- c:MOD-E4B-063 --> ML Drift's own RoPE kernel is FP32, but the E4B graphs
  don't use it. <!-- c:MD-KRN-076 -->
- **Do.** Use the patched file: `patch_rope.py --split --block 512` splits each position into
  512-token blocks so every angle stays exact <!-- c:MOD-E4B-065 -->; it was exact in 88/88 copies on
  an M6 up to 29,960 tokens and 14/14 (plus 6/6 with MTP) on the iPhone 17 Pro.
  <!-- c:MOD-E4B-070 --> <!-- c:MOD-E4B-071 --> Agello downloads that file. <!-- c:MOD-E4B-074 -->
  FLOAT32 activations also avoid it, at about half the decode speed. <!-- c:MOD-E4B-082 --> For E2B,
  512-token blocks were exact on an M1 Max and an M4 Max. <!-- c:MOD-E2B-062 --> <!-- c:MOD-E2B-063 -->
  Details: [models/gemma4-e4b.md](models/gemma4-e4b.md#rotary-positions-and-litert-lm-3913).
- **Applies to.** Both runtimes with Google's files; neither with ours.

#### Positions at or above 32,768 with the patched file

- **Cause.** The rotary tables cover 0–32,767; LiteRT-LM never reaches 32,768 with these files
  because the export window caps the allocated context at 32,002. <!-- c:MOD-E4B-073 -->
- **Do.** Re-run `patch_rope.py` with more blocks for any re-export with a larger window.

#### Replies differ between devices or settings

- **Cause.** Apple10 GPUs run prefill on `ConvAppleMPP` with INT8 inputs, Apple9 and older on
  `ConvWaveMatrix` in FP16 <!-- c:MD-KRN-002 --> <!-- c:MD-KRN-003 -->; decode picks the same kernels
  on all families. <!-- c:MD-KRN-001 --> FP16 logit masks use −65,504, not −∞. <!-- c:LM-CMP-043 -->
- **Do.** Compare replies for correctness, not bits ([apple-platform.md](apple-platform.md#what-ml-drift-does-on-each-family)).

### Memory and the device

#### The app is terminated for memory

- **Symptom.** iOS ends the app; the jetsam report says `per-process-limit`. <!-- c:APL-021 -->
- **Cause and budget.** See [memory.md](memory.md#a-budget-for-e4b-by-context-size): E4B at 32K
  reached a 4.81 GB high-water mark during a long prompt on the iPhone 17 Pro <!-- c:X-MEM-005 -->;
  the first image adds about 1.3 GB <!-- c:X-MEM-009 -->; a kept chat adds one KV cache.
- **Do.** Stop replies when available memory runs low (Agello: below 400,000,000 bytes, not at the
  first memory warning) <!-- c:APL-015 -->; keep the default context at 8K.
- **Applies to.** Both.

#### The app limit reads 6.4 GB, or the Settings stepper stops at 16,384

- **Cause.** Early in some runs, before the first engine load, footprint + available returned
  6 GiB instead of 8 GiB. <!-- c:APL-001 --> <!-- c:APL-004 --> Agello's 32K ceiling needs 6.5·10⁹
  bytes, so an early reading caps the stepper at 16,384. <!-- c:APL-014 --> The memory envelope test
  read the limit once, so three runs used too little ballast. <!-- c:APL-025 -->
- **Do.** Re-read the limit; don't cache it ([apple-platform.md](apple-platform.md#c1-86-gb-or-64-gb)).
- **Applies to.** Agello on the iPhone 17 Pro.

#### A CPU engine takes several GB, or iOS kills it

- **Cause.** Our source build's XNNPACK, before our patch, sized a per-channel scale array for the INT8
  K cache's matrix product as T × 4 × (2 × T) bytes, 512 MiB per 8K cache.
  <!-- c:RT-CPU-012 --> <!-- c:RT-CPU-013 --> Upstream XNNPACK `c72e2a4ae0` fixed it
  <!-- c:RT-CPU-015 -->; our fork applies that diff to TensorFlow's pin <!-- c:RT-CPU-016 -->, after
  which a CPU reply on the iPhone ran at 3.67 GB. <!-- c:RT-CPU-017 -->
- **Do.** Keep the patch until TensorFlow's XNNPACK pin includes the fix. <!-- c:OUR-012 -->
- **Applies to.** Our source build (fixed by our patch). Whether Google's v0.18.0 binary has the fix
  isn't recorded; upstream's open `WORKSPACE` got it only with `c05e67a2`. <!-- c:OUR-012 -->

#### Device tests show about 1.2 GB more than expected

- **Cause.** Before Agello `8bae20b`, the app hosting the tests loaded its own E4B engine next to the
  tests'. <!-- c:X-MEM-011 -->
- **Do.** Compare only reports recorded the same way; reports from the afternoon of 2026-10-15 on
  (`*host-idle*`) are without it.

### Agello's rules, traced

The `agello-inference-rules` skill (Agello `93122b647a`) lists rules from failures reproduced on
device. Each one, with the runtime behavior behind it:

| Rule | What breaks without it | Traced to |
| --- | --- | --- |
| Check the context before every send; cap the output | a full cache ends the session's work; the skill records heap corruption with the shipped binary | [KV full](#the-reply-stops-or-the-next-message-answers-nothing), [too long](#input-token-ids-are-too-long-or-prefill-input-length-exceeds-available-state-entries) |
| Reseed after every cancel | the next message is born cancelled | [cancel](#the-next-message-fails-with-cancelled-after-a-cancel) |
| While a helper runs, cancel the helper, never the chat | the kept chat would be poisoned | [cancel](#the-next-message-fails-with-cancelled-after-a-cancel) |
| Validate every new engine, retry once after 750 ms | a section that failed to map leaves an engine that fails every message | [validation](#the-engine-loads-but-every-message-fails-no-per-layer-embeddings-found-in-inputs) |
| Set engine-level `ExperimentalFlags` before creating the engine | flags are read once at `initialize()` | [flags](#an-experimentalflags-setting-has-no-effect) |
| Engine caches through `EngineCache`, never the root | per-size weight copies (Google's), stale programs across builds | [caches](#engine-caches-folders-that-grow-or-caches-from-another-build) |
| Only the patched model file | digits wrong past position 2,048 at FP16 | [#3913](#digits-repeat-or-shift-in-long-replies-3913) |
| MTP off for JSON helpers | JSON broken on device | [MTP and JSON](#json-constrained-replies-break-with-mtp-on) |
| Don't trust the file's `thinking` / `functionCalling` flags | features hidden that work | [capabilities](#the-model-reports-no-thinking-no-tools-and-no-context) |
| Set `filterChannelContentFromKvCache = true` | reasoning stays in the cache | [reasoning](#reasoning-fills-the-context) |
| `pendingSeedTokens` stands in for `getTokenCount()` | 0 before the first send | [token count](#gettokencount-reads-0-before-the-first-message) |
| Stop replies below 400 MB available | jetsam | [memory](#the-app-is-terminated-for-memory) |
| One engine operation at a time (`serialized { }`) | two engines held at once; loads take turns anyway | [loads](#two-loads-take-turns-or-a-load-is-slow) |

## On Apple

| Item | Documented build | Shipped (Google's v0.18.0 binary) |
| --- | --- | --- |
| Error reporting | C API 1.0.0: status codes, out-parameters | C API 0.2: pointers and `int`s, `litert_lm_get_last_error_code()` |
| Sampler | CPU fallback (log warning) | Metal TopK sampler linked (not observed running) |
| `-gpu` files | rejected | artisan executor present, untried |
| KV snapshot restore | copied into the bound buffers | moved in as the live state; Agello reseeds |
| Channel filter timing | right after the reply | at the next message |
| XNNPACK | our patch | not pinned (see [litert/cpu-xnnpack.md](litert/cpu-xnnpack.md)) |

## Where in the code

| Concept | Symbol | Citation |
| --- | --- | --- |
| Status codes | `LiteRtLmStatusCode` | [`c/error_reporter.h:140-158`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/c/error_reporter.h#L140-L158) |
| Swift errors | `LiteRTLMError` | [`swift/LiteRTLMError.swift:19-27`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/LiteRTLMError.swift#L19-L27) |
| Session stream end states | `CreateCallback` | [`c/engine.cc:70-92`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/c/engine.cc#L70-L92) |
| Conversation stream callback | `CreateConversationCallback` | [`c/conversation.cc:44-71`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/c/conversation.cc#L44-L71) |
| Inherited end states | `ThreadedExecutionManager::CreateTask` | [`runtime/framework/resource_management/threaded_execution_manager.cc:304-324`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/framework/resource_management/threaded_execution_manager.cc#L304-L324) |
| Conversation end states | `CreateInternalCallback` | [`runtime/conversation/internal_callback_util.cc:302-343`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L302-L343) |
| Stop conditions | `ShouldStop` | [`runtime/core/tasks.cc:96-110`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/core/tasks.cc#L96-L110) |
| Prompt too long | `Tasks::Prefill` | [`runtime/core/tasks.cc:560-576`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/core/tasks.cc#L560-L576) |
| No engine for a backend | `EngineFactory::CreateDefault` | [`runtime/engine/engine_factory.h:88-110`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_factory.h#L88-L110) |
| Backend constraints | `ValidateBackendConstraint` | [`runtime/engine/engine_settings.cc:81-112`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L81-L112) |
| Per-layer lookup skipped | `InitializeEmbeddingLookups` | [`runtime/executor/litert_compiled_model_executor_utils.cc:1294-1322`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/litert_compiled_model_executor_utils.cc#L1294-L1322) |
| Text LoRA | `ResourceManager::CreateContextHandler` | [`runtime/framework/resource_management/resource_manager.cc:610-622`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/framework/resource_management/resource_manager.cc#L610-L622) |
| First sampler wins | `InitializeSampler` | [`runtime/executor/llm_litert_compiled_model_executor.cc:1707-1712`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_litert_compiled_model_executor.cc#L1707-L1712) |
| Ops without CPU fallback | `LiteRtCompiledModelT::Create` | [`litert/runtime/compiled_model.cc:1185-1198`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/litert/runtime/compiled_model.cc#L1185-L1198) |
| Weight cache rejection | `SerializationWeightCache::Load` | [`ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.cc:218-229`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.cc#L218-L229) |
| Agello's guards | `fitContext`, `runSend`, `ContextBudget`, `ToolBudget` | `Agello/Inference/ChatEngine.swift:376-392`, `:618-640`, `:695-712`; `Agello/Inference/InferencePlanner.swift:48-110`, `:177-181`; `Agello/Inference/ToolHost.swift:52-62` (Agello `93122b647a`) |
| Agello's validation | `validate`, the load loop | `Agello/Inference/ChatEngine.swift:180-216`, `:295-307` |

## Open questions

- **Heap corruption on KV overflow.** Agello's skill records native heap corruption when the KV cache
  overflowed with the shipped binary. Traced since ([MTP near the end of the cache](#mtp-near-the-end-of-the-cache)):
  prefills and the filter's refill are refused rather than written past the end, and only an MTP round
  addresses positions past N; its KV writes are dropped on our GPU path, and its CPU mask fill runs up
  to 4 bytes past the verify mask. Open: which buffer type that mask gets on Metal, what the shipped
  binary's GPU delegate does with positions past N, and whether either explains the corruption; not
  reproduced.
- **MTP and JSON on our build.** The mechanism is traced (shared llguidance state,
  [JSON-constrained replies break with MTP on](#json-constrained-replies-break-with-mtp-on)); a device
  run with MTP on and a JSON schema hasn't been done with the documented build.
- **`-gpu` files on the shipped binary.** Whether Google's v0.18.0 binary runs an artisan file on
  iOS hasn't been tried.
- **Cross-family program caches.** Whether a program cache restored on another GPU family fails
  validation or runs wrong kernels isn't verified ([litert/ml-drift-delegate.md](litert/ml-drift-delegate.md#open-questions)).
- **Shipped conversation paths.** The KV-full and cancel chains above are traced in the documented
  build; the v0.18.0 conversation code differs (lazy channel filter) and wasn't traced end to end.
