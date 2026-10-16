# Returns: lm-engine (area A1, Run A)

Page written: `litert-lm/engine-and-sessions.md`. Claims: `_work/claims/lm-engine.jsonl`
(LM-ENG-001 to LM-ENG-100; 52 tier A, 48 tier B). All citations at LiteRT-LM `7b872544` unless they
say v0.18.0 (`b2f686e2` / `85b1a776`).

## Settings rows (for configuration.md)

| name | layer | default | where read | effect on Apple | version added | Agello sets it? | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `parallel_file_section_loading` | `EngineSettings`; C `litert_lm_engine_settings_set_parallel_file_section_loading` | true | [`runtime/core/engine_advanced_impl.cc:536-549`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/engine_advanced_impl.cc#L536-L549) | same as elsewhere: the tokenizer is built on a second thread during load (Gemma 4 files) | not checked | no | its only reader at `7b872544`; the name promises more than it does (LM-ENG-005, -006) |
| `single_threaded_execution` | `EngineSettings`; C `litert_lm_engine_settings_set_single_threaded_execution` | false | [`runtime/core/engine_advanced_impl.cc:662-679`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/engine_advanced_impl.cc#L662-L679) | `SerialExecutionManager`: no threads, tasks run inside waits, not thread-safe | not checked | no | meant for Wasm without pthreads (LM-ENG-016, -043) |
| `max_num_tokens` | `LlmExecutorSettings` via `EngineSettings`; C `set_max_num_tokens`; Swift `EngineConfig.maxNumTokens` | file `LlmMetadata.max_num_tokens`, else 4096 (C API, no prompt hint) | [`runtime/engine/engine_settings.cc:346-361`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L346-L361) | not clamped to 4096 on Apple (`#if !defined(__APPLE__)`) | not checked | yes | KV-full stop condition (LM-ENG-008, -077) |
| `max_vision_tokens_per_image` | `EngineSettings`; C setter | `max_num_patches / pooling²` from metadata: 280 for E4B | [`runtime/core/engine_advanced_impl.cc:457-494`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/engine_advanced_impl.cc#L457-L494) | | not checked | no (not passed in `makeEngine` at Agello `93122b6`) | must be positive (LM-ENG-014) |
| `BenchmarkParams.num_prefill_tokens`, `num_decode_tokens` | `EngineSettings`; C `enable_benchmark`, `set_num_prefill_tokens`, `set_num_decode_tokens` | off | [`runtime/engine/engine_settings.cc:512-515`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L512-L515), [`runtime/core/tasks.cc:95-117`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/core/tasks.cc#L95-L117) | benchmark mode keeps LiteRT's active Metal wait (ours `645aaf65`) | not checked | not in `makeEngine` | decode ignores stop tokens when `num_decode_tokens` > 0; prefill always waits (LM-ENG-090, -091) |
| sampler parameters (type, k, p, temperature, seed) | `SessionConfig`; C `session_config_set_sampler_params`; Swift `ConversationConfig.samplerConfig` | file `sampler_params`, else TOP_P k=1 p=0.95 temperature 1 seed 0 | [`runtime/engine/engine_settings.cc:417-436`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L417-L436), [`runtime/engine/engine_settings.cc:721-729`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L721-L729) | the executor's internal sampler takes the parameters of the first session that samples, for the engine's life | v0.18.0 or earlier (same code) | yes (chat: user settings; helpers: greedy) | LM-ENG-010, -062 |
| stop token IDs | `SessionConfig` | from metadata (E4B: 1, 50, 106) | [`runtime/engine/engine_settings.cc:759-769`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L759-L769) | | | no | error if none (LM-ENG-020) |
| `num_output_candidates` | `SessionConfig` | 1 | [`runtime/engine/engine_settings.cc:698-707`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L698-L707) | | | no | |
| sampler backend | `SessionConfig` | GPU for a GPU engine, else CPU | [`runtime/engine/engine_settings.cc:809-818`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L809-L818) | the GPU sampler falls back to the CPU when unavailable (components.md) | | no | |
| `apply_prompt_template_in_session` | `SessionConfig`; C `session_config_set_apply_prompt_template` | true | [`runtime/core/session_advanced.cc:96-150`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/core/session_advanced.cc#L96-L150) | | | no (`Conversation` sets false) | LM-ENG-024 |
| `use_external_sampler` | `SessionConfig` (C++ only) | false | [`runtime/framework/resource_management/threaded_execution_manager.cc:134-193`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/framework/resource_management/threaded_execution_manager.cc#L134-L193) | | | no | CPU only |
| `enable_audio_session_advanced` | `SessionConfig` (C++ only) | false | [`runtime/core/engine_advanced_impl.cc:287-310`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/engine_advanced_impl.cc#L287-L310) | | | no | |
| `max_output_tokens` | `SessionConfig`; `DecodeConfig`; Swift `sendMessage(maxOutputTokens:)` | `INT_MAX` | [`runtime/core/session_advanced.cc:301-314`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/core/session_advanced.cc#L301-L314) | | v0.18.0 or earlier | yes (helpers, validation probe) | thinking tokens count (LM-ENG-021) |
| `enable_speculative_decoding` (session) | `SessionConfig` (optional); C `session_config_set_enable_speculative_decoding`; Swift `ConversationConfig` | unset: inherit the engine's | [`runtime/engine/engine_settings.h:331-395`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.h#L331-L395) | | | yes | passed to every decode step |
| scoped LoRA file | `SessionConfig` | none | [`runtime/framework/resource_management/resource_manager.cc:605-620`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/framework/resource_management/resource_manager.cc#L605-L620) | | | no | an unregistered file fails with "Lora is not supported." |
| `gpu_enable_metal_residency_set` | `AdvancedSettings`; C setter; `Engine::UpdateGpuEnableMetalResidencySet` at runtime | not checked | [`runtime/executor/llm_litert_compiled_model_executor.cc:1873-1883`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_litert_compiled_model_executor.cc#L1873-L1883) | applied to run options only under `#if defined(__APPLE__)` | | not in `makeEngine` | LM-ENG-093 |
| cache directory | `LlmExecutorSettings`; Swift `EngineConfig.cacheDir` | empty | [`runtime/engine/engine_settings.cc:147-156`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L147-L156) | | | yes | `:nocache`, `:memory` accepted |
| `--define litert_lm_std_thread=true` | Bazel define | unset | [`runtime/framework/BUILD:39-48`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/framework/BUILD#L39-L48) | `std::thread` workers instead of pthreads | | no | Windows uses `std::thread` always |
| `Engine::kDefaultTimeout` | C++ constant (not a setting) | 10 minutes | [`runtime/engine/engine.h:452-453`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine.h#L452-L453) | | v0.18.0 or earlier | n/a | session waits and destructors (LM-ENG-018, -041) |

## Terms (for glossary.md)

| term | meaning | prefer / avoid |
| --- | --- | --- |
| execution thread | the one thread of `ThreadedExecutionManager`'s `execution_thread_pool` that runs every task of an engine (name `execution_threa` on Apple) | prefer "execution thread"; avoid "decode thread" (it runs prefill too) |
| callback thread | the one thread of `callback_thread_pool` that runs final task callbacks (`callback_thread`) | |
| task | one unit of session work (prefill, decode, clone, score) with a state, dependencies and a callback | avoid "job", "request" |
| task chain | a session's tasks linked through `last_task_ids_`; each new task depends on the previous one | |
| poisoned session | a session whose chain ends in a cancelled (or KV-full) task, so new tasks are born ended | say "cancelled session" in user docs |
| context handler | a session's view of the shared executor (`ContextHandler`): config, state and a shared processed context | |
| context switch | loading another session's context into the executor: snapshot out, restore in | avoid "session swap" |
| KV snapshot | the host-memory copy of the KV state taken at a context switch or clone divergence | prefer "snapshot"; "raw snapshot" for ours (byte copy in GPU layout), "lock-path snapshot" for Google's |
| orphaned handler | the current handler of a released session, held only by the resource manager | |
| kept conversation | Agello's chat conversation kept alive (with a snapshot) while a helper conversation runs | |
| internal sampler | the executor's own sampler (`LlmLiteRtCompiledModelExecutorBase::sampler_`), created once per engine | avoid "session sampler" for it |
| allocated context vs output limit | `max_num_tokens` (KV capacity) vs `max_output_tokens` (decode steps per request) | never use "max tokens" alone |

## Contradictions

- **C11** (kept conversation's KV copy: until the next message vs freed when the helper ends).
  Resolution: in agello-next the snapshot lives until the kept session's handler next acquires the
  executor (any of its tasks, or a step query); T3 described Agello before `6d389e6`, where that was
  the next message, and VALIDATION describes Agello since `6d389e6`, whose `endHelper` calls
  `getTokenCount()` and so restores and frees it right after the helper. Claims LM-ENG-064, -065,
  -066 (and -056, -059, -060 for what each switch copies).
- **NEW (refinement of T3's "Google's v0.18.0 wipes the chat's conversation").** v0.18.0 snapshots
  the chat correctly but restores by swapping the host snapshot in as the live state; per the
  comment that replaced it upstream (`76594366`, 2026-10-12), the GPU delegate keeps using its bound
  device buffers, which hold the helper's cache. "Wiped" is the visible effect; the mechanism is a
  failed restore. Claim LM-ENG-055.
- **NEW (AUDIT §2.2, v0.18.0 line numbers).** "The tokenizer is built concurrently with the executor"
  holds only for files with an `llm_model_type` and with `parallel_file_section_loading` true (the
  default). Claim LM-ENG-005.
- **NEW (AUDIT §7, "`tasks.cc:640` builds 262,144 strings").** True of v0.18.0 and of upstream at
  the base; fixed by ours `5e946049`. Claim LM-ENG-082.

## Notes for other pages

- **stack.md.**
  - The decode loop (`Tasks::Decode`) runs on the execution thread (`execution_threa`), one decode
    task at a time for the whole engine (LM-ENG-030, -061).
  - Streamed chunks reach the C callback, and so the Swift `StreamContext` callback, on the execution
    thread between decode steps; the final chunk (done, max tokens, cancelled, error) on the callback
    thread (`callback_thread`); `kCreated`/`kQueued` on the caller's thread (LM-ENG-039, -087). The C
    API calls the user's function directly ([`c/conversation.cc:45-55`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/c/conversation.cc#L45-L55)).
  - `Conversation` schedules its decode from the prefill's final callback, on the callback thread,
    while the execution thread waits up to 10 s in `FinishTask` (LM-ENG-038, -040).
  - Tokenization (and, outside `Conversation`, templating) happens on the caller's thread before the
    prefill task is queued; image and audio encoders run inside the prefill task on the execution
    thread (LM-ENG-024, -100).
  - Per token at the session layer: the executor returns the step's token IDs as a host
    `std::vector<std::vector<int>>`; the stop filter and the streaming detokenizer produce a
    `std::string` per candidate; one `Responses` (text and token IDs, moved) is passed to the callback
    per step that released text, so with MTP one callback can carry several tokens (LM-ENG-085,
    -086). What the executor copies per step (logits readback, embeddings) is in executors.md; what
    `Conversation` and the C API copy per chunk (JSON) is in conversation.md and c-api.md.
- **executors.md.** The internal sampler is created once per executor from the first sampling
  context and never updated (LM-ENG-062); MTP's drafter and verifier use greedy samplers (LM-ENG-063);
  `RestoreState` copies host snapshots into device buffers in agello-next but swaps them in v0.18.0
  (LM-ENG-054, -055); `SetCurrentStep` moves only the step (LM-ENG-069); our `e43f9a8d`
  (LM-ENG-095). Open there: what happens when one turn's prefill exceeds the remaining KV.
- **conversation.md.** `Conversation` turns off session templating (LM-ENG-024), starts decode from
  the prefill callback (LM-ENG-040), and its `GetTokenCount` acquires the executor and can switch
  contexts (LM-ENG-065). After `CancelProcess`, the session's chained tasks are born
  `kDependentTaskCancelled` until a rewind (LM-ENG-034); the checkpoint used to strip thinking moves
  only the step, no KV copy (LM-ENG-069).
- **memory.md.** A kept conversation costs one KV snapshot until the kept session next acquires the
  executor; a switch toward a session holding a snapshot briefly holds the live cache plus two
  snapshots (LM-ENG-059, -064); measured peaks (LM-ENG-060, -066); our snapshots live in anonymous
  pages that are unmapped when freed (LM-ENG-052); releasing the last session keeps the KV buffers
  allocated (LM-ENG-098).
- **troubleshooting.md.** `kDependentTaskCancelled` after a cancel (LM-ENG-034);
  `kMaxNumTokensReached` blocks the session the same way and the C API says "Max number of tokens
  reached." (LM-ENG-035, -078); destroying a busy session blocks up to 10 minutes (LM-ENG-027);
  "Input token ids are too long" (LM-ENG-075); "Lora is not supported." (LM-ENG-099); a `-gpu`
  artisan file selects `GPU_ARTISAN`, which no open-source engine serves (LM-ENG-002, -003).
- **models/gemma4-e4b.md** (C2 context). The engine's default per-image vision token limit for E4B
  is 2,520 / 3² = 280 from the file's metadata (LM-ENG-014); start and stop token IDs from the file
  (LM-ENG-007).
- **apple-platform.md.** pthread workers named `execution_threa` / `callback_thread`, no QoS, no
  affinity (LM-ENG-032); raw KV snapshots compiled only for iOS and macOS (LM-ENG-053); residency
  sets applied under `__APPLE__` (LM-ENG-093); `max_num_tokens` not clamped on Apple (LM-ENG-008).
- **patches.md.** What `9f619cf9`, `e43f9a8d`, `ef197a04`, `5e946049`, `aec81cb7`, `ed998d80`,
  `585203d3` change in these files (LM-ENG-094, -095, -096, -082, -051, -056, -052) and the bench
  evidence for `ed998d80` (LM-ENG-060).
- **swift-api.md.** `Conversation.getTokenCount()` differs between the shipped wrapper (returns the
  C function's `int` directly, [`swift/Conversation.swift:644-647`](https://github.com/PowerBeef/agello-litert-lm/blob/85b1a77695bc90fc8b34a1f455bbe5b2c7fdceff/swift/Conversation.swift#L644-L647)) and agello-next (status code and
  out-parameter, [`swift/Conversation.swift:647-653`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L647-L653)).
- **For Agello (not a reference page).** Agello's `validate` probe (a default conversation, one
  token) is the first sampling on each engine, so per LM-ENG-062 its default parameters (TOP_P, k=1)
  would become the engine's sampler for the chat too on non-MTP steps. Worth a probe before any user
  doc describes temperature or top-k.

## Open questions

- Agello's chat temperature and top-k under LM-ENG-062: not verified by a run (a fresh engine,
  validation probe, then the chat at temperature 1 vs 0 with MTP off).
- The raw snapshot path reads and writes `MTLBuffer.contents` without waiting on tensor-buffer
  events; a switch happens between tasks, after the previous task read its results on the CPU, but
  that is not proven for every path (a task that failed mid-step, for example).
- Snapshots copy the whole allocated cache, not the filled length (T3, still open).
- What the executor does when one turn's prefill is longer than the remaining KV capacity
  (executors.md).
- After the 10 s `FinishTask` wait expires, the next task runs while the previous final callback is
  still running; whether anything relies on their order was not checked.
- Defaults of `gpu_enable_metal_residency_set` and the "version added" of most settings were not
  checked.
