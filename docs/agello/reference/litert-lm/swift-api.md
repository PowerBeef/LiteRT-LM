# LiteRT-LM Swift API (`LiteRTLM`)

> **Scope:** the Swift package in `swift/`: the `Engine` actor and the `Conversation` class, threads,
> lifecycle, streaming (`AsyncThrowingStream`, `StreamContext`, our weak-reference fix), tools
> (`Tool`, `@ToolParam`, automatic calling, our error-response patch), response formats,
> `ExperimentalFlags`, errors, `ModelInfo`, `EmbeddingEngine`, `BuildInfo` and `Telemetry` (ours),
> the `apple_fm` Foundation Models adapter; the differences between the wrapper Agello ships
> (`85b1a776`) and agello-next; how Agello uses it. **Out of scope:** the C calls underneath
> ([c-api.md](c-api.md)), what the engine and conversation do ([engine-and-sessions.md](engine-and-sessions.md),
> [conversation.md](conversation.md)), every public symbol
> ([`generated/swift-api-inventory.md`](../generated/swift-api-inventory.md)), building the
> xcframework ([build-and-release.md](build-and-release.md)).
> **Pinned to:** LiteRT-LM `7b872544` (agello-next); the shipped wrapper at `85b1a776`; Agello at
> `93122b647a`.
> **Applies to:** Apple (iOS, macOS). **Last verified:** 2026-10-15

## Summary

`LiteRTLM` is a Swift wrapper over the C API: an `Engine` actor that loads a model, `Conversation`
objects that exchange JSON messages with it, an `EmbeddingEngine` actor, `ModelInfo` for reading a
file's capabilities, and process-wide `ExperimentalFlags`. Generation calls block the thread that
makes them (`sendMessage`) or feed an `AsyncThrowingStream` from LiteRT-LM's own threads
(`sendMessageStream`). Tools are Swift types the wrapper describes to the model and runs itself, up to
25 rounds per message.

Agello ships the v0.18.0 wrapper with two Swift patches of ours, a failed tool call answered with an
error response and a stream that holds its conversation weakly, on top of Google's prebuilt v0.18.0
binary. agello-next starts from Google's newer wrapper (0.19.0 nightly, already ported to C API
1.0.0 upstream) and carries the same two patches plus `BuildInfo` and `Telemetry`. The public API is
otherwise the same, but each wrapper only compiles against its own headers, so the wrapper and the
binary move together.

## Concepts

| Type | Kind | Role |
| --- | --- | --- |
| `Engine` | `actor` | Owns the native engine handle; `initialize()`, `createConversation(with:)` |
| `EngineConfig` | `struct`, `Sendable` | Model path, backends, `maxNumTokens`, `cacheDir`, LoRA ranks, activation type |
| `Conversation` | `final class`, `Sendable` | One native conversation: `sendMessage`, `sendMessageStream`, `cancel`, token count, benchmark |
| `ConversationConfig` | `struct` | System message, initial messages, tools, sampler, thinking, response format switch, visual token budget, speculative decoding, chat template |
| `Message`, `Contents`, `Content`, `Role`, `ToolCall` | `struct`/`enum` | The JSON message model |
| `Tool`, `@ToolParam`, `ToolManager` | protocol, property wrapper, class | Tools the model can call |
| `ResponseFormat` | `struct` | JSON schema or regex constraint for one message |
| `ExperimentalFlags` | `struct` of static vars | Process-wide switches, opt-in |
| `LiteRTLMError` | `enum` | Every error the package throws |
| `ModelInfo`, `LLMCapability`, `EmbeddingCapability` | classes | Read a `.litertlm` file without an engine |
| `EmbeddingEngine`, `EmbeddingEngineConfig`, `EmbeddingOptions` | actor, structs | Embedding models |
| `BuildInfo`, `Telemetry` | enums (ours, agello-next) | Build stamp and engine probes |
| `benchmark(...)`, `BenchmarkInfo` | function, struct | Google's benchmark helper |

### A minimal session

```swift
let config = try EngineConfig(
  modelPath: modelURL.path, backend: .gpu, maxNumTokens: 8192, cacheDir: cachesURL.path)
let engine = Engine(engineConfig: config)
try await engine.initialize()  // seconds; blocks the actor
let conversation = try await engine.createConversation(
  with: ConversationConfig(systemMessage: Message("Be brief.")))
for try await chunk in conversation.sendMessageStream(Message("Hello")) {
  print(chunk.toString, terminator: "")  // text; chunk.channels["thought"] holds thinking
}
```

### Messages

`Role` raw values are `system`, `user`, `assistant` (the Swift case is `.model`) and `tool`; content
parts serialize as `{"type": "text"}`, `{"type": "image"}` with a base64 `blob` or a `path`,
`{"type": "audio"}` likewise, and `{"type": "tool_response", "name", "response"}`
[`swift/Message.swift:19-24`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Message.swift#L19-L24) [`swift/Message.swift:49-72`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Message.swift#L49-L72). <!-- c:LM-SWIFT-051 --> A reply is parsed back with
`Conversation.jsonToMessage`, which accepts `assistant` or `model` as the model role and throws
`message(.invalidContent)` when the JSON has no content, channels or tool calls [`swift/Conversation.swift:805-816`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L805-L816).
Channels (for Gemma 4, `thought`) arrive in `Message.channels`.

### Configuration checks

`EngineConfig` rejects `maxNumTokens <= 0`; `SamplerConfig` rejects `topK <= 0`, `topP` outside
`[0, 1]` and a negative temperature; `ThinkingConfig()` means thinking on with an unlimited budget
(`-1`) [`swift/Config.swift:109-120`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Config.swift#L109-L120) [`swift/Config.swift:155-169`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Config.swift#L155-L169) [`swift/Config.swift:189-192`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Config.swift#L189-L192). <!-- c:LM-SWIFT-053 --> `ConversationConfig`
drops an empty system message and gives any other system message the `system` role
[`swift/Config.swift:251-285`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Config.swift#L251-L285).

## How it works

### Engine: initialization

```mermaid
sequenceDiagram
  participant App
  participant E as Engine (actor)
  participant C as C API
  App->>E: Engine(engineConfig:)
  App->>E: await initialize()
  E->>C: litert_lm_engine_settings_create(path, backend, vision, audio)
  E->>C: setters (max tokens, cache dir, activation type, LoRA, flags)
  E->>C: litert_lm_engine_create(settings)
  Note over E: blocks the actor for the whole load
  E->>C: litert_lm_engine_settings_delete (defer)
  E-->>App: returns; handle set
```

`Engine` is an `actor` whose only mutable state is the native handle; `initialize()` is a synchronous,
actor-isolated method, so it holds the actor (and the thread running it) for the whole load, and a
second call throws `engine(.alreadyInitialized)` [`swift/Engine.swift:28-43`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Engine.swift#L28-L43) [`swift/Engine.swift:67-71`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Engine.swift#L67-L71). <!-- c:LM-SWIFT-001 -->
It creates the settings from `backend.rawValue` (`"cpu"` or `"gpu"`) and the optional vision and audio
backends, applies the configuration, creates the engine and deletes the settings
[`swift/Engine.swift:78-85`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Engine.swift#L78-L85) [`swift/Config.swift:20-40`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Config.swift#L20-L40) [`swift/Engine.swift:179-186`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Engine.swift#L179-L186). <!-- c:LM-SWIFT-002 -->

Settings it applies, in order [`swift/Engine.swift:89-177`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Engine.swift#L89-L177):

| Source | C setter | When |
| --- | --- | --- |
| `maxNumTokens` | `litert_lm_engine_settings_set_max_num_tokens` | non-nil |
| `cacheDir` | `..._set_cache_dir` | non-nil (otherwise the model's folder) |
| `activationDataType` | `..._set_activation_data_type` | non-nil |
| `loraRank`, `audioLoraRank` | `..._set_lora_rank`, `..._set_supported_lora_ranks` (and audio) | non-nil; ranks only when > 0 |
| benchmark arguments or `ExperimentalFlags.enableBenchmark` | `..._enable_benchmark` (+ token counts) | benchmark run, or flag true |
| `ExperimentalFlags.enableSpeculativeDecoding` | `..._set_enable_speculative_decoding` | non-nil |
| `ExperimentalFlags.visualTokenBudget` | `..._set_max_vision_tokens_per_image` | non-nil |
| `ExperimentalFlags.gpuEnableMetalResidencySet` | `..._set_gpu_enable_metal_residency_set` | non-nil |
| `ExperimentalFlags.enableYnnpack` (Google's, not in the shipped wrapper) | `..._set_enable_ynnpack` | non-nil |

<!-- c:LM-SWIFT-004 -->

`Backend.cpu(threadCount:)` carries a thread count, but `Engine.initialize` never calls
`litert_lm_engine_settings_set_num_threads`: the count is ignored for language models, and only
`EmbeddingEngine` applies it [`swift/Config.swift:20-40`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Config.swift#L20-L40) [`swift/Engine.swift:89-177`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Engine.swift#L89-L177) [`swift/EmbeddingEngine.swift:84-94`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/EmbeddingEngine.swift#L84-L94).
<!-- c:LM-SWIFT-003 -->

`deinit` deletes the native engine [`swift/Engine.swift:422-427`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Engine.swift#L422-L427). Each `Conversation` keeps a strong reference to
its `Engine`, so the native engine outlives every Swift conversation made from it, which is what the
C API's ownership rule requires [`swift/Conversation.swift:64-98`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L64-L98) [`swift/Engine.swift:388-394`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Engine.swift#L388-L394) [`c/engine.h:1111-1120`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/c/engine.h#L1111-L1120).
<!-- c:LM-SWIFT-005 -->

### Engine: creating a conversation

`createConversation(with:)` checks the system messages (a `systemMessage` plus an initial system
message, or two initial system messages, throw `config(.multipleSystemMessages)`)
[`swift/Engine.swift:203-214`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Engine.swift#L203-L214), builds a `ToolManager` from the tools, then translates the config to C objects that
it deletes right after `litert_lm_conversation_create`:

| `ConversationConfig` field or flag | C call | Note |
| --- | --- | --- |
| `samplerConfig` | `litert_lm_sampler_params_create(kLiteRtLmSamplerTypeTopP)` + top-k, top-p, temperature, seed | always type top-p; nil keeps the model's sampler |
| `loraPath`, `audioLoraPath` | `litert_lm_session_config_set_lora_path` (and audio) | |
| `enableSpeculativeDecoding` | `litert_lm_session_config_set_enable_speculative_decoding` | nil inherits the engine |
| `systemMessage` | `litert_lm_conversation_config_set_system_message` | the contents as a JSON array |
| `tools` | `..._set_tools` | `ToolManager.toolsJsonDescription` (`[]` with no tools) |
| `initialMessages` | `..._set_messages` | JSON array |
| `chatTemplate` | `..._set_prompt_template` | |
| `enableResponseFormat` | `..._set_constraint_provider(LlGuidance)` + `..._set_enable_constrained_decoding(true)` | |
| `ExperimentalFlags.enableConversationConstrainedDecoding` | `..._set_enable_constrained_decoding` | only when `enableResponseFormat` is false |
| `enableToolCallStreaming` and the streaming flags | `..._set_stream_tool_calls` | both the field and the flag must be true |
| `ExperimentalFlags.filterChannelContentFromKvCache` | `..._set_filter_channel_content_from_kv_cache` | non-nil only; the C++ default is true |
| `thinkingConfig` | `litert_lm_thinking_config_*`, `..._set_thinking_config` | |

When a `SamplerConfig` is given, the sampler type is always `kLiteRtLmSamplerTypeTopP`
[`swift/Engine.swift:244-268`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Engine.swift#L244-L268). <!-- c:LM-SWIFT-007 --> With `enableResponseFormat`, the conversation gets the
llguidance constraint provider and constrained decoding; otherwise constrained decoding follows
`ExperimentalFlags.enableConversationConstrainedDecoding` (default `false`), and tool-call streaming
needs both `enableToolCallStreaming` and the experimental flag [`swift/Engine.swift:330-353`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Engine.swift#L330-L353) [`swift/ExperimentalFlags.swift:78-94`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ExperimentalFlags.swift#L78-L94).
<!-- c:LM-SWIFT-008 --> The channel filter flag is passed only when set, and the C++ builder's own
default is to filter [`swift/Engine.swift:354-360`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Engine.swift#L354-L360) [`runtime/conversation/conversation.h:272-282`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/conversation.h#L272-L282). Which constraint provider and grammar
engine a JSON format reaches in our build and in the shipped binary is in
[conversation.md](conversation.md).

### Conversation: threads and lifetime

`Conversation` is a `final class` declared `Sendable`: every stored property is a `let` (the logger,
the handle, the `ToolManager`, the automatic-tool flag, the engine, the response-format switch, the
visual token budget), there is no lock and no actor, and `deinit` calls `litert_lm_conversation_delete`
[`swift/Conversation.swift:64-98`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L64-L98). <!-- c:LM-SWIFT-010 --> Two consequences:

- Nothing in Swift serializes calls on one conversation; ordering is whatever the native engine
  does with concurrent tasks (see [engine-and-sessions.md](engine-and-sessions.md)). Agello
  serializes all engine work in its own actor.
- `deinit` runs on whichever thread drops the last reference, and deleting a busy native
  conversation waits for its work (up to 10 minutes; see
  [c-api.md](c-api.md#deleting-while-work-runs)).

### Conversation: `sendMessage`

`sendMessage` is `async` only in name: it calls the blocking `litert_lm_conversation_send_message`
on the thread running the task and holds it for the whole reply [`swift/Conversation.swift:206-212`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L206-L212) [`c/conversation.h:496-497`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/c/conversation.h#L496-L497).
<!-- c:LM-SWIFT-011 --> On a Swift concurrency thread pool, that is one cooperative thread taken for
seconds.

The automatic tool loop makes at most 25 native sends (`recurringToolCallLimit = 25`). A reply
without `tool_calls` is returned if it has content or channels, and otherwise throws
`conversation(.invalidResponse)` [`swift/Conversation.swift:155-160`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L155-L160). A reply with `tool_calls` is
returned as is when `automaticToolCalling` is false; otherwise its tools run and their responses
are sent as the next message. The thinking config is passed on the first send only, and if the
25th reply still has tool calls, the loop throws
`conversation(.recurringToolCallLimitExceeded(limit: 25))` [`swift/Conversation.swift:124-169`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L124-L169) [`swift/Conversation.swift:44`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L44).
<!-- c:LM-SWIFT-012 -->

### Conversation: `sendMessageStream`

```mermaid
sequenceDiagram
  participant T as Caller task
  participant C as Conversation
  participant N as C API / engine
  participant X as execution thread
  participant K as callback thread
  T->>C: sendMessageStream(message)
  C->>C: StreamContext(continuation, weak conversation)
  C->>N: send_message_stream(json, streamCallback, passRetained(context))
  C-->>T: AsyncThrowingStream
  loop decoding
    X->>X: streamCallback(partial JSON chunk)
    X-->>T: continuation.yield(Message) if content or channels
  end
  K->>K: streamCallback(unflushed text, then final chunk)
  alt pending tool calls
    K->>T: Task { run tools; sendToStream(tool responses) }
  else no tool calls
    K-->>T: continuation.finish(); release(context)
  end
```

`sendMessageStream` returns an `AsyncThrowingStream<Message, Error>` whose build closure starts the
native stream at once; it sets no `onTermination` handler, so cancelling the consuming task or
dropping the stream does not stop the generation: only `cancel()` does [`swift/Conversation.swift:478-521`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L478-L521).
<!-- c:LM-SWIFT-013 --> `sendToStream` serializes the message, builds the per-turn optional args,
retains a `StreamContext` with `Unmanaged.passRetained` and passes it as `callback_data`; if the
stream fails to start, it releases the context and throws
`conversation(.failedToStartStream(status:message:))` [`swift/Conversation.swift:574-590`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L574-L590). <!-- c:LM-SWIFT-014 -->

`streamCallback` is the C callback, so it runs on LiteRT-LM's threads: the execution thread for
chunks streamed during decoding (`Tasks::Decode` receives the callback), and the callback thread for
the task's final callback, which first delivers any text not yet flushed (`SendCompleteMessage`) and
then the final chunk [`runtime/core/tasks.cc:792-799`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/core/tasks.cc#L792-L799) [`runtime/conversation/internal_callback_util.cc:95-149`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L95-L149) [`swift/Conversation.swift:574-590`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L574-L590) [`runtime/framework/resource_management/threaded_execution_manager.cc:1111-1117`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/framework/resource_management/threaded_execution_manager.cc#L1111-L1117)
[`runtime/framework/resource_management/threaded_execution_manager.cc:526-538`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/framework/resource_management/threaded_execution_manager.cc#L526-L538). <!-- c:LM-SWIFT-015 --> For each chunk it reads the final flag, the error and
the text; an error finishes the stream with `conversation(.invalidResponse(<error text>))` and
releases the context; text is parsed as JSON, `tool_calls` are added to `pendingToolCalls`, and only
chunks with `content` or `channels` are yielded [`swift/Conversation.swift:871-903`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L871-L903) [`swift/Conversation.swift:905-928`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L905-L928).
<!-- c:LM-SWIFT-016 --> A chunk that carries only tool calls is not yielded, so the consumer sees
tool calls only when they share a chunk with content (Agello reports the tools that ran from its own
tool code instead).

On the final chunk, with no pending tool calls, the callback finishes the stream and releases the
context. With pending calls (and fewer than 25 rounds so far) it starts a Swift `Task` that runs the
tools and sends their responses through `sendToStream` with a new retained context, then releases the
old one [`swift/Conversation.swift:930-971`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L930-L971). This path never looks at `automaticToolCalling` (the context doesn't carry
it), so a streamed reply runs the conversation's tools even when automatic tool calling is off
[`swift/Conversation.swift:826-844`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L826-L844) [`swift/Conversation.swift:478-521`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L478-L521). <!-- c:LM-SWIFT-017 --> The follow-up send passes the original repetition,
n-gram, suppressed-token, output-limit and response-format settings, but not the extra context or
the thinking config [`swift/Conversation.swift:930-971`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L930-L971).

### Error chunks and the retained context

`streamCallback` releases the context retained for a send on the first final chunk and on every chunk
that carries an error, final or not, both in the wrapper Agello ships and at the pin
([`swift/Conversation.swift:867-903`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L867-L903);
shipped: [`swift/Conversation.swift:830-848`](https://github.com/PowerBeef/agello-litert-lm/blob/85b1a77695bc90fc8b34a1f455bbe5b2c7fdceff/swift/Conversation.swift#L830-L848),
[`swift/Conversation.swift:579-592`](https://github.com/PowerBeef/agello-litert-lm/blob/85b1a77695bc90fc8b34a1f455bbe5b2c7fdceff/swift/Conversation.swift#L579-L592)).
<!-- c:X-GAP-016 --> It is safe only if the native side ends each send with one terminal chunk, and one
path, traced in both v0.18.0 and agello-next, sends two:

1. A Gemma 4 conversation declares tools, and the model writes a tool-call block that the FC parser
   rejects.
2. When the block's end fence streams, the internal callback's `SendMessage` hands the failed
   `ToMessage` result to the user callback: `InvalidArgument`, because `return_error_on_parse_failure`
   defaults to true and neither C nor Swift can change it. The C layer makes it a final error chunk
   ([c-api.md](c-api.md#threads-and-callbacks)), and Swift releases the context. The native stream
   goes on.
3. At the end of decode, `SendCompleteMessage` parses the whole reply again, fails on the same block,
   and sends a second final error chunk. Swift reads the context it already released and releases it
   again: a use-after-free and over-release on LiteRT-LM's thread.

<!-- c:X-GAP-015 --> Sources: [`runtime/conversation/internal_callback_util.cc:61-71`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L61-L71),
[`runtime/conversation/internal_callback_util.cc:447-466`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L447-L466),
[`runtime/conversation/internal_callback_util.cc:330-343`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L330-L343),
[`runtime/conversation/internal_callback_util.cc:95-149`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L95-L149),
[`runtime/conversation/model_data_processor/gemma4_data_processor.cc:146-155`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/conversation/model_data_processor/gemma4_data_processor.cc#L146-L155),
[`runtime/conversation/model_data_processor/data_utils.cc:86-108`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/data_utils.cc#L86-L108),
[`runtime/components/tool_use/parser_utils.cc:138-151`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/tool_use/parser_utils.cc#L138-L151),
[`runtime/conversation/model_data_processor/model_data_processor.h:123-133`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/model_data_processor.h#L123-L133);
v0.18.0: [`runtime/conversation/internal_callback_util.cc:447-466`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/runtime/conversation/internal_callback_util.cc#L447-L466),
[`runtime/conversation/internal_callback_util.cc:95-149`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/runtime/conversation/internal_callback_util.cc#L95-L149).
Agello streams its chat replies on conversations that declare its tools whenever tools or web search
are on, so a malformed tool call reaches this path in the app. <!-- c:MOD-TOK-053 --> <!-- c:X-GAP-025 -->

The other terminal paths of the conversation layer deliver one terminal event: `SendCompleteMessage`
returns after reporting an error without sending the end marker, each error, cancel and failure branch
of the internal callback makes one call and returns, and in agello-next the deferred end marker is
suppressed when the channel-filter refill reports an error itself
([`runtime/conversation/internal_callback_util.cc:292-343`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L292-L343),
[`runtime/conversation/conversation.cc:702-723`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/conversation.cc#L702-L723)).
The wrappers have a second, Swift-side gap: a non-final chunk whose text isn't valid JSON or doesn't
convert to a `Message` also finishes the stream and releases the context while the native stream goes
on ([`swift/Conversation.swift:905-928`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L905-L928)).

Either gap closes if `StreamContext` records that it has finished and releases only once; on the
native side, `SendMessage` could end the stream after an error.

### Our fix: a stream holds its conversation weakly

Upstream's `StreamContext` held `let conversation: Conversation` [`swift/Conversation.swift:781-784`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Conversation.swift#L781-L784). Ours holds
`weak var conversation: Conversation?` (ours: `0b2ca12fc4` on agello-next, `85b1a77695` in the
shipped wrapper) [`swift/Conversation.swift:826-844`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L826-L844) [`swift/Conversation.swift:798-800`](https://github.com/PowerBeef/agello-litert-lm/blob/85b1a77695bc90fc8b34a1f455bbe5b2c7fdceff/swift/Conversation.swift#L798-L800). <!-- c:LM-SWIFT-019 -->

Why it matters, step by step:

1. The engine marks a task finished only after its final callback returns: the callback-thread
   closure calls the callback, then `UpdateTaskState` [`runtime/framework/resource_management/threaded_execution_manager.cc:526-538`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/framework/resource_management/threaded_execution_manager.cc#L526-L538) [`runtime/framework/resource_management/threaded_execution_manager.cc:515-540`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/runtime/framework/resource_management/threaded_execution_manager.cc#L515-L540).
2. `streamCallback` releases its context at the end of the final callback.
3. With a strong reference, if the app had already dropped the conversation (a stream abandoned
   early, or a conversation replaced while its cancelled stream was finishing), that release was the
   last one: `Conversation.deinit` ran on the callback thread and called
   `litert_lm_conversation_delete`.
4. Deleting the conversation destroys its session, and `SessionAdvanced`'s destructor waits for the
   session's tasks (`Engine::kDefaultTimeout`, 10 minutes) without cancelling them
   [`runtime/core/session_advanced.cc:484-489`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/core/session_advanced.cc#L484-L489) [`runtime/core/session_advanced.cc:478-479`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/runtime/core/session_advanced.cc#L478-L479) [`runtime/engine/engine.h:453`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine.h#L453). The task it waits for is the one whose callback is
   still running, so the wait could only end at the timeout, and the engine's single callback thread
   was blocked the whole time.

<!-- c:LM-SWIFT-020 -->

With the weak reference the conversation is deleted where the app drops it, and that deletion waits
for the stream's work to finish. If the conversation is gone when a tool round is due, the follow-up
`Task` finishes the stream with `conversation(.notAlive)` [`swift/Conversation.swift:930-971`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L930-L971) [`swift/Conversation.swift:826-844`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L826-L844).
<!-- c:LM-SWIFT-021 --> Dropping a conversation in the middle of a stream still blocks the dropping
thread until the generation ends; call `cancel()` first. Agello's device probe 9 checks the fix
(Agello `Vendor/README.md`).

### Cancellation

`cancel()` calls `litert_lm_conversation_cancel_process` [`swift/Conversation.swift:594-599`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L594-L599). A cancelled stream then
ends with an error chunk whose text is `CANCELLED: Task cancelled`, which the wrapper throws as
`conversation(.invalidResponse("CANCELLED: Task cancelled"))`; the wrapper has no cancellation case
of its own. Upstream documents that a session is not reusable after a cancel
[`runtime/conversation/internal_callback_util.cc:302-308`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L302-L308) [`c/conversation.cc:44-71`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/c/conversation.cc#L44-L71) [`runtime/core/session_advanced.h:192-207`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/session_advanced.h#L192-L207). <!-- c:LM-SWIFT-022 --> Agello therefore
treats every cancelled conversation as spent and reseeds a new one from saved history (Agello
`ChatEngine.swift:80-84`).

### Tools

A tool is a `Decodable` type with a static `name` and `description`, an empty `init()` and
`run() async throws -> Any`; its parameters are properties wrapped in `@ToolParam(description:)`,
of type `String`, `Int`, `Bool`, `Double` or `Float` (JSON `number`), an `Optional` of one
(`nullable`), or an `Array` of one [`swift/Tool.swift:211-221`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Tool.swift#L211-L221) [`swift/Tool.swift:18-49`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Tool.swift#L18-L49) [`swift/Tool.swift:84-101`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Tool.swift#L84-L101).
<!-- c:LM-SWIFT-025 --> A parameter is required unless it is optional or has a default; reading a
required parameter that was never set is a `fatalError` [`swift/Tool.swift:84-101`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Tool.swift#L84-L101) [`swift/Tool.swift:107-121`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Tool.swift#L107-L121).

**Schema.** `getSchema()` reflects the tool with `Mirror` and emits
`{"type": "function", "function": {"name", "description"}}`; when the tool has at least one
`@ToolParam`, it adds `"parameters": {"type": "object", "properties"}`, with a `"required"` list
when any parameter is required. With `ExperimentalFlags.convertCamelToSnakeCaseInToolDescription` (default `true`), tool and
parameter names are converted from camelCase to snake_case, after dropping the property wrapper's
leading underscore [`swift/Tool.swift:225-273`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Tool.swift#L225-L273) [`swift/ExperimentalFlags.swift:65-76`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ExperimentalFlags.swift#L65-L76). <!-- c:LM-SWIFT-027 --> The conversation passes the
array of schemas to the C API as its tools, and the model's chat template renders them (see
[conversation.md](conversation.md)).

**Execution.** `ToolManager.execute(name:arguments:)` looks the name up in the registry built at
conversation creation, makes a default instance to collect default values (and `null` for optional
parameters), merges the model's arguments over them, decodes a new instance with `JSONDecoder` and
runs it; the result is normalized for JSON (dictionaries and arrays recursively, strings, numbers and
booleans as they are, `Void` as `""`, anything else through `String(describing:)`)
[`swift/ToolManager.swift:64-118`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ToolManager.swift#L64-L118) [`swift/ToolManager.swift:124-151`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ToolManager.swift#L124-L151). The snake_case flag is read again on every execution, for the default
keys and the decoder's key strategy, although its documentation says it is read only when a
conversation is created; the registry's names, on the other hand, are fixed at creation
[`swift/ToolManager.swift:64-118`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ToolManager.swift#L64-L118) [`swift/ToolManager.swift:32-56`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ToolManager.swift#L32-L56) [`swift/ExperimentalFlags.swift:65-76`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ExperimentalFlags.swift#L65-L76). <!-- c:LM-SWIFT-028 --> Changing the flag while
conversations are alive can therefore make arguments fail to decode.

**Our patch: failed calls answered with an error.** Upstream's `handleToolCalls` skipped calls whose
arguments weren't a JSON object and turned any error from a tool into
`conversation(.toolExecutionError(name:error:))`, which ended the whole reply [`swift/Conversation.swift:386-409`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Conversation.swift#L386-L409).
Ours (ours: `f3408e0671` on agello-next, `a3c0d0ef72` in the shipped wrapper) answers every call
[`swift/Conversation.swift:393-425`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L393-L425) [`swift/Conversation.swift:427-456`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L427-L456) [`swift/Conversation.swift:315-347`](https://github.com/PowerBeef/agello-litert-lm/blob/85b1a77695bc90fc8b34a1f455bbe5b2c7fdceff/swift/Conversation.swift#L315-L347): <!-- c:LM-SWIFT-023 -->

| The model's call | Tool response sent back |
| --- | --- |
| `arguments` missing or `null` | the tool runs with no arguments (`{}`) |
| `arguments` an object, or a JSON string holding an object | the tool runs |
| `arguments` anything else | `{"error": "The arguments for '<name>' must be a JSON object."}` |
| an unknown tool name (`tool(.notFound)`) | `{"error": "There is no tool named '<name>'. Use only the tools you were given, or answer without one."}` |
| arguments that don't decode (`DecodingError`) | `{"error": "The arguments for '<name>' don't match its parameters. Check their names and types, then try again."}` |
| the tool throws anything else | `{"error": "'<name>' failed: <localizedDescription>"}` |

The model reads the error in the tool response and can correct the call or answer without it. The
`toolExecutionError` case still exists in `LiteRTLMError`, but nothing throws it any more
[`swift/LiteRTLMError.swift:323-334`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/LiteRTLMError.swift#L323-L334). <!-- c:LM-SWIFT-024 --> Agello's device probe 15 checks this (Agello
`Vendor/README.md`).

### Response formats

`ResponseFormat.json(schema:)` takes a JSON schema as a string (checked to parse as JSON) or a
dictionary; `ResponseFormat.regex(pattern:)` takes a pattern; they map to
`kLiteRtLmConstraintTypeJsonSchema` and `kLiteRtLmConstraintTypeRegex` [`swift/ResponseFormat.swift:20-72`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ResponseFormat.swift#L20-L72). Passing
one to a conversation created without `enableResponseFormat: true` throws
`conversation(.responseFormatNotEnabled)` [`swift/Conversation.swift:124-169`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L124-L169) [`swift/Conversation.swift:478-521`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L478-L521). <!-- c:LM-SWIFT-030 -->
The format is not applied to a user message when automatic tool calling is on and the conversation
has tools: then it applies only to the tool-response message, so the final answer after the tools
is the constrained one [`swift/Conversation.swift:100-110`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L100-L110). <!-- c:LM-SWIFT-031 -->

### `ExperimentalFlags`

`ExperimentalFlags` is a struct of `static var`s: process-wide, shared by every engine and
conversation in the process, with no synchronization. Each setter ignores the new value and logs an
error unless `optIntoExperimentalAPIs()` was called first, and the opt-in can't be undone
[`swift/ExperimentalFlags.swift:21-46`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ExperimentalFlags.swift#L21-L46). <!-- c:LM-SWIFT-032 -->

| Flag | Default | Read when | Effect |
| --- | --- | --- | --- |
| `enableBenchmark` | `false` | `Engine.initialize`; every `getBenchmarkInfo()` | benchmark info collected; `getBenchmarkInfo` throws `benchmarkNotEnabled` while false |
| `convertCamelToSnakeCaseInToolDescription` | `true` | conversation creation and every tool execution | snake_case tool and parameter names |
| `enableConversationConstrainedDecoding` | `false` | conversation creation | constrained decoding without a response format |
| `enableConversationToolCallStreaming` | `false` | conversation creation | with the config field, streams tool-call tokens |
| `conversationToolCallStreamingChannelName` | `"tool_call"` | conversation creation | channel for streamed tool calls |
| `enableSpeculativeDecoding` | `nil` | `Engine.initialize` | engine-level MTP on or off |
| `visualTokenBudget` | `nil` | `Engine.initialize`; every native send whose conversation has no budget of its own | maximum vision tokens per image (engine); per-message budget |
| `filterChannelContentFromKvCache` | `nil` | conversation creation | keeps thinking out of the KV cache when true (C++ default true) |
| `gpuEnableMetalResidencySet` | `nil` | `Engine.initialize` | `MTLResidencySet` for the GPU backend |
| `enableYnnpack` (not in the shipped wrapper) | `nil` (or `true` from the environment) | `Engine.initialize` | YNNPACK before XNNPACK on CPU |

[`swift/ExperimentalFlags.swift:78-94`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ExperimentalFlags.swift#L78-L94) [`swift/ExperimentalFlags.swift:96-125`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ExperimentalFlags.swift#L96-L125) [`swift/ExperimentalFlags.swift:127-145`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ExperimentalFlags.swift#L127-L145) [`swift/ExperimentalFlags.swift:158-173`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ExperimentalFlags.swift#L158-L173) [`swift/ExperimentalFlags.swift:175-187`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ExperimentalFlags.swift#L175-L187) [`swift/ExperimentalFlags.swift:189-203`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ExperimentalFlags.swift#L189-L203)
[`swift/Engine.swift:141-177`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Engine.swift#L141-L177) [`swift/Conversation.swift:660-665`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L660-L665) [`swift/Conversation.swift:264-269`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L264-L269)

`visualTokenBudget` is read at two levels: `Engine.initialize` turns it into the engine's maximum
vision tokens per image, and every native send uses `ConversationConfig.visualTokenBudget` or, if
that is nil, the flag's value at send time. In agello-next the per-send read is in
`makeOptionalArgs`; the shipped wrapper reads it inline in `attemptSendMessage` and `sendToStream`
[`swift/Engine.swift:162-166`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Engine.swift#L162-L166) [`swift/Conversation.swift:264-269`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L264-L269) [`swift/Conversation.swift:197-200`](https://github.com/PowerBeef/agello-litert-lm/blob/85b1a77695bc90fc8b34a1f455bbe5b2c7fdceff/swift/Conversation.swift#L197-L200) [`swift/Conversation.swift:487-490`](https://github.com/PowerBeef/agello-litert-lm/blob/85b1a77695bc90fc8b34a1f455bbe5b2c7fdceff/swift/Conversation.swift#L487-L490) [`swift/ExperimentalFlags.swift:158-173`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ExperimentalFlags.swift#L158-L173).
<!-- c:LM-SWIFT-033 --> `enableYnnpack` is Google's: it is in the upstream wrapper at the
agello-next base (`df6df5b8`), so agello-next has it and the shipped v0.18.0-based wrapper, whose
`ExperimentalFlags` ends with `gpuEnableMetalResidencySet`, does not. When unset, its getter returns
`true`, without any opt-in, if the environment variable `ENABLE_YNNPACK` or `enable_ynnpack` is
`"true"` or a launch argument contains `enable_ynnpack=true`, and `nil` otherwise [`swift/ExperimentalFlags.swift:205-238`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ExperimentalFlags.swift#L205-L238) [`swift/ExperimentalFlags.swift:189-207`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/swift/ExperimentalFlags.swift#L189-L207).
<!-- c:LM-SWIFT-034 --> Google's `benchmark(...)` function opts the process in and sets
`enableBenchmark` for its duration [`swift/Benchmark.swift:63-75`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Benchmark.swift#L63-L75). <!-- c:LM-SWIFT-035 -->

Because the flags are global, an app that runs two engines (Agello's chat engine and embedder) or
changes a flag between conversations must set every flag immediately before the call that reads it.
`Engine.updateGPUEnableMetalResidencySet(_:)` is the one engine method gated on the opt-in
[`swift/Engine.swift:405-420`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Engine.swift#L405-L420).

### Errors

`LiteRTLMError` has seven top-level cases, `engine`, `embeddingEngine`, `conversation`, `config`,
`tool`, `message` and `modelInfo`, each wrapping a nested enum [`swift/LiteRTLMError.swift:19-27`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/LiteRTLMError.swift#L19-L27).
<!-- c:LM-SWIFT-036 --> In agello-next every C call goes through three helpers: `check` turns a
non-OK status into the caller's error with the text `"<function> failed with status <code>: <last
error message>"`, reading and clearing the thread-local message right after the call on the same
thread; `create` also rejects a NULL handle; `get` returns an out-parameter
[`swift/LiteRTLMError.swift:84-91`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/LiteRTLMError.swift#L84-L91) [`swift/LiteRTLMError.swift:105-115`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/LiteRTLMError.swift#L105-L115) [`swift/LiteRTLMError.swift:62-71`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/LiteRTLMError.swift#L62-L71). <!-- c:LM-SWIFT-037 --> The numeric status
survives only inside that string, except in `failedToStartStream(status:message:)`
[`swift/Conversation.swift:574-590`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L574-L590). Equality between two `EngineError`s with detail strings is true when the strings
match or either is empty [`swift/LiteRTLMError.swift:223-250`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/LiteRTLMError.swift#L223-L250).

| Case | Thrown by |
| --- | --- |
| `engine(.alreadyInitialized / .notInitialized)` | `initialize()` twice; `createConversation` before `initialize` |
| `engine(.failedToCreateSettings / .failedToCreateEngine / ...)` | a C call during `initialize` or `createConversation`, with the C message |
| `config(.multipleSystemMessages / .invalidMaxNumTokens / .invalidTopK / ...)` | config checks in Swift |
| `conversation(.invalidResponse(...))` | a C failure during a send, an error chunk (including `CANCELLED: ...`), a reply without content |
| `conversation(.recurringToolCallLimitExceeded(limit: 25))` | the 25th reply of a `sendMessage` tool loop still has tool calls |
| `conversation(.failedToStartStream(status:message:))` | `send_message_stream` refused to start |
| `conversation(.responseFormatNotEnabled)` | a `ResponseFormat` without `enableResponseFormat` |
| `conversation(.benchmarkNotEnabled / .benchmarkInfoUnavailable)` | `getBenchmarkInfo()` |
| `tool(.notFound)` | `ToolManager.execute` (answered to the model by our patch) |
| `message(.invalidContent / .failedToConvertToJson)` | `jsonToMessage` |
| `modelInfo(.failedToLoadModel)` | `ModelInfo(throwingModelPath:)` |

### `ModelInfo`

`ModelInfo(modelPath:)` (failable) and `ModelInfo(throwingModelPath:)` open the file with
`litert_lm_loaded_file_create` [`swift/ModelInfo.swift:298-312`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ModelInfo.swift#L298-L312). The accessors don't throw in either
wrapper. In agello-next, `modelInfoValue` turns a failed C call or a `NotFound` into a fallback
(`maxContextTokens()` 0, `maxVisionTokenBudget()` -1, `false` for capabilities, `nil` for arrays and
strings), and `fetchIntArray` returns `nil` when the count query fails [`swift/ModelInfo.swift:78-93`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ModelInfo.swift#L78-L93) [`swift/ModelInfo.swift:367-370`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ModelInfo.swift#L367-L370) [`swift/ModelInfo.swift:95-117`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ModelInfo.swift#L95-L117).
The shipped wrapper has no `modelInfoValue`: it calls the older C functions that return their value
directly (`maxContextTokens()` returns the C result as is), and its `fetchIntArray` returns `nil`
when the count is negative [`swift/ModelInfo.swift:79-96`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/swift/ModelInfo.swift#L79-L96) [`swift/ModelInfo.swift:328-330`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/swift/ModelInfo.swift#L328-L330). Agello reads it before every load to intersect its plan with the file's
real capabilities (MTP, thinking, function calling, modalities, context and visual-token limits).

### `EmbeddingEngine`

An `actor` like `Engine`; `initialize()` creates the native embedding engine (applying
`Backend.cpu(threadCount:)` when the count is positive), `computeEmbedding` and
`computeEmbeddingBatch` run synchronously on the actor, and `close()` or `deinit` delete the native
engine [`swift/EmbeddingEngine.swift:30-52`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/EmbeddingEngine.swift#L30-L52) [`swift/EmbeddingEngine.swift:84-94`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/EmbeddingEngine.swift#L84-L94) [`swift/EmbeddingEngine.swift:281-287`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/EmbeddingEngine.swift#L281-L287). <!-- c:LM-SWIFT-041 -->
`EmbeddingOptions` exposes `normalize`, `insertSpecialTokens`, `outputSize` and
`visionTokensPerImage`, but not the input overflow strategy, so Swift callers always get the C++
default: an input longer than the longest signature is an error [`swift/EmbeddingEngineConfig.swift:62-76`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/EmbeddingEngineConfig.swift#L62-L76) [`runtime/engine/embedding_engine.h:31-52`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/embedding_engine.h#L31-L52).
<!-- c:LM-SWIFT-042 -->

### `BuildInfo` and `Telemetry` (ours)

Both exist only in agello-next.

- `BuildInfo` (ours: `c7c744ebc6`): `json` returns `litert_lm_get_build_info()`, `fields` decodes it
  as `[String: String]`, and `id` is its `build_id` (12 hex digits for `--config=agello` builds,
  `dev` otherwise) [`swift/BuildInfo.swift:18-30`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/BuildInfo.swift#L18-L30) [`c/build_info.h:29-33`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/c/build_info.h#L29-L33). <!-- c:LM-SWIFT-043 -->
- `Telemetry` (ours: `4aed9274a4`, `a3a0c0adeb`): `Level` from `off` (0) to `deep` (4), a `level`
  property over `litert_lm_telemetry_get_level`/`_set_level`, `snapshotJSON()` (which frees the C
  string), `snapshot()`, `reset()` and `writeTrace(to:)` [`swift/Telemetry.swift:18-62`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Telemetry.swift#L18-L62). <!-- c:LM-SWIFT-044 -->
  Levels and environment variables are in [c-api.md](c-api.md#our-additions-telemetry-and-build-info).

### The Foundation Models adapter (`apple_fm`)

`swift/apple_fm/` is a second library product, `LiteRTLMFoundationModels`, that makes a LiteRT-LM
model a backend for Apple's Foundation Models framework [`Package.swift:59-69`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/Package.swift#L59-L69). It compiles only when
`FoundationModels` can be imported and the compiler is 6.4 or newer, and `LiteRTLanguageModel` is
available from iOS 27 and macOS 27. Each turn, its executor builds a fresh `Conversation` from the
whole Foundation Models transcript (system text, history, the last prompt or tool output); one
`Engine` per configuration is shared process-wide by an engine cache
[`swift/apple_fm/LiteRTLanguageModel.swift:31-39`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/apple_fm/LiteRTLanguageModel.swift#L31-L39) [`swift/apple_fm/LiteRTLanguageModel.swift:51-52`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/apple_fm/LiteRTLanguageModel.swift#L51-L52) [`swift/apple_fm/LiteRTLanguageModel.swift:163-181`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/apple_fm/LiteRTLanguageModel.swift#L163-L181) [`swift/apple_fm/LiteRTLanguageModel.swift:146-152`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/apple_fm/LiteRTLanguageModel.swift#L146-L152). <!-- c:LM-SWIFT-049 --> Tools are described
in the prompt and detected in the text (a JSON object with a `tool_call` key), not through
LiteRT-LM's tool calling; guided generation adds the schema to the prompt and constrains decoding
with it.

When the Swift API inventory was generated at the pin, the `LiteRTLMFoundationModels` target didn't
compile against the macOS 27 SDK, so the inventory lists only `LiteRTLM`
([`generated/swift-api-inventory.md`](../generated/swift-api-inventory.md)). <!-- c:LM-SWIFT-050 -->
The compiler errors were not recorded; the protocol names the adapter uses (`LanguageModel`,
`LanguageModelExecutor`, `LanguageModelExecutorGenerationChannel`) do exist in the macOS 27 SDK's
`FoundationModels` interface, so the failure is in details not checked here. Agello doesn't use
the adapter.

## Shipped wrapper and agello-next

| | Shipped (`agello` `85b1a776`) | Documented (`agello-next` `7b872544`) |
| --- | --- | --- |
| Base | Google's v0.18.0 Swift wrapper | Google's 0.19.0-nightly.20261013 wrapper |
| Binary | Google's v0.18.0 `CLiteRTLM.xcframework` (iOS) and `CLiteRTLM_mac.xcframework` (macOS), by URL and checksum (shipped [`Package.swift:35-55`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/Package.swift#L35-L55)) | one local `dist/CLiteRTLM.xcframework` with iPhone and Mac slices, built from the branch (ours: `3b2327cfcd`) [`Package.swift:35-40`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/Package.swift#L35-L40) |
| C API | 0.2.0: `guard let` on returned pointers, unchecked `void` setters [`swift/Engine.swift:78-83`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/swift/Engine.swift#L78-L83) [`swift/Engine.swift:143-151`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/swift/Engine.swift#L143-L151) | 1.0.0 (Google's port): `check` / `create` / `get` on every call [`swift/Engine.swift:78-85`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Engine.swift#L78-L85) |
| `LiteRTLMError.getLastErrorCode()` | present, calls `litert_lm_get_last_error_code` [`swift/LiteRTLMError.swift:56-64`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/swift/LiteRTLMError.swift#L56-L64) | removed (the C function is gone) |
| `ExperimentalFlags.enableYnnpack` (Google's) | absent [`swift/ExperimentalFlags.swift:189-207`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/swift/ExperimentalFlags.swift#L189-L207) | present, from the upstream base |
| `BuildInfo`, `Telemetry` | absent | ours |
| Tool-call error responses | ours: `a3c0d0ef72` | ours: `f3408e0671` |
| Weak `StreamContext.conversation` | ours: `85b1a77695` | ours: `0b2ca12fc4` |
| Public declarations of `Engine`, `Conversation`, `EmbeddingEngine`, `ModelInfo` | same | same |

<!-- c:LM-SWIFT-045 --> <!-- c:LM-SWIFT-039 --> <!-- c:LM-SWIFT-064 -->

The shipped `swift/Conversation.swift` also carries a comment-only commit of ours, `7c4ee32d59`,
that updates the patch notice to the renamed fork; the notice at the top of the file names both
patches [`swift/Conversation.swift:15-20`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L15-L20).

The wrappers are tied to their headers: the shipped `Engine.swift` calls
`litert_lm_engine_settings_create` with four arguments and tests the returned pointer, while the
1.0.0 function takes a fifth, out-parameter, and returns a status (shipped [`swift/Engine.swift:78-83`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/swift/Engine.swift#L78-L83);
1.0.0 [`c/engine.cc:639-660`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/c/engine.cc#L639-L660); v0.18.0's `litert_lm_engine_create` likewise returns a pointer
[`c/engine.h:820-828`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/c/engine.h#L820-L828)). <!-- c:LM-SWIFT-048 --> Moving
Agello to agello-next means replacing the vendored wrapper and the binary in one step.

Agello at `93122b647a` vendors the shipped wrapper unchanged: `Vendor/LiteRT-LM/Package.swift` and
every file under `Vendor/LiteRT-LM/swift/` are byte-identical to the fork's `85b1a776`
[`Package.swift:35-55`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/Package.swift#L35-L55). <!-- c:LM-SWIFT-046 -->

## How Agello uses it

All of it lives in `LiteRTChatEngine` (`Agello/Inference/ChatEngine.swift`) and `LiteRTEmbedder`
(`Agello/Knowledge/Embedding.swift`) at `93122b647a`.

| What | How | Agello source |
| --- | --- | --- |
| Flags | opts in, `enableBenchmark = true` (for the stats line), `filterChannelContentFromKvCache = true`, then `enableSpeculativeDecoding` from its plan before each engine | `ChatEngine.swift:141-147`, `:174` |
| Engine | `.gpu` or `.cpu()`, vision and audio executors on `.cpu()` when enabled, `maxNumTokens` from its budget, a cache directory: `kv-<maxNumTokens>` per context size with Google's binary, one `kv-all-<build id>` for every size with ours <!-- c:X-CFG-008 -->; `initialize()` awaited from a detached task | `ChatEngine.swift:273-287` |
| Validation | a one-token reply on a throwaway conversation before committing to a new engine | `ChatEngine.swift:295-307` |
| Chat conversation | system prompt, replayed history, tools (`get_current_date_time`, a calculator, `web_search`), `SamplerConfig`, `ThinkingConfig(budget:)`, visual token budget, MTP per conversation | `ChatEngine.swift:309-334` |
| JSON helpers | `enableResponseFormat: true`, `enableSpeculativeDecoding: false`, `sendMessage(..., responseFormat: .json(schema:))` | `ChatEngine.swift:490-498` |
| Replies | `sendMessageStream(_:maxOutputTokens:)`, `Task.isCancelled` checked per chunk, `cancel()` on cancellation or when output would overflow the KV room | `ChatEngine.swift:684-711` |
| Cancel | `cancel()` on the helper or main conversation; a cancelled conversation is reseeded | `ChatEngine.swift:80-84`, `:754-759` |
| Unload | `unloadLocked` sets the conversation to nil, then the engine | `ChatEngine.swift:247-249` |
| Embeddings | `EmbeddingEngine` on `.gpu`, then `.cpu()`; `EmbeddingOptions(normalize: true, outputSize: 256)`; batches of 8 | `Embedding.swift:22-35`, `:41-53`, `:76-94` |

Agello's load opts into experimental APIs, sets `enableBenchmark` and
`filterChannelContentFromKvCache` to true, and sets `enableSpeculativeDecoding` from its plan right
before building each engine. <!-- c:LM-SWIFT-056 --> It builds engines with the vision and audio
executors on the CPU backend and awaits `initialize()` from a detached task.
<!-- c:LM-SWIFT-057 --> Because a load can report success with a section that failed to map, it
proves each new engine with a one-token reply before using it. <!-- c:LM-SWIFT-058 --> Its JSON
helpers turn speculative decoding off, because in its experience drafted tokens bypassed the JSON
mask. <!-- c:LM-SWIFT-059 --> It cancels through `Conversation.cancel()` and reseeds afterwards.
<!-- c:LM-SWIFT-060 --> Its embedder asks for normalized 256-dimension vectors (EmbeddingGemma's
Matryoshka truncation), in native batches of 8, GPU first. <!-- c:LM-SWIFT-061 -->

On unload, `unloadLocked` sets its `Conversation` to nil before its `Engine`. The native
conversation is deleted before the engine regardless of that order: each Swift `Conversation` holds
its `Engine` strongly, so `Engine.deinit` (`litert_lm_engine_delete`) runs only after every
`Conversation`, including any helper conversation still referenced, has been released
(`Conversation.deinit` calls `litert_lm_conversation_delete`) [`swift/Conversation.swift:73`](https://github.com/PowerBeef/agello-litert-lm/blob/85b1a77695bc90fc8b34a1f455bbe5b2c7fdceff/swift/Conversation.swift#L73) [`swift/Conversation.swift:94-98`](https://github.com/PowerBeef/agello-litert-lm/blob/85b1a77695bc90fc8b34a1f455bbe5b2c7fdceff/swift/Conversation.swift#L94-L98) [`swift/Engine.swift:303-309`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/swift/Engine.swift#L303-L309) [`swift/Engine.swift:337-342`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/swift/Engine.swift#L337-L342)
(Agello `ChatEngine.swift:247-249`).

## On Apple

- **The package.** Platforms `iOS 15` and `macOS 12` in both `Package.swift` files; our xcframework
  itself requires iOS 27 and macOS 27 ([build-and-release.md](build-and-release.md)). The shipped
  package picks `CLiteRTLM` on iOS and `CLiteRTLM_mac` on macOS by platform condition
  [`Package.swift:35-55`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/Package.swift#L35-L55); agello-next has one xcframework for both [`Package.swift:35-40`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/Package.swift#L35-L40). Agello runs the iPhone app on
  the Mac as "Designed for iPad/iPhone"; which slice that loads is covered in
  [apple-platform.md](../apple-platform.md).
- **Swift language mode.** The package declares `swift-tools-version: 5.9` [`Package.swift:1`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/Package.swift#L1)
  [`Package.swift:1`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/Package.swift#L1), so it builds in Swift 5
  mode even inside Agello's Swift 6 app; that is why the unsynchronized `static var`s in
  `ExperimentalFlags` compile.
- **Threads.** `Engine` and `EmbeddingEngine` loads and embedding calls block a Swift concurrency
  thread for their duration; `sendMessage` blocks one for the whole reply; streamed chunks arrive on
  LiteRT-LM's threads, not the main actor. Hop to the main actor before touching UI state (Agello
  yields into its own `AsyncThrowingStream` and consumes it from `AppState`).
- **Metal residency set.** Off unless `ExperimentalFlags.gpuEnableMetalResidencySet` is set before
  `initialize()`; Agello leaves it unset.

## Settings

Swift-level settings (`EngineConfig`, `ConversationConfig`, `SamplerConfig`, `ThinkingConfig`,
`EmbeddingEngineConfig`, `EmbeddingOptions`, `ExperimentalFlags`) are listed with defaults and their C
counterparts in [configuration.md](../configuration.md) (layer "Swift"); the flag table above says when
each flag is read.

## Limits and failure modes

| Symptom | Cause | What to do |
| --- | --- | --- |
| The engine stalls for 10 minutes after a stream (shipped v0.18.0 wrapper before `85b1a77695`, or upstream's wrapper) | the last reference to a conversation was released in the final stream callback | use the weak-reference patch; never let a stream own its conversation |
| A reply keeps generating after its task was cancelled | the stream has no `onTermination`; only `cancel()` stops the native work | call `cancel()` (Agello checks `Task.isCancelled` per chunk) |
| The next message fails after a cancel | the session is poisoned after `CancelProcess` | create a new conversation |
| A streamed reply runs tools although `automaticToolCalling` is false | the stream path ignores the flag | don't register tools on conversations that mustn't call them |
| `getBenchmarkInfo()` throws `benchmarkNotEnabled` or `benchmarkInfoUnavailable` | the flag is false at call time (`benchmarkNotEnabled`), or was false when the engine was created, so the engine collected nothing | opt in and set it before `initialize()` |
| A flag has no effect | not opted in (the setter only logs), or set after the engine or conversation read it | opt in, then set before the reading call |
| Tool arguments fail to decode | `convertCamelToSnakeCaseInToolDescription` changed after the conversation was created | set it once, before creating conversations |
| A crash on LiteRT-LM's callback thread after a malformed tool call | two final error chunks for one send; the wrapper releases its context on each | see [Error chunks and the retained context](#error-chunks-and-the-retained-context) |
| A long embedding input throws | the overflow strategy is always *error* from Swift | chunk inputs below the longest signature |
| A cooperative thread is blocked for seconds | `sendMessage`, `initialize`, embedding calls are blocking C calls | run them off the main actor; expect one thread per call |
| `fatalError` inside a tool | a required `@ToolParam` read without a value | give parameters defaults or make them optional |

## Where in the code

| Concept | Symbol | Citation |
| --- | --- | --- |
| Engine actor | `Engine` | [`swift/Engine.swift:28-43`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Engine.swift#L28-L43) |
| Settings sequence | `Engine.initializeInternal` | [`swift/Engine.swift:89-177`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Engine.swift#L89-L177) |
| Conversation | `Conversation` | [`swift/Conversation.swift:64-98`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L64-L98) |
| Blocking send and tool loop | `Conversation.sendMessage` | [`swift/Conversation.swift:124-169`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L124-L169) |
| Stream | `Conversation.sendMessageStream`, `sendToStream` | [`swift/Conversation.swift:478-521`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L478-L521), [`swift/Conversation.swift:574-590`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L574-L590) |
| Stream bridge | `StreamContext`, `streamCallback` | [`swift/Conversation.swift:826-844`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L826-L844), [`swift/Conversation.swift:871-903`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L871-L903), [`swift/Conversation.swift:930-971`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L930-L971) |
| Tool error responses (ours) | `handleToolCalls`, `toolArguments`, `toolErrorMessage` | [`swift/Conversation.swift:393-425`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L393-L425), [`swift/Conversation.swift:427-456`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L427-L456) |
| Tools | `Tool`, `ToolParam`, `ToolManager` | [`swift/Tool.swift:211-221`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Tool.swift#L211-L221), [`swift/Tool.swift:84-101`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Tool.swift#L84-L101), [`swift/ToolManager.swift:64-118`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ToolManager.swift#L64-L118) |
| Response format | `ResponseFormat` | [`swift/ResponseFormat.swift:20-72`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ResponseFormat.swift#L20-L72) |
| Flags | `ExperimentalFlags` | [`swift/ExperimentalFlags.swift:21-46`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ExperimentalFlags.swift#L21-L46) |
| Errors | `LiteRTLMError`, `check` | [`swift/LiteRTLMError.swift:19-27`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/LiteRTLMError.swift#L19-L27), [`swift/LiteRTLMError.swift:84-91`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/LiteRTLMError.swift#L84-L91) |
| Model info | `ModelInfo` | [`swift/ModelInfo.swift:298-312`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ModelInfo.swift#L298-L312) |
| Embeddings | `EmbeddingEngine` | [`swift/EmbeddingEngine.swift:30-52`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/EmbeddingEngine.swift#L30-L52) |
| Build info, telemetry (ours) | `BuildInfo`, `Telemetry` | [`swift/BuildInfo.swift:18-30`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/BuildInfo.swift#L18-L30), [`swift/Telemetry.swift:18-62`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Telemetry.swift#L18-L62) |
| Foundation Models | `LiteRTLanguageModel` | [`swift/apple_fm/LiteRTLanguageModel.swift:51-52`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/apple_fm/LiteRTLanguageModel.swift#L51-L52) |

## Open questions

- Why `LiteRTLMFoundationModels` fails to compile against the macOS 27 SDK (the generator only
  detected that it failed).
- Whether concurrent sends on one `Conversation` are safe in the native engine was not checked; the
  wrapper does nothing to prevent them.
- **A second terminal chunk.** Traced: an unparsable tool call sends two final error chunks, and the
  wrappers over-release on the second ([Error chunks and the retained context](#error-chunks-and-the-retained-context)).
  Not reproduced; the execution manager's own error paths (a task's callback getting an error and an
  end state) were not traced chunk by chunk.
- Agello's note that drafted tokens bypass the JSON mask with MTP on is its own observation; a
  mechanism is now traced in the C++ (shared llguidance state across draft branches,
  [../troubleshooting.md](../troubleshooting.md#json-constrained-replies-break-with-mtp-on)), not
  reproduced.
