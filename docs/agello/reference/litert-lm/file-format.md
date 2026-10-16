# LiteRT-LM: the `.litertlm` file format

> **Scope:** the `.litertlm` container (magic, version, header flatbuffer, 16 KiB alignment, sections
> and their items), section data types and `model_type` strings, the metadata protos, how
> `LitertLmLoader` maps sections and what the runtime does with `backend_constraint` and
> `prefer_activation_type`, external weights, `ModelInfo`, the Python builder and peek tool, and the
> file variants Google publishes (`-gpu`, `-web`, `.task`). Evidence: the bytes of the real files in
> `generated/models/`.
> **Out of scope:** what the runtime builds from each section ([components.md](components.md),
> [executors.md](executors.md)); TFLite's own flatbuffer and its large-model modes
> ([../litert/runtime.md](../litert/runtime.md)); what each Gemma section contains as a model
> ([../models/gemma4-e4b.md](../models/gemma4-e4b.md), [../models/gemma4-e2b.md](../models/gemma4-e2b.md),
> [../models/embeddinggemma.md](../models/embeddinggemma.md)); container versions over time
> ([../versions.md](../versions.md)).
> **Pinned to:** LiteRT-LM `7b872544` (agello-next: Google's `df6df5b8` plus our commits). Shipped
> comparison: the v0.18.0 source at `b2f686e2` (wrapper `85b1a776`).
> **Applies to:** all platforms; Apple specifics in [On Apple](#on-apple). **Last verified:** 2026-10-15

## Summary

A `.litertlm` file is one container for everything a model needs: a 32-byte preamble, a FlatBuffers
header that lists typed sections by byte range, and the sections themselves, each starting on a
16 KiB boundary: TFLite graphs (one per role, named by `model_type`), external weights, a tokenizer,
and a metadata protobuf (`LlmMetadata` for chat models, `EmbeddingMetadata` for embedding models).
The runtime maps sections lazily and keys them by data type and `model_type`.

What matters for Agello:

- Every reader at both pins accepts any container version whose major number is 1; our files are
  1.5.0 (E4B, E2B), 1.6.0 (`-gpu`) and 1.7.0 (EmbeddingGemma 2).
- Two sections with the same data type and `model_type` don't give the runtime a choice: the last one
  in the header replaces the earlier ones. `backend_constraint` only validates the engine's backend,
  and `prefer_activation_type` only fills in an activation type the caller didn't set.
- The `-gpu` files hold a single `tf_lite_artisan_text_decoder` section for Google's closed GPU artisan
  executor; the open build rejects them.

## Concepts

### Layout

| Bytes | Content |
| --- | --- |
| 0–7 | magic `LITERTLM` |
| 8–19 | major, minor, patch version: three little-endian `uint32` |
| 20–23 | padding (zero) |
| 24–31 | header end offset: little-endian `uint64`, absolute |
| 32 … header end | the `LiteRTLMMetaData` flatbuffer (system metadata, section list) |
| then, each at a multiple of 16,384 | section data, in header order |

[`schema/core/litertlm_read.cc:71-139`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/schema/core/litertlm_read.cc#L71-L139) [`python/litert_lm_builder/litertlm_core.py:24-31`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm_builder/litertlm_core.py#L24-L31) [`python/litert_lm_builder/litertlm_builder.py:1438-1504`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm_builder/litertlm_builder.py#L1438-L1504) <!-- c:LM-FMT-001 -->

The flatbuffer schema (`schema/core/litertlm_header_schema.fbs`):

- `LiteRTLMMetaData { system_metadata: SystemMetadata; section_metadata: SectionMetadata }`.
- `SystemMetadata.entries` and every section's `items` are `KeyValuePair { key: string; value: VData }`,
  where `VData` is a union of `UInt8`, `Int8`, `UInt16`, `Int16`, `UInt32`, `Int32`, `Float32`, `Bool`,
  `StringValue`, `UInt64`, `Int64`, `Double`. [`schema/core/litertlm_header_schema.fbs:73-92`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/schema/core/litertlm_header_schema.fbs#L73-L92)
- `SectionObject { items; begin_offset: ulong; end_offset: ulong; data_type: AnySectionDataType }`
  covers `[begin_offset, end_offset)`. [`schema/core/litertlm_header_schema.fbs:147-169`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/schema/core/litertlm_header_schema.fbs#L147-L169)
- Alignment: the schema comment says section i+1 begins no sooner than the smallest multiple of
  `BLOCK_SIZE = 16 * 1024` greater than the previous section's end
  [`schema/core/litertlm_header_schema.fbs:147-152`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/schema/core/litertlm_header_schema.fbs#L147-L152). The Python builder places each section
  exactly at the previous end rounded up to 16,384 (the first at 32 + metadata size rounded up)
  [`python/litert_lm_builder/litertlm_builder.py:1476-1486`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm_builder/litertlm_builder.py#L1476-L1486) [`python/litert_lm_builder/litertlm_builder.py:1507-1511`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm_builder/litertlm_builder.py#L1507-L1511), and in all six files in
  `generated/models` every section begins at the previous end (or the header end) rounded up to
  16,384.

<!-- c:LM-FMT-007 -->

### Section data types

`AnySectionDataType` (a `ubyte` enum; additions bump the minor version, removals or reordering the
major one) [`schema/core/litertlm_header_schema.fbs:105-145`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/schema/core/litertlm_header_schema.fbs#L105-L145):

| Value | Name | Content | Read by |
| --- | --- | --- | --- |
| 0 | `NONE` | default | — |
| 1 | `GenericBinaryData` | arbitrary bytes, optionally named by an item `name` | `GetGenericBinaryData(name)` |
| 2 | `Deprecated` | outdated section in an edited file | — |
| 3 | `TFLiteModel` | a TFLite flatbuffer; item `model_type` says which role | `GetTFLiteModel` |
| 4 | `SP_Tokenizer` | a serialized SentencePiece `ModelProto` | `GetSentencePieceTokenizer` |
| 5 | `LlmMetadataProto` | `litert.lm.proto.LlmMetadata` | `GetLlmMetadata` |
| 6 | `HF_Tokenizer_Zlib` | a Hugging Face `tokenizer.json`, zlib-compressed | `GetHuggingFaceTokenizer` |
| 7 | `TFLiteWeights` | external weights for the `TFLiteModel` with the same `model_type` | `GetTFLiteWeights`, `GetWeightsSectionOffset` |
| 8 | `EmbeddingMetadataProto` | `litert.lm.proto.EmbeddingMetadata` | `GetEmbeddingMetadata` |
| 9 | `ExecutorMetadataProto` | `litert.lm.proto.ExecutorMetadata` | `GetExecutorMetadata` |
| 10 | `TtsMetadataProto` | TTS metadata | `GetTtsMetadata` |
| 11 | `AsrMetadataProto` | ASR metadata | `GetAsrMetadata` |
| 12 | `Text2ImageMetadataProto` | text-to-image metadata | `GetText2ImageMetadata` |

<!-- c:LM-FMT-008 --> The zlib section starts with a little-endian `uint64` holding the uncompressed
size, which the reader caps at 1 GB. [`schema/core/litertlm_read.cc:337-353`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/schema/core/litertlm_read.cc#L337-L353) [`python/litert_lm_builder/litertlm_builder.py:1335`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm_builder/litertlm_builder.py#L1335)

### `model_type` strings

A `TFLiteModel` or `TFLiteWeights` section names its role with a string item `model_type`, matched
case-insensitively. The runtime accepts a string if it parses as a value of any of the
capability-scoped proto enums (`LlmMetadata`, `EmbeddingMetadata`, `TtsMetadata`, `AsrMetadata`,
`Text2ImageMetadata` `TfLiteModelType`) or of the legacy `ModelType`; anything else makes loading
fail with `Unknown model type`. [`runtime/util/litert_lm_loader.cc:70-99`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.cc#L70-L99) [`runtime/util/litert_lm_loader.cc:144-210`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.cc#L144-L210) <!-- c:LM-FMT-010 --> The LLM strings
[`runtime/proto/llm_metadata.proto:63-94`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/proto/llm_metadata.proto#L63-L94):

| String | Role | In our Gemma 4 E4B file |
| --- | --- | --- |
| `tf_lite_prefill_decode` | main graph: prefill, decode, verify signatures | section 10, 2,261.0 MB |
| `tf_lite_embedder` | token embeddings (CPU) | section 2, 170.9 MB |
| `tf_lite_per_layer_embedder` | per-layer embeddings (CPU) | section 3, 836.8 MB |
| `tf_lite_aux` | auxiliary graph (AOT NPU files) | — |
| `tf_lite_audio_encoder_hw` | audio encoder | section 4, 94.1 MB |
| `tf_lite_end_of_audio` | end-of-audio embedding | section 6 |
| `tf_lite_vision_adapter` | vision adapter | section 8, 7.9 MB |
| `tf_lite_vision_encoder` | vision encoder | section 7, 224.1 MB |
| `tf_lite_audio_frontend` | weightless STFT front end | — |
| `tf_lite_audio_adapter` | audio adapter | section 5, 15.7 MB |
| `tf_lite_artisan_text_decoder` | text decoder for the closed GPU artisan executor | — (`-gpu` files only) |
| `tf_lite_end_of_vision` | end-of-image embedding | section 9 |
| `tf_lite_mtp_drafter` | MTP drafter | section 11, 45.3 MB |
| `tf_lite_mtp_aux` | MTP auxiliary graph | — |

EmbeddingGemma uses the `EmbeddingMetadata` enum, which adds `tf_lite_text_encoder`. [`runtime/proto/embedding_metadata.proto:81-129`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/proto/embedding_metadata.proto#L81-L129)
The sizes are from [`generated/models/e4b-ours.md`](../generated/models/e4b-ours.md).

### Metadata protos

The field-by-field tables are generated in [`generated/proto-fields.md`](../generated/proto-fields.md);
what matters here:

- **`LlmMetadata`** (`runtime/proto/llm_metadata.proto`): `start_token`, `stop_tokens`,
  `prompt_templates`, `sampler_params`, `max_num_tokens`, `llm_model_type` (a oneof per model family,
  `gemma4` for ours), `jinja_prompt_template`, `channels`, `suppress_tokens`, `kv_cache_init_value`,
  `supports_thinking`, `supports_function_calling`, `pad_token`, `min_runtime_version`.
  [`runtime/proto/llm_metadata.proto:62-156`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/proto/llm_metadata.proto#L62-L156) `min_runtime_version` only produces a warning when the runtime is older.
  [`runtime/engine/engine_settings.cc:280-292`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L280-L292)
- **`ExecutorMetadata`** (`executor_metadata.proto`): `llm_executor_metadata` with
  `max_history_size`, `state_buffers` (KV caches by type, sequence axis, minimum and maximum length)
  and `attention_mask_settings`. [`runtime/proto/executor_metadata.proto:110-127`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/proto/executor_metadata.proto#L110-L127) It is optional: the executor uses it when present.
  [`runtime/executor/llm_litert_compiled_model_executor.cc:315-318`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_litert_compiled_model_executor.cc#L315-L318)
- **`EmbeddingMetadata`** (`embedding_metadata.proto`): `embedding_model_type`, `bos_token`,
  `eos_token`, `audio_preprocessor`, `min_input_length`, `max_input_length`, `min_runtime_version`.
  [`runtime/proto/embedding_metadata.proto:81-129`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/proto/embedding_metadata.proto#L81-L129)

What the real files carry (from their bytes):

| File | Container | Header end | Sections | Metadata highlights |
| --- | --- | --- | --- | --- |
| `gemma-4-E4B-it-fp16-rope.litertlm` (ours, Agello downloads it) | 1.5.0 | 1,872 | 12 | `LlmMetadata`: start 2, stops 1/50/106, `gemma4` type (image and audio delimiters, tool-call fences, patch 16 × 16, at most 2,520 patches), Jinja template, one channel `thought`; no `sampler_params`, no `max_num_tokens` |
| `gemma-4-E4B-it.litertlm` (Google) | 1.5.0 | 1,872 | 12 | same metadata |
| `gemma-4-E2B-it.litertlm` (Google) | 1.5.0 | 1,872 | 12 | same shape |
| `gemma-4-E4B-it-gpu.litertlm`, `gemma-4-E2B-it-gpu.litertlm` | 1.6.0 | 704 | 3 | adds `sampler_params` `TOP_P`, k 64, p 0.95, temperature 1 |
| `embeddinggemma-2-text-270m.litertlm` | 1.7.0 | 880 | 6 | `EmbeddingMetadata`: `embedding_gemma_v2`, BOS 2, EOS 1, miniaudio config, `max_input_length` 1024 |

<!-- c:LM-FMT-004 --> <!-- c:LM-FMT-026 --> Sources:
[`e4b-ours.md`](../generated/models/e4b-ours.md), [`e4b-google.md`](../generated/models/e4b-google.md),
[`e2b-google.md`](../generated/models/e2b-google.md), [`e4b-gpu.md`](../generated/models/e4b-gpu.md),
[`e2b-gpu.md`](../generated/models/e2b-gpu.md), [`embeddinggemma.md`](../generated/models/embeddinggemma.md).
None of them has an `ExecutorMetadataProto` section.

## How it works

### Reading the header

`ReadHeaderFromLiteRTLM` checks the magic, reads the three version numbers, rejects a major version
other than `LITERTLM_MAJOR_VERSION` (1) with `Unimplemented`, skips the padding, reads the header end
offset and copies the flatbuffer that follows. The minor and patch numbers are read and logged, never
compared. [`schema/core/litertlm_read.cc:71-139`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/schema/core/litertlm_read.cc#L71-L139) [`schema/core/litertlm_read.cc:96-101`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/schema/core/litertlm_read.cc#L96-L101) The v0.18.0 reader has the same check. [`schema/core/litertlm_read.cc:96-101`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/schema/core/litertlm_read.cc#L96-L101)
<!-- c:LM-FMT-002 --> The two builds' own version constants differ (1.9.0 in agello-next, 1.7.0 in
v0.18.0), and the Python builder writes 1.9.0. [`schema/core/litertlm_header.h:31-38`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/schema/core/litertlm_header.h#L31-L38) [`schema/core/litertlm_header.h:36-38`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/schema/core/litertlm_header.h#L36-L38) [`python/litert_lm_builder/litertlm_core.py:24-31`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm_builder/litertlm_core.py#L24-L31)
<!-- c:LM-FMT-003 -->

So, to the question "which container versions does the reader accept": any 1.x.y, at both pins. A
newer minor version can still fail later, at the section level: an unknown `model_type` string is an
error (above), and a data type value the reader doesn't know is stored under its number and never
read.

`LitertLmLoader` maps at most the first 16 KiB of the file to read the header
(`kLitertLmHeaderMaxSize`), so a header that ends after byte 16,384 fails with "Failed to read header
data" through this loader. Our files' headers end at byte 704 to 1,872, and their first section starts
at 16,384. [`runtime/util/litert_lm_loader.h:44`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.h#L44) [`runtime/util/litert_lm_loader.cc:322-327`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.cc#L322-L327) <!-- c:LM-FMT-005 -->

### Building the section index

`LitertLmLoader::Initialize` walks the section list once [`runtime/util/litert_lm_loader.cc:304-377`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.cc#L304-L377):

1. For each section, `ExtractBufferKeyAndTfLiteSectionHint` derives a `BufferKey`:
   - `TFLiteModel`, `TFLiteWeights`, `SP_Tokenizer`, `HF_Tokenizer_Zlib` with an items list: the
     data type plus the lowercased `model_type` item (`InvalidArgument` for an unknown model type).
     A `TFLiteModel` or `TFLiteWeights` section whose items lack `model_type` is keyed as
     `tf_lite_prefill_decode` (with a warning), for old files; one with no items at all is keyed by
     data type alone.
     [`runtime/util/litert_lm_loader.cc:189-194`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.cc#L189-L194)
   - `GenericBinaryData`: the data type plus the `name` item.
   - Everything else: the data type alone.
   It also collects the items `backend_constraint` and `prefer_activation_type` (those keys matched
   case-insensitively) into a `TfLiteSectionHint`. [`runtime/util/litert_lm_loader.cc:143-210`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.cc#L143-L210) [`runtime/util/litert_lm_loader.h:49-65`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.h#L49-L65) [`runtime/util/litert_lm_loader.h:70-73`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.h#L70-L73)
2. It stores the hint and the `[begin, end)` range in two `unordered_map`s keyed by `BufferKey`,
   by assignment. [`runtime/util/litert_lm_loader.cc:350-368`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.cc#L350-L368) [`runtime/util/litert_lm_loader.h:236-259`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.h#L236-L259)

Nothing is mapped yet. Items other than `model_type`, `backend_constraint`, `prefer_activation_type`
and `name` (for example the `-gpu` files' `gpu_artisan_weights_version`) are ignored by this loader.

### Duplicate sections

Because step 2 assigns into a map keyed by `BufferKey`, a later section with the same data type and
`model_type` replaces the range and the hint of an earlier one: **the last such section in header
order wins**, whatever its `backend_constraint` or `prefer_activation_type`. The runtime never picks
among duplicates by backend or activation type. [`runtime/util/litert_lm_loader.cc:350-368`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.cc#L350-L368) <!-- c:LM-FMT-011 --> `ModelInfo`
behaves the same way for the main LLM graph (each `kMainLlm` section overwrites the previous one).
[`schema/model_info/model_info.cc:680-685`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/schema/model_info/model_info.cc#L680-L685) None of our files has two sections with the same key (checked from the header bytes of
all six files). The sections that look alike in EmbeddingGemma differ by data type: a `TFLiteModel`
and a `TFLiteWeights` section per role.

### What the hints do

The hints are read back per role (`GetTFLiteModelBackendConstraint`,
`GetTFLiteModelPreferActivationType`) [`runtime/util/litert_lm_loader.cc:516-523`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.cc#L516-L523) and, in the LLM engine, read only by `EngineSettings::MaybeUpdateAndValidate` (called twice for files
with an `LlmModelType`: before executor creation and again after the tokenizer loads), for three roles
only: the main graph (`tf_lite_prefill_decode`), the vision encoder (`tf_lite_vision_encoder`) and the
audio encoder (`tf_lite_audio_encoder_hw`). The embedding engine
does the same for its text encoder, vision encoder and audio encoder
([`runtime/core/embedding_engine_impl.cc:319-331`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/embedding_engine_impl.cc#L319-L331)).
[`runtime/core/engine_advanced_impl.cc:515-564`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/engine_advanced_impl.cc#L515-L564) [`runtime/core/engine_advanced_impl.cc:636-650`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/engine_advanced_impl.cc#L636-L650) [`runtime/engine/engine_settings.cc:530-548`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L530-L548) <!-- c:LM-FMT-012 -->

- **`backend_constraint`** is split on commas and compared case-insensitively with the executor's
  backend name (`CPU`, `GPU`, `NPU`, `GPU_ARTISAN`, ...). No match → `InvalidArgument("Main backend
  constraint mismatch. Model requires one of [...]")`; no constraint → anything goes.
  [`runtime/engine/engine_settings.cc:81-111`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L81-L111) [`runtime/executor/executor_settings_base.cc:40-57`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/executor_settings_base.cc#L40-L57)
- **`prefer_activation_type`** sets the executor's activation data type only when the caller set
  none: `fp32`/`float32` → FP32, `fp32_fp16` → FP32 with mixed precision, `fp16`/`float16` → FP16,
  `int16`, `int8`; anything else is an error. [`runtime/engine/engine_settings.cc:126-145`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L126-L145) [`runtime/executor/executor_settings_base.cc:112-131`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/executor_settings_base.cc#L112-L131) Without the item,
  the loader logs that the system default applies (FP16 for the text decoder, FP32 for the vision and
  audio encoders). [`runtime/util/litert_lm_loader.cc:541-544`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.cc#L541-L544) <!-- c:LM-FMT-013 -->

In the Gemma 4 E2B and E4B files, the audio encoder, the audio adapter and the vision adapter carry
`backend_constraint = cpu`, and the vision encoder and the main graph carry
`prefer_activation_type = fp16`; the embedders, end-of-media graphs and MTP drafter carry neither.
<!-- c:LM-FMT-015 --> So asking for a GPU audio encoder fails validation, while the adapters'
constraints aren't checked at all: `EngineSettings::CreateDefault` puts both adapters on the CPU
regardless. [`runtime/engine/engine_settings.cc:244-245`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L244-L245)

The validation splits on `,` without trimming spaces, while the Python builder accepts and stores a
constraint such as `'cpu, gpu'` (its own example), validating it with `strip()` but writing the string
as given (lowercased). A section written that way would reject a GPU engine, because `" gpu"` doesn't
equal `GPU`. Not observed in any file we have; found by reading the code. [`runtime/engine/engine_settings.cc:81-111`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L81-L111)
[`python/litert_lm_builder/litertlm_builder.py:1545-1555`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm_builder/litertlm_builder.py#L1545-L1555) [`python/litert_lm_builder/litertlm_builder.py:1124-1132`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm_builder/litertlm_builder.py#L1124-L1132) [`python/litert_lm_builder/litertlm_builder.py:1383`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm_builder/litertlm_builder.py#L1383) <!-- c:LM-FMT-014 -->

### Mapping a section

`GetSectionBuffer(key)` maps a section on first use, under a mutex, and caches the `BufferRef`.
[`runtime/util/litert_lm_loader.cc:379-412`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.cc#L379-L412) With a file descriptor as the source (how the C API opens a model path), `MapSection`
memory-maps from the section's begin offset rounded down to the page size (`getpagesize()`) and
points the buffer past the gap; with an already mapped file it uses pointer arithmetic.
[`runtime/util/litert_lm_loader.cc:213-254`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.cc#L213-L254) [`support/util/memory_mapped_file_posix.cc:91`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/support/util/memory_mapped_file_posix.cc#L91) <!-- c:LM-FMT-019 --> The POSIX mapping is `PROT_READ | PROT_WRITE`,
`MAP_PRIVATE` (copy on write: writes such as magic-number replacement stay private to the process),
followed by `madvise(MADV_DONTNEED)` on Apple and `MADV_WILLNEED` elsewhere. [`support/util/memory_mapped_file_posix.cc:133-145`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/support/util/memory_mapped_file_posix.cc#L133-L145)
<!-- c:LM-FMT-020 -->

How a model path gets here: `litert_lm_engine_settings_create` builds a path-only `ModelAssets`
[`c/engine.cc:646`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/c/engine.cc#L646) [`runtime/executor/executor_settings_base.cc:159-161`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/executor_settings_base.cc#L159-L161); the file is opened as a `ScopedFile` later, by
`GetOrCreateScopedFile` [`runtime/executor/executor_settings_base.cc:221-227`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/executor_settings_base.cc#L221-L227). `GetFileFormat` trusts a `.tflite`, `.task` or
`.litertlm` extension, and checks the first 8 bytes (`TFL3`, `PK`, or a `LITERTLM` prefix) only when
the assets hold a memory-mapped file or an already-open `ScopedFile`; so a path from the C API with
another extension fails with "Unsupported or unknown file format."
[`runtime/util/file_format_util.cc:63-102`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/file_format_util.cc#L63-L102) [`runtime/util/file_format_util.cc:40-51`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/file_format_util.cc#L40-L51). `BuildLiteRtCompiledModelResources` accepts only the
TASK and LITERT_LM formats. [`runtime/executor/litert_compiled_model_executor_utils.cc:920-933`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/litert_compiled_model_executor_utils.cc#L920-L933) [`runtime/executor/litert_compiled_model_executor_utils.cc:131-164`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/litert_compiled_model_executor_utils.cc#L131-L164) A stream-backed `ModelAssets`
goes through the streaming loader instead, which reads the same header and hands out sections in file
order without memory-mapping. [`runtime/core/engine_advanced_impl.cc:405-410`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/engine_advanced_impl.cc#L405-L410) [`runtime/util/litert_lm_streaming_loader.h:70-74`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_streaming_loader.h#L70-L74) [`runtime/util/litert_lm_streaming_loader.h:47-56`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_streaming_loader.h#L47-L56)

The loader's `GetLlmMetadata()` calls `.value()` on the optional section buffer without checking it
[`runtime/util/litert_lm_loader.h:152-156`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.h#L152-L156), and `EngineAdvancedImpl::Create` calls `model_resources->GetLlmMetadata()` for
every model not backed by a data stream (data-stream assets go to `CreateStreamingWeights` first)
[`runtime/core/engine_advanced_impl.cc:398-410`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/engine_advanced_impl.cc#L398-L410) [`runtime/core/engine_advanced_impl.cc:450-451`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/engine_advanced_impl.cc#L450-L451) [`runtime/components/model_resources_litert_lm.cc:228-239`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/model_resources_litert_lm.cc#L228-L239). So an
LLM engine created from a mapped `.litertlm` file without an `LlmMetadataProto` section (an embedding
model, for instance) fails on that optional access instead of returning an error status. Agello opens
EmbeddingGemma through the embedding engine, which doesn't take this path.

### External weights

A TFLite graph's constant data can live in three places, and our files use all three
([`generated/models/`](../generated/models/e4b-ours.md)):

| Placement | How it's found | Our files |
| --- | --- | --- |
| Inline in the flatbuffer | TFLite `Buffer.data` | every E2B section; E4B's embedder, per-layer embedder (42 INT2 tables, 704,643,072 bytes, plus per-row scales and zero points <!-- c:MOD-E4B-010 --> <!-- c:MOD-E4B-011 -->), encoders, adapters and drafter |
| After the graph, in the same section | `Buffer.offset` / `Buffer.size` (LiteRT's large-model mode, [../litert/runtime.md](../litert/runtime.md)) | E4B `tf_lite_prefill_decode`: 63.1 MB of graph, 2,197.9 MB of weights after it |
| A `TFLiteWeights` section | tensors' external buffer IDs, resolved against the section | EmbeddingGemma 2: 67.1 MB for `tf_lite_embedder`, 68.4 MB for `tf_lite_text_encoder` |

<!-- c:LM-FMT-030 --> For a `TFLiteWeights` section the runtime passes LiteRT a duplicated file
descriptor plus one region named `tflite_weights` (offset, length) through
`SetExternalWeightScopedFile`; no such section → no external weights. [`runtime/executor/litert_compiled_model_executor_utils.cc:1200-1219`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/litert_compiled_model_executor_utils.cc#L1200-L1219)
[`runtime/components/model_resources_litert_lm.cc:348-351`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/model_resources_litert_lm.cc#L348-L351) The LLM executor does this for the main graph [`runtime/executor/llm_litert_compiled_model_executor.cc:2101-2102`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_litert_compiled_model_executor.cc#L2101-L2102); the embedding
lookups, the audio executor, the MTP drafter and the embedding executor do it for theirs. On
WebAssembly only, a GPU engine whose model has external weights is switched to the streaming path.
[`runtime/core/engine_advanced_impl.cc:209-230`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/engine_advanced_impl.cc#L209-L230)

### The `-gpu`, `-web` and `.task` variants

Google's Hugging Face repositories for Gemma 4 E2B and E4B list, besides the `.litertlm` file Agello
starts from, `-gpu.litertlm`, `-web.litertlm` and `-web.task` files (and NPU-specific E2B files).
<!-- c:LM-FMT-024 -->

- **`-gpu`**: container 1.6.0 with three sections: `LlmMetadataProto`, `SP_Tokenizer`, and one
  `TFLiteModel` with `model_type = tf_lite_artisan_text_decoder`, `backend_constraint = gpu_artisan`
  and `gpu_artisan_weights_version = 0.0` (2,964.3 MB for E4B, 2,003.7 MB for E2B); no embedders,
  encoders or drafter sections. <!-- c:LM-FMT-018 --> When `EngineSettings::CreateDefault` sees a
  `tf_lite_artisan_text_decoder` section, it switches a GPU engine to `GPU_ARTISAN` (and a GPU audio
  backend too) [`runtime/engine/engine_settings.cc:217-223`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L217-L223) [`runtime/engine/engine_settings.cc:225-232`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L225-L232); the open executor factory then returns
  `Unsupported backend` for anything but CPU, GPU and NPU. [`runtime/executor/llm_litert_compiled_model_executor_factory.cc:315-331`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/llm_litert_compiled_model_executor_factory.cc#L315-L331) So the documented build
  can't run a `-gpu` file; Google's v0.18.0 binary contains a closed GPU artisan executor
  ([`generated/binaries.md`](../generated/binaries.md): 166 defined symbols against our 29).
  <!-- c:LM-FMT-017 -->
- **`-web.litertlm`**: the same byte size as `-gpu` in both repositories; not inspected here.
- **`.task`**: a ZIP archive (`PK` magic) of named files;
  `BuildModelResourcesFromTaskFormat` requires a `TF_LITE_PREFILL_DECODE` entry. [`runtime/executor/litert_compiled_model_executor_utils.cc:109-127`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/litert_compiled_model_executor_utils.cc#L109-L127)

### Our E4B file against Google's

`gemma-4-E4B-it-fp16-rope.litertlm` is Google's file with the rotary-position computation in the main
graph rewritten into 512-token blocks ([../models/gemma4-e4b.md](../models/gemma4-e4b.md)). The
container is unchanged in version (1.5.0), header end (1,872), system metadata, and the order and
start offsets of sections 0–10; section 10 (`tf_lite_prefill_decode`) grows by 983,040 bytes and
section 11 (`tf_lite_mtp_drafter`) by 212,992 bytes, which moves section 11's start. <!-- c:LM-FMT-040 -->
Both E4B files are padded with zeros to a multiple of 16 KiB after their last section (11,760 bytes
for ours); the `-gpu` files and EmbeddingGemma end exactly at their last section.

### Writing files: the builder

`python/litert_lm_builder` (`litert-lm-builder` on PyPI, documented in Google's
`docs/litert_lm_builder.md`) builds, unpacks and inspects files. `_build_sections`
[`python/litert_lm_builder/litertlm_builder.py:1438-1504`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm_builder/litertlm_builder.py#L1438-L1504):

1. Fill in system metadata (a UUID if missing, a fresh `creation_timestamp`).
2. Pack the metadata with placeholder offsets to learn its size.
3. Write each section starting at `round_up(32 + metadata size)`, then at the next 16 KiB boundary
   after each section ends. [`python/litert_lm_builder/litertlm_builder.py:1507-1511`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm_builder/litertlm_builder.py#L1507-L1511)
4. Seek to 0 and write the magic, version 1.9.0, padding, the header end offset and the final
   metadata (same size as the placeholder pass).

<!-- c:LM-FMT-034 --> `add_tflite_model` stores `backend_constraint` (validated against `cpu`, `gpu`,
`npu`, `gpu_artisan`) and `prefer_activation_type` as lowercase string items; `add_tflite_weights`
adds a `TFLiteWeights` section for a model type; `add_hf_tokenizer` compresses a JSON tokenizer with
zlib. [`python/litert_lm_builder/litertlm_builder.py:266-273`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm_builder/litertlm_builder.py#L266-L273) [`python/litert_lm_builder/litertlm_builder.py:1124-1132`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm_builder/litertlm_builder.py#L1124-L1132) [`python/litert_lm_builder/litertlm_builder.py:1163-1166`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm_builder/litertlm_builder.py#L1163-L1166) [`python/litert_lm_builder/litertlm_builder.py:1335`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm_builder/litertlm_builder.py#L1335) The peek tool prints the header,
system metadata and every section (decoding the metadata protos as text) and can dump each section to
a file: [`python/litert_lm_builder/litertlm_peek_main.py:20-21`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm_builder/litertlm_peek_main.py#L20-L21)

```sh
bazel run //python/litert_lm_builder:litertlm_peek_main -- --litertlm_file=<file.litertlm>
# or, from the PyPI package:
litert-lm-peek --litertlm_file demo.litertlm --dump_files_dir ./extracted_files
```

`generated/models/*.md` are produced by our own reader (`tools/agello-docs/gen_model_tables.py`),
which reads the same header layout.

### `ModelInfo`

`schema/model_info` reads a file's header and metadata without loading a model and reports, for LLMs,
input and output modalities, thinking and tool-calling support, speculative decoding (a drafter
section), vision token budgets, default sampler parameters, supported backends per modality (from
`backend_constraint`), the minimum runtime version and the maximum context, and for embedding models
the embedding size and input lengths. [`schema/model_info/model_info.h:110-227`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/schema/model_info/model_info.h#L110-L227) `GetModelInfo` classifies a file as an embedding
model when it has `EmbeddingMetadata`, or a text encoder and neither LLM metadata nor a main graph;
it fills the LLM capability when `LlmMetadata` exists or the file isn't an embedding model.
[`schema/model_info/model_info.cc:1301-1361`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/schema/model_info/model_info.cc#L1301-L1361) <!-- c:LM-FMT-037 --> The C and Swift wrappers around it are on
[c-api.md](c-api.md) and [swift-api.md](swift-api.md).

## On Apple

- The Swift wrapper passes a file path to `litert_lm_engine_settings_create` [`swift/Engine.swift:81-84`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Engine.swift#L81-L84); the runtime later opens it as a `ScopedFile` (`GetOrCreateScopedFile`), and the loader maps each section the first time it is
  needed with `mmap(MAP_PRIVATE)` and `madvise(MADV_DONTNEED)` (`#ifdef __APPLE__`), so reading the
  header and creating the engine don't page in sections that are never used (on E4B, the audio
  sections when no audio arrives). [`support/util/memory_mapped_file_posix.cc:133-145`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/support/util/memory_mapped_file_posix.cc#L133-L145)
- Whenever the page size divides 16,384, every section's begin offset is page-aligned and
  `MapSection`'s alignment gap is zero; the code doesn't rely on it and rounds down to `getpagesize()`
  on every platform. [`runtime/util/litert_lm_loader.cc:213-254`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.cc#L213-L254) [`support/util/memory_mapped_file_posix.cc:91`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/support/util/memory_mapped_file_posix.cc#L91)
- Models are created from these mapped buffers, not from file descriptors, on Apple's GPU and CPU
  backends ([components.md](components.md#model-resources)).

## Settings

| Setting | Layer | Default | Effect |
| --- | --- | --- | --- |
| `backend_constraint` item | file (section item) | none | engine creation fails if the main, vision-encoder or audio-encoder backend isn't listed |
| `prefer_activation_type` item | file (section item) | none | activation type when the caller set none |
| `LlmMetadata.sampler_params` | file | absent: `TOP_P`, k 1, p 0.95, temperature 1 (CPU/GPU) | session default sampler ([components.md](components.md#samplers)) |
| `LlmMetadata.max_num_tokens` | file | absent in our files | default allocated context when the caller passes 0 ([engine-and-sessions.md](engine-and-sessions.md)) |
| `min_runtime_version` | file | empty | warning only |
| `enable_skia_image_preprocessor`, tokenizer flags | Bazel | off / on | see [components.md](components.md#settings) |

## Limits and failure modes

- **Header larger than 16 KiB:** unreadable by `LitertLmLoader` ([Reading the header](#reading-the-header)).
- **Major version other than 1:** `Unimplemented` ("This reader doesn't support version N").
- **Unknown `model_type`:** the whole file fails to load, so a file built for a newer runtime with a
  new role string can't be opened by an older one even if the role is never used.
- **Duplicate keys:** silently resolved to the last section ([Duplicate sections](#duplicate-sections)).
- **Spaces in `backend_constraint`:** see [What the hints do](#what-the-hints-do).
- **`-gpu` files in the open build:** `Unsupported backend: GPU_ARTISAN`.
- **No `LlmMetadataProto`:** an LLM engine created from a mapped file fails on an unchecked optional ([Mapping a section](#mapping-a-section)).

## Where in the code

| Concept | Symbol | Citation |
| --- | --- | --- |
| Header schema | `LiteRTLMMetaData`, `SectionObject`, `AnySectionDataType` | [`schema/core/litertlm_header_schema.fbs:147-169`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/schema/core/litertlm_header_schema.fbs#L147-L169) |
| Version constants | `LITERTLM_MAJOR_VERSION` … | [`schema/core/litertlm_header.h:31-38`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/schema/core/litertlm_header.h#L31-L38) |
| Header reader | `ReadHeaderFromLiteRTLM` | [`schema/core/litertlm_read.cc:71-139`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/schema/core/litertlm_read.cc#L71-L139) |
| Section keys and hints | `ExtractBufferKeyAndTfLiteSectionHint` | [`runtime/util/litert_lm_loader.cc:144-210`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.cc#L144-L210) |
| Section index | `LitertLmLoader::Initialize` | [`runtime/util/litert_lm_loader.cc:304-377`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.cc#L304-L377) |
| Lazy mapping | `LitertLmLoader::MapSection`, `GetSectionBuffer` | [`runtime/util/litert_lm_loader.cc:213-254`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.cc#L213-L254) |
| Backend constraint check | `ValidateBackendConstraint` | [`runtime/engine/engine_settings.cc:81-111`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L81-L111) |
| Activation hint | `MaybeOverrideActivationType` | [`runtime/engine/engine_settings.cc:126-145`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L126-L145) |
| Artisan detection | `EngineSettings::CreateDefault` | [`runtime/engine/engine_settings.cc:225-232`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L225-L232) |
| External weights | `GetExternalWeightResources` | [`runtime/executor/litert_compiled_model_executor_utils.cc:1200-1219`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/litert_compiled_model_executor_utils.cc#L1200-L1219) |
| Format detection | `GetFileFormatFromFileContents` | [`runtime/util/file_format_util.cc:40-51`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/file_format_util.cc#L40-L51) |
| Builder | `LitertLmFileBuilder._build_sections` | [`python/litert_lm_builder/litertlm_builder.py:1438-1504`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm_builder/litertlm_builder.py#L1438-L1504) |
| Model info | `GetModelInfo` | [`schema/model_info/model_info.cc:1301-1361`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/schema/model_info/model_info.cc#L1301-L1361) |
| Streaming loader | `LitertLmStreamingLoader` | [`runtime/util/litert_lm_streaming_loader.h:70-74`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_streaming_loader.h#L70-L74) |

## Open questions

- Not inspected: the `-web.litertlm` and `-web.task` files (only their sizes are known; reading their
  headers would need a range request we didn't make).
- Not verified: what Google's closed GPU artisan executor does with `gpu_artisan_weights_version`.
- Not verified at run time: the duplicate-section and `backend_constraint` spacing behaviors are read
  from the code; no file we have exercises them.
- `gen_model_tables.py` reads at most 256 MiB of a section, so the generated tables don't list the
  E4B per-layer embedder (836.8 MB); its contents above come from the `inspect_e4b.py` helpers (see
  [../tools.md](../tools.md)).
