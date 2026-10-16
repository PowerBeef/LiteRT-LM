# Gemma 4: tokenizer and chat format

> **Scope:** the Gemma 4 tokenizer (the SentencePiece model inside the `.litertlm` files and the
> Hugging Face tokenizer), its special tokens and IDs, the chat format as trained (turns, system,
> thinking, tool declarations, calls and responses, image and audio markers), stop tokens, the chat
> templates in circulation and how they differ, and an exact token sequence for a short chat.
> **Out of scope:** how `Conversation` renders, diffs and prefills a template, and how reasoning is
> removed from the KV cache ([../litert-lm/conversation.md](../litert-lm/conversation.md)); the tokenizer
> classes, streaming detokenizer and stop-token detector ([../litert-lm/components.md](../litert-lm/components.md));
> the container sections ([../litert-lm/file-format.md](../litert-lm/file-format.md)); the models
> themselves ([gemma4-family.md](gemma4-family.md)).
> **Pinned to:** LiteRT-LM `7b872544` (agello-next); model files and Hugging Face revisions in
> `tools/agello-docs/pins.json`.
> **Applies to:** all platforms. **Last verified:** 2026-10-15

## Summary

Every Gemma 4 model and EmbeddingGemma 2 use one 262,144-entry SentencePiece BPE vocabulary with byte
fallback. The `.litertlm` files carry it as an `SP_Tokenizer` section (4,689,013 bytes, identical in
our E4B file, Google's E4B file and the EmbeddingGemma file); LiteRT-LM encodes with it directly.
The chat format is built from control tokens: `<|turn>` (105) / `<turn|>` (106) around each turn,
`<|think|>` (98) to turn thinking on, `<|channel>` (100) / `<channel|>` (101) around the reasoning,
and six tool tokens (46 to 51) plus `<|"|>` (52) for strings. Generation stops on 1 (`<eos>`), 106
(`<turn|>`) or 50 (`<|tool_response>`). Four Jinja templates are in circulation: the one inside the
published E2B/E4B files (the one Agello runs), a variant in the `-gpu` files, the newer canonical one
in LiteRT-LM's repository, and Google's on Hugging Face. They agree on plain chats and differ around
tool calls: the file template closes the model's turn after a tool call and reopens it after the
tool response, which neither Google's template nor the paper's examples do.

## Concepts

| Term | Meaning |
| --- | --- |
| **Piece** | a vocabulary entry of the SentencePiece model; its index is the token ID |
| **Control token** | SentencePiece type `CONTROL` (`<pad>`, `<eos>`, `<bos>`): never produced by encoding text |
| **User-defined symbol** | type `USER_DEFINED`: matched as a whole wherever its text appears in the input; all chat-format markers are of this type |
| **Byte token** | `<0x00>` to `<0xFF>`: used for characters with no piece (byte fallback) |
| **Template** | the Jinja program that turns a message list into prompt text (`jinja_prompt_template` in a file's metadata, `chat_template.jinja` on Hugging Face) |
| **Placeholder** | `<|image|>` / `<|audio|>` in rendered text: where LiteRT-LM inserts media embeddings |
| **Stop token** | a token ID that ends decoding; from the file's `stop_tokens` unless the caller sets its own |

## How it works

### The tokenizer in the `.litertlm` files

Each Gemma 4 file in `generated/models` has one `SP_Tokenizer` section (section 1) and no
`HF_Tokenizer_Zlib` section (claim LM-CMP-005). In all of them it spans bytes 32,768 to 4,721,781
(4,689,013 bytes). <!-- c:MOD-TOK-001 --> Reading the section of Agello's E4B file, Google's E4B file
and the EmbeddingGemma 2 file gives the same bytes (SHA-256 `e594c8a9…ef03`); E2B and the `-gpu` files
were only read remotely and match in size. <!-- c:MOD-TOK-001 -->

Parsed as a SentencePiece `ModelProto`, the section holds:

| Field | Value |
| --- | --- |
| Pieces | 262,144: 255,474 `NORMAL`, 6,410 `USER_DEFINED`, 256 `BYTE`, 3 `CONTROL`, 1 `UNKNOWN` |
| `trainer_spec.model_type` | `BPE` |
| `trainer_spec.vocab_size` | 262,144 |
| `byte_fallback` | true |
| `split_digits` | true |
| `split_by_whitespace` | false |
| `allow_whitespace_only_pieces` | true |
| `unk_id` / `bos_id` / `eos_id` / `pad_id` | 3 / 2 / 1 / 0 |
| `bos_piece` / `eos_piece` | `<bos>` / `<eos>` |
| `normalizer_spec.name` | `identity` |
| `add_dummy_prefix` / `remove_extra_whitespaces` | false / false |

<!-- c:MOD-TOK-002 --> <!-- c:MOD-TOK-003 -->

So text is not normalized, no space is added at the start, runs of spaces are kept, digits are split
one per token, and a character without a piece becomes its UTF-8 bytes. The report describes the
same tokenizer: SentencePiece "with split digits, preserved whitespace, and byte-level encodings",
262k entries, shared with Gemini. <!-- c:MOD-TOK-004 -->

### Vocabulary layout and special tokens

| ID | Piece | Type | Role |
| --- | --- | --- | --- |
| 0 | `<pad>` | control | padding |
| 1 | `<eos>` | control | end of sequence (base models end with it) |
| 2 | `<bos>` | control | start of sequence |
| 3 | `<unk>` | unknown | |
| 4 | `<mask>` | user-defined | |
| 5 | `[multimodal]` | user-defined | |
| 6–45, 53–97, 99, 102–104, … | `<unused0>` … `<unused6226>` | user-defined | 6,227 reserved IDs in total |
| 46 / 47 | `<\|tool>` / `<tool\|>` | user-defined | around a tool declaration |
| 48 / 49 | `<\|tool_call>` / `<tool_call\|>` | user-defined | around a tool call |
| 50 / 51 | `<\|tool_response>` / `<tool_response\|>` | user-defined | around a tool response |
| 52 | `<\|"\|>` | user-defined | string delimiter inside tool syntax |
| 98 | `<\|think\|>` | user-defined | turns thinking on (in the system turn) |
| 100 / 101 | `<\|channel>` / `<channel\|>` | user-defined | around a channel (`thought`) |
| 105 / 106 | `<\|turn>` / `<turn\|>` | user-defined | start / end of a turn |
| 107–137 | `\n` … 31 × `\n` | user-defined | newline runs |
| 138–… | `▁▁` … | user-defined | runs of spaces (`▁` is a space) |
| 238–493 | `<0x00>` … `<0xFF>` | byte | byte fallback |
| 255968–255998 | `\t` … 31 × `\t` | user-defined | tab runs |
| 255999 | `<\|image>` | user-defined | start of image embeddings |
| 256000 | `<\|audio>` | user-defined | start of audio embeddings |
| 258880 | `<\|image\|>` | user-defined | image placeholder |
| 258881 | `<\|audio\|>` | user-defined | audio placeholder |
| 258882 | `<image\|>` | user-defined | end of image embeddings |
| 258883 | `<audio\|>` | user-defined | end of audio embeddings |
| 258884 | `<\|video\|>` | user-defined | video placeholder |

<!-- c:MOD-TOK-005 --> <!-- c:MOD-TOK-006 --> <!-- c:MOD-TOK-007 --> <!-- c:MOD-TOK-008 -->
<!-- c:MOD-TOK-009 -->

The Hugging Face configs agree wherever they give an ID: `text_config.bos_token_id` 2,
`text_config.eos_token_id` 1 (the top-level `eos_token_id` is `[1, 106]`), `text_config.pad_token_id` 0,
`boi_token_id` 255999, `boa_token_id` 256000, `image_token_id` 258880, `audio_token_id` 258881,
`eoi_token_id` 258882, `eoa_token_index` 258883 (and `eoa_token_id` 258883 except in 12B-it) and,
where present, `video_token_id` 258884 ([hf-configs](../generated/hf-configs.md)).
<!-- c:MOD-TOK-010 --> Older Gemma markers (`<start_of_turn>`, `<end_of_turn>`, `<start_of_image>`,
`<start_of_audio>`) are not in the vocabulary.

Because the newline runs are single pieces, `\n\n` is one token (108) and `\n\n\n` another (109): the
whitespace a template emits changes the token sequence, not only the text.

### The Hugging Face tokenizer

Google's Hugging Face repositories ship `tokenizer.json` (32,169,626 bytes in `gemma-4-E4B-it` at
`ee0ef602`) for the `tokenizers` library, and `tokenizer_config.json` with `tokenizer_class`
`GemmaTokenizer`. <!-- c:MOD-TOK-012 --> The config names the chat-format tokens: `sot_token`
`<|turn>`, `eot_token` `<turn|>`, `soc_token` `<|channel>`, `eoc_token` `<channel|>`, `think_token`
`<|think|>`, `std_token`/`etd_token` `<|tool>`/`<tool|>`, `stc_token`/`etc_token`
`<|tool_call>`/`<tool_call|>`, `str_token`/`etr_token` `<|tool_response>`/`<tool_response|>`,
`escape_token` `<|"|>`, plus the image and audio markers. The instruction-tuned
models' config also carries a `response_schema` and `response_template` for parsing replies: a thought
between `<|channel>thought\n` and `<channel|>`, tool calls matched as `<|tool_call>call:NAME{…}<tool_call|>`,
and answer text closed by `<turn|>`, `<|tool_response>` or `<eos>`
([tokenizer_config.json](../generated/hf/google__gemma-4-E4B-it/tokenizer_config.json)).
<!-- c:MOD-TOK-014 -->

The `.litertlm` files don't carry the Hugging Face tokenizer, and LiteRT-LM prefers the SentencePiece
section when both exist (claims LM-CMP-002, LM-CMP-005). Whether the two tokenizers produce identical
IDs for every text was not tested; for the special tokens above they agree.

### How LiteRT-LM tokenizes

`SentencePieceTokenizer::TextToTokenIds` calls SentencePiece's `Encode` on the text and returns the IDs
([`support/tokenizer/sentencepiece_tokenizer.cc:65-73`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/support/tokenizer/sentencepiece_tokenizer.cc#L65-L73)).
Two consequences:

- **Markers are matched anywhere.** SentencePiece matches user-defined symbols as whole pieces wherever
  their text occurs, and the file template inserts user and system text only trimmed, not escaped. A
  user message containing the text `<turn|>` therefore becomes token 106, exactly as if the template had
  written it, and likewise for the other markers. Model-role text first passes through
  `strip_thinking`, which removes `<|channel>…<channel|>` spans. <!-- c:MOD-TOK-016 -->
- **The start token is added by the session, not the template.** LiteRT-LM passes an empty
  `bos_token` to templates
  ([`runtime/components/prompt_template.h:94-98`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/prompt_template.h#L94-L98));
  on the first prefill of a session, whether templates are applied outside or inside the session, it
  inserts the text of the start token (`<bos>`) as its own input and then replaces it with ID 2 after
  tokenizing
  ([`runtime/core/session_utils.cc:97-114`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/session_utils.cc#L97-L114),
  [`runtime/core/session_utils.cc:55-80`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/session_utils.cc#L55-L80)).
  <!-- c:MOD-TOK-017 --> The start token ID comes from the file's `start_token` (2) when the session
  config has none
  ([`runtime/engine/engine_settings.cc:749-768`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L749-L768)).

The report says the same about `[BOS]`: add the token after tokenizing, don't tokenize the text.


### The chat format as trained

From the model card (`google/gemma-4-E4B-it` at `ee0ef602`), the report (arXiv 2607.02770v1, Table 11)
and Google's prompt-formatting page (unversioned, *volatile*, "Last updated 2026-06-03"):

| Element | Format |
| --- | --- |
| Turn | `<|turn>ROLE\n` … `<turn|>`, roles `system`, `user`, `model` |
| Thinking on | `<|think|>` at the start of the system turn; remove it to turn thinking off |
| Thought | `<|channel>thought\n` … `<channel|>` before the answer |
| Tool declaration | `<|tool>declaration:NAME{…}<tool|>` in the system turn |
| Tool call | `<|tool_call>call:NAME{key:value,…}<tool_call|>` with strings in `<|"|>` |
| Tool response | `<|tool_response>response:NAME{…}<tool_response|>` |
| Media | `<|image|>`, `<|audio|>` placeholders; embeddings between `<|image>`…`<image|>` and `<|audio>`…`<audio|>` |

<!-- c:MOD-TOK-019 --> <!-- c:MOD-TOK-020 -->

Rules from Google that a runtime has to follow (card "Best Practices", prompt page):

- Thinking is enabled per conversation by `<|think|>` in the system prompt. <!-- c:MOD-TOK-019 -->
- Thoughts from earlier turns must be stripped from the history, "with the exception of tool call
  turns where thinking content should be preserved"; the prompt page says thoughts "must NOT be
  removed between the function calls" of one model turn. <!-- c:MOD-TOK-021 -->
- For all models except E2B and E4B, a model with thinking off may still emit an empty
  `<|channel>thought\n<channel|>` before the answer; the 12B, 26B-A4B and 31B Hugging Face templates
  therefore add that empty block to the generation prompt when thinking is off. <!-- c:MOD-TOK-022 -->
- After a tool call, the prompt page's example ends the model output with `<|tool_response>` and
  says it "acts as an additional stop sequence"; the application then appends the response.
  <!-- c:MOD-TOK-023 --> The report's Table 11 instead ends its tool-call example with
  `<tool_call|><turn|>`. <!-- c:MOD-TOK-024 --> Both 50 (`<|tool_response>`) and 106 (`<turn|>`) are
  stop tokens in the files, so LiteRT-LM stops on either. gemma.ai's function-calling page shows E2B,
  run with Transformers and the Hugging Face template, ending its tool call with `<|tool_response>`.
  <!-- c:X-GAI-007 -->
- Images go before the text and audio after it, for best results (card).
- The instruction-tuned models end each answer with `<turn|>`; base models end with `<eos>` (report §3).

### Stop tokens

| Source | Start | Stop IDs (in order) |
| --- | --- | --- |
| E2B, E4B (ours and Google's) and `-gpu` files, `LlmMetadata` | 2 | 1, 50, 106 |
| Hugging Face `generation_config.json` of E2B-it, E4B-it, 12B-it, 26B-A4B-it, 31B-it and the 31B assistant | `bos_token_id` 2 | 1, 106, 50 |
| `gemma-4-12B-it-assistant` `generation_config.json` | 2 | 1 |
| LiteRT-LM `models/gemma4/LlmMetadataProto_12b.pbtext` | `<bos>` (by string) | 1, 106, 50 |

<!-- c:MOD-TOK-026 -->

The three IDs are `<eos>` (1), `<|tool_response>` (50) and `<turn|>` (106). <!-- c:MOD-TOK-005 -->
LiteRT-LM copies the file's `stop_tokens` into the session when the caller gives none
([`runtime/engine/engine_settings.cc:749-768`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L749-L768));
the detector and the stream filter that keeps them out of the text are on
[../litert-lm/components.md](../litert-lm/components.md). The order matters in
one place: llguidance takes the first single-token stop sequence as its end token, which is 1 for
these files (claim LM-CNV-066).

The standard E2B/E4B files carry no `sampler_params`; the `-gpu` files carry `TOP_P` with k 64,
p 0.95 and temperature 1.0 ([e4b-gpu](../generated/models/e4b-gpu.md), section 0).
<!-- c:MOD-TOK-029 -->

### Four templates

| Template | Where | Used by |
| --- | --- | --- |
| **File template** (`a8c20bf5`) | `jinja_prompt_template` of the published E2B and E4B files, ours included; byte-identical to `chat_template.jinja` in the `litert-community` E2B and E4B repositories | LiteRT-LM unless overridden; Agello |
| **GPU file template** (`f72275b3`) | the `-gpu` files | Google's artisan executor |
| **Repository template** (`a6b935b2`) | LiteRT-LM `models/gemma4/chat_template_e2b_e4b.jinja`, also embedded in `LlmMetadataProto.pbtext` | files built from the repository metadata; none of the published files |
| **Hugging Face template** (`9d795d4e`) | `chat_template.jinja` in `google/gemma-4-E2B-it` and `-E4B-it`, "Published: 2026-07-09" | Transformers |

The labels are MD5 prefixes of the template text. <!-- c:MOD-TOK-030 -->
For 12B, 26B-A4B and 31B, Hugging Face publishes one template (`f016bd4b`) that differs from the
E2B/E4B one only in the empty thought block above, and LiteRT-LM's repository has a matching
`chat_template.jinja` embedded in the 12B and 26B/31B metadata. <!-- c:MOD-TOK-022 -->

The repository's README lists how its templates differ from Hugging Face's: tool responses in the
`content` of a `tool` message (Hugging Face reads `tool_responses`), system content as a string or a
list of parts, and turn termination decided by the previous message type
([`models/gemma4/README.md:1-36`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/models/gemma4/README.md#L1-L36)).
The repository template dates from 2026-08-21 and was refactored through
2026-09-17; the published files were created on 2026-04-28 (their `creation_timestamp`) and predate
it.

### What the file template renders

The file template is the one that runs in Agello.
[conversation.md](../litert-lm/conversation.md#the-gemma-4-template-in-the-e2b-and-e4b-files) describes its main rules (LM-CNV-025,
LM-CNV-026); rendered with the same inputs LiteRT-LM passes (see the method in
[the example](#a-two-turn-chat-with-thinking-and-a-tool-call)), the details that shape tokens are:

| Situation | File template | Hugging Face template | Repository template |
| --- | --- | --- | --- |
| Thinking on, no tools, no system message | `<|turn>system\n<|think|><turn|>\n` | `<|turn>system\n<|think|>\n<turn|>\n` | `<|turn>system\n<|think|><turn|>\n` |
| Thinking on, system message S, no tools | `<|turn>system\n<|think|>S<turn|>\n` (S trimmed; no newline after `<|think|>`) | `<|turn>system\n<|think|>\nS<turn|>\n` | `<|turn>system\n<|think|>\nS<turn|>\n` |
| Thinking on with tools | `<|think|>\n\n<|tool>…` | `<|think|>\n<|tool>…` | `<|think|>\n<|tool>…` |
| Image part in a user message | `\n\n<|image|>\n\n` | `<|image|>` | `\n\n<|image|>\n\n` |
| Audio part | `<|audio|>` | `<|audio|>` | `<|audio|>` |
| Assistant message with a tool call | `<|turn>model\n<|tool_call>…<tool_call|><turn|>\n` | `<|turn>model\n<|tool_call>…<tool_call|>` (then the responses, same turn) | `<|turn>model\n<|tool_call>…<tool_call|>` |
| The answer after a tool response, in history | `<tool_response|><|turn>model\nANSWER<turn|>\n` | `<tool_response|>ANSWER<turn|>\n` | `<tool_response|>ANSWER<turn|>\n` |
| Generation prompt after a tool response | none | `<|channel>thought\n` if thinking, else none | `<|channel>thought\n` if thinking, else none |
| Reasoning of earlier replies | never rendered (`strip_thinking`; history has no channel text) | rendered (`reasoning` or `reasoning_content`) for messages after the last user message, or for tool-calling messages when `preserve_thinking` is true | not rendered |
| Tool message shape it reads | `content: [{type: tool_response, name, response}]` | OpenAI `role: tool` with `content`, or `tool_responses` on the assistant message | `content: [{name, response}]` |

<!-- c:MOD-TOK-034 --> <!-- c:LM-CNV-025 --> <!-- c:LM-CNV-027 --> <!-- c:MOD-TOK-030 --> <!-- c:MOD-TOK-035 --> <!-- c:MOD-TOK-036 --> <!-- c:MOD-TOK-037 -->
<!-- c:MOD-TOK-038 -->

(The `-gpu` template is the file template plus `<|channel>thought\n` as the generation prompt after a
tool response when thinking is on; claim LM-CNV-028.)

**Tool declarations.** A tool `{name, description, parameters}` becomes
`<|tool>declaration:NAME{description:<|"|>…<|"|>,parameters:{properties:{…},required:[…],type:<|"|>OBJECT<|"|>}}<tool|>`;
property types are upper-cased and every string is wrapped in `<|"|>`. <!-- c:MOD-TOK-039 -->

**Tool calls.** `<|tool_call>call:NAME{key:value,…}<tool_call|>`, keys unquoted and sorted, strings
in `<|"|>`, numbers and booleans bare. A call whose `arguments` is a JSON string is inserted verbatim
between the braces by the file template; the Hugging Face template raises an error for it. <!-- c:MOD-TOK-040 -->

**Tool responses.** `<|tool_response>response:NAME{key:value,…}<tool_response|>`, keys sorted; a
response that isn't an object becomes `{value:…}`. Several responses follow one another without
separators. <!-- c:MOD-TOK-041 -->

LiteRT-LM's Swift wrapper builds exactly the shape the file template reads: a `tool` message whose
`content` is a list of `{"type": "tool_response", "name", "response"}`
([`swift/Conversation.swift:397-425`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L397-L425)),
in agello-next and in the shipped wrapper. The Hugging Face template doesn't read
that shape: it collects only `text` parts of a tool message, so the same message renders as
`<|tool_response>response:get_weather{value:<|"|><|"|>}<tool_response|>`, an empty response.
<!-- c:MOD-TOK-043 --> Overriding the template with Google's Hugging Face one through
`ConversationConfig.chatTemplate` therefore breaks tool calling in LiteRT-LM.

**Media.** LiteRT-LM never feeds the placeholders to the model. The Gemma 4 processor splits the
rendered text at `<|image|>`, `<start_of_image>`, `<|audio|>` and `<start_of_audio>`, and replaces each
image placeholder with the text so far plus `<|image>`, the image's embeddings, and an image-end
input ([`runtime/conversation/model_data_processor/gemma4_data_processor.cc:115-129`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/conversation/model_data_processor/gemma4_data_processor.cc#L115-L129),
[`runtime/conversation/model_data_processor/multimodal_processor_helper.cc:115-188`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/multimodal_processor_helper.cc#L115-L188)).
<!-- c:MOD-TOK-044 --> So with the file template an image costs `\n\n\n` (109) before `<|image>` and
`\n\n` (108) after the image end, plus the image tokens (280 by default; claim LM-CNV-034).

### How LiteRT-LM applies the template

Covered on [../litert-lm/conversation.md](../litert-lm/conversation.md); in short:

1. `Conversation` renders the whole message list with minijinja (trailing newlines kept, blocks
   trimmed) and prefills only the difference from the previous render (claims LM-CNV-018, LM-CNV-021).
2. Thinking on adds `enable_thinking` to the template inputs and makes the session track the
   `thought` channel ([`runtime/conversation/conversation.cc:360-388`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/conversation.cc#L360-L388)).
3. After a reply with reasoning, agello-next rewinds to the end of the prefill and refills the reply
   rendered without it (claims LM-CNV-050, LM-CNV-052).

Step 3 applies to tool-call replies too: the thought before a tool call leaves the KV cache before
the tool response is prefilled, and the file template never renders it back. That departs from
Google's rule to keep thoughts within a tool-calling turn. <!-- c:MOD-TOK-045 --> The effect on
answer quality was not measured.

### A two-turn chat with thinking and a tool call

The chat: thinking on, one tool `get_weather(city)`, the user asks "Weather in Paris?", the model
thinks and calls the tool, the tool answers, the model answers, the user says "Thanks!", the model
answers again. The messages are in LiteRT-LM's shapes (the tool response as the Swift wrapper sends it;
the assistant replies as `Conversation` stores them, with reasoning in `channels`, not `content`).

Method: the file template (from `e4b-ours`, the same in every published E2B/E4B file) rendered with
Python Jinja2 configured like LiteRT-LM's minijinja environment (`trim_blocks`, `lstrip_blocks`,
`keep_trailing_newline`, empty `bos_token`;
[`runtime/components/rust/minijinja_template.rs:177-192`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/rust/minijinja_template.rs#L177-L192)),
each prefill computed as LiteRT-LM does (new render minus old render), and every string encoded with
SentencePiece 0.2.2 from the file's own `SP_Tokenizer` section. The model's
outputs are illustrative; their token IDs are the encoding of that text, which the model may
segment differently when it generates.

**Turn 1, step 1: first prefill** (the session's start token first; the length depends on the tool's
description strings):

```text
<|turn>system\n<|think|>\n\n<|tool>declaration:get_weather{description:<|"|>Current weather for a city.<|"|>,parameters:{properties:{city:{description:<|"|>City name<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>city<|"|>],type:<|"|>OBJECT<|"|>}}<tool|><turn|>\n<|turn>user\nWeather in Paris?<turn|>\n<|turn>model\n
```

```text
2 <bos> | 105 <|turn> | 9731 system | 107 \n | 98 <|think|> | 108 \n\n | 46 <|tool> |
163688 declaration | 236787 : | 828 get | 236779 _ | 19323 weather | 236782 { |
7777 description | 236787 : | 52 <|"|> | 8782 Current | 7606 ▁weather | 573 ▁for | 496 ▁a |
3207 ▁city | 236761 . | 52 <|"|> | 236764 , | 19031 parameters | 29616 :{ | 15921 properties |
29616 :{ | 13319 city | 29616 :{ | 7777 description | 236787 : | 52 <|"|> | 17698 City |
1463 ▁name | 52 <|"|> | 236764 , | 2084 type | 236787 : | 52 <|"|> | 35410 STRING | 52 <|"|> |
5237 }}, | 15979 required | 24845 :[ | 52 <|"|> | 13319 city | 52 <|"|> | 1604 ], | 2084 type |
236787 : | 52 <|"|> | 60688 OBJECT | 52 <|"|> | 1807 }} | 47 <tool|> | 106 <turn|> | 107 \n |
105 <|turn> | 2364 user | 107 \n | 26588 Weather | 528 ▁in | 9079 ▁Paris | 236881 ? |
106 <turn|> | 107 \n | 105 <|turn> | 4368 model | 107 \n
```

<!-- c:MOD-TOK-047 -->

**Step 2: the model generates** a thought and a tool call, then stops on `<|tool_response>` (50) or
`<turn|>` (106). Encoded, `<|channel>thought\nCall the tool.<channel|><|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>`
is 22 tokens: `100 45518 107 7029 506 5904 236761 101 48 6639 236787 828 236779 19323 236782 13319 236787 52 50429 52 236783 49`.

**Step 3: refill without the thought** (16 tokens). The reply has a `thought` channel, so the session
rewinds to the end of step 1 and prefills the reply as the template renders it:

```text
<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|><turn|>\n
48 <|tool_call> | 6639 call | 236787 : | 828 get | 236779 _ | 19323 weather | 236782 { |
13319 city | 236787 : | 52 <|"|> | 50429 Paris | 52 <|"|> | 236783 } | 49 <tool_call|> |
106 <turn|> | 107 \n
```

<!-- c:MOD-TOK-048 -->

**Step 4: the tool response** (21 tokens; no generation prompt follows a tool response in this
template):

```text
<|tool_response>response:get_weather{sky:<|"|>sunny<|"|>,temp_c:18}<tool_response|>
50 <|tool_response> | 6275 response | 236787 : | 828 get | 236779 _ | 19323 weather | 236782 { |
16012 sky | 236787 : | 52 <|"|> | 185060 sunny | 52 <|"|> | 236764 , | 7617 temp | 236779 _ |
236755 c | 236787 : | 236770 1 | 236828 8 | 236783 } | 51 <tool_response|>
```

<!-- c:MOD-TOK-049 -->

**Step 5: the model answers** "18 °C and sunny." (7 tokens: `236770 236828 14324 236780 532 17635
236761`) and stops on `<turn|>`. If that reply also had a thought, the refill is (12 tokens):

```text
<|turn>model\n18 °C and sunny.<turn|>\n
105 <|turn> | 4368 model | 107 \n | 236770 1 | 236828 8 | 14324 ▁° | 236780 C | 532 ▁and |
17635 ▁sunny | 236761 . | 106 <turn|> | 107 \n
```

<!-- c:MOD-TOK-050 -->

Note the `<|turn>model\n` the model never saw while answering: the file template renders the answer
after a tool response as a new model turn.

**Turn 2, step 6: the second user message** (10 tokens):

```text
<|turn>user\nThanks!<turn|>\n<|turn>model\n
105 <|turn> | 2364 user | 107 \n | 12879 Thanks | 236888 ! | 106 <turn|> | 107 \n |
105 <|turn> | 4368 model | 107 \n
```

<!-- c:MOD-TOK-051 -->

The model answers "You're welcome." (`3048 236789 500 8349 236761`) and `<turn|>`.

**The whole chat, rendered by the file template:**

```text
<|turn>system\n<|think|>\n\n<|tool>declaration:get_weather{…}<tool|><turn|>\n
<|turn>user\nWeather in Paris?<turn|>\n
<|turn>model\n<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|><turn|>\n
<|tool_response>response:get_weather{sky:<|"|>sunny<|"|>,temp_c:18}<tool_response|>
<|turn>model\n18 °C and sunny.<turn|>\n
<|turn>user\nThanks!<turn|>\n
<|turn>model\nYou're welcome.<turn|>\n
```

(Line breaks added between turns for reading; the `\n` shown are the only newlines.)

**The same chat in Google's format.** The Hugging Face template, given the same chat in its own
message shape (an OpenAI-style `tool` message with `tool_call_id`), renders the tool turn as one model
turn: `<|turn>model\n<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|><|tool_response>response:get_weather{…}<tool_response|>18 °C and sunny.<turn|>\n`,
and its generation prompt after the tool response is `<|channel>thought\n` when thinking is on.
LiteRT-LM's repository template renders the same turn structure. <!-- c:MOD-TOK-036 -->
<!-- c:MOD-TOK-037 --> Compared with the file template, Google's form has no `<turn|>\n` (106, 107)
after the tool call and no `<|turn>model\n` (105, 4368, 107) before the answer.

## On Apple

Nothing here depends on the platform: the tokenizer and template run inside `CLiteRTLM` on iOS and
macOS as anywhere else. What Agello does with them:

- **The file template.** Agello sets no `chatTemplate`, so its chats use the template inside the E4B
  file (Agello `Agello/Inference/ChatEngine.swift` lines 309-334 at `93122b6`). <!-- c:MOD-TOK-053 -->
- **Thinking per conversation.** It passes a `ThinkingConfig` when thinking is on, which puts
  `<|think|>` at the top of the system turn for the conversation's life (claim LM-CNV-025 and
  conversation.md, "Per-message thinking toggles"). <!-- c:MOD-TOK-053 -->
- **Tools.** When tools are enabled, Agello declares `CurrentDateTimeTool` and `CalculatorTool`; it
  declares `WebSearchTool` whenever `options.webSearchTool` is set, whether or not the other tools are
  enabled. The system turn holds their declarations. <!-- c:MOD-TOK-053 -->
- **Reseeds are plain text.** When Agello recreates a conversation it replays saved turns as plain
  user and model text messages, so earlier tool calls and responses are not rendered in their Gemma 4
  form after a reseed. <!-- c:MOD-TOK-054 -->
- **The shipped build is the same here.** The template, the stop tokens and the tool-message shape
  come from the file and the Swift wrapper, which are the same for v0.18.0 and agello-next.

## Settings

| Setting | Layer | Effect on the chat format |
| --- | --- | --- |
| `ConversationConfig.chatTemplate` / `litert_lm_conversation_config_set_prompt_template` | Swift / C | replaces the file template (the Hugging Face template breaks tool responses: above) |
| `ThinkingConfig(enableThinking:thinkingTokenBudget:)` | Swift | `<|think|>` in the system turn; thought tracking and budget |
| `ExperimentalFlags.filterChannelContentFromKvCache` | Swift | nil keeps the default (true): reasoning leaves the KV cache after each reply, tool-call replies included |
| `SessionConfig` stop tokens | C++ | replace the file's 1, 50, 106 |
| `visualTokenBudget` | Swift | image tokens between `<|image>` and the image end |

The rows live in [../configuration.md](../configuration.md).

## Limits and failure modes

- **Marker injection.** Text that contains `<|turn>`, `<turn|>`, `<|tool_call>` or any other marker is
  encoded as the control token. A pasted document or a web page that contains such strings can end a
  turn or open a tool call in the prompt the model sees. Nothing in LiteRT-LM escapes them.
- **The file template differs from Google's around tools.** It closes the model turn after a tool
  call with `<turn|>\n` and opens a new `<|turn>model\n` for the answer after a tool response, and it
  doesn't keep thoughts within a tool-calling turn. Google's current template, its prompt page and
  LiteRT-LM's own repository template do neither. Whether this changes E4B's tool use was not measured.
- **The KV cache doesn't always hold the render.** After a tool call without reasoning nothing is
  refilled: the cache holds what was generated (ending at the stop token), while later renders contain
  `<turn|>\n` after the call and `<|turn>model\n` before the answer. Traced in the code: the stop
  token is written at the next prefill from an id-and-zeros row instead of its embedding
  <!-- c:X-GAP-003 -->, and when `<turn|>` was the last token of its decode step or MTP round (so it
  is the pending token), the `\n` after it is never written <!-- c:X-GAP-002 -->; after a reply that is
  only a tool call and stopped on `<|tool_response>` or `<turn|>` as the last token of its step or MTP
  round, the tool response follows the stop token's slot directly, and the answer follows
  `<tool_response|>` without `<|turn>model\n` <!-- c:X-GAP-006 --> ([../litert-lm/conversation.md](../litert-lm/conversation.md#what-the-kv-cache-holds-at-a-turn-boundary)).
- **Template overrides need LiteRT-LM's message shapes.** A template written for Transformers
  (OpenAI-style tool messages, `tool_call_id`, `reasoning_content`) silently drops data from
  LiteRT-LM's messages, as the empty tool response above shows.
- **A JSON-string `arguments` in a replayed tool call** is pasted verbatim by the file template and
  rejected by the Hugging Face template.

## Where in the code

| Concept | Symbol | Citation |
| --- | --- | --- |
| Encoding | `SentencePieceTokenizer::TextToTokenIds` | [`support/tokenizer/sentencepiece_tokenizer.cc:65-73`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/support/tokenizer/sentencepiece_tokenizer.cc#L65-L73) |
| Start token text | `MaybeGetBosString` | [`runtime/core/session_utils.cc:41-49`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/session_utils.cc#L41-L49) |
| Start token ID insertion | `StringToProcessedInputText` | [`runtime/core/session_utils.cc:55-80`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/session_utils.cc#L55-L80) |
| Empty `bos_token` for templates | `PromptTemplateInput` | [`runtime/components/prompt_template.h:94-98`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/prompt_template.h#L94-L98) |
| minijinja environment | `create_env` | [`runtime/components/rust/minijinja_template.rs:177-192`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/rust/minijinja_template.rs#L177-L192) |
| Start and stop tokens from metadata | `SessionConfig::MaybeUpdateAndValidate` | [`runtime/engine/engine_settings.cc:749-768`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L749-L768) |
| Media placeholders | `Gemma4DataProcessor::ToInputDataVectorImpl`, `ProcessMultimodalPrompt` | [`gemma4_data_processor.cc:115-129`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/conversation/model_data_processor/gemma4_data_processor.cc#L115-L129), [`multimodal_processor_helper.cc:115-188`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/multimodal_processor_helper.cc#L115-L188) |
| Reply to message (tool-call parsing) | `ResponseTextToMessage` | [`runtime/conversation/model_data_processor/data_utils.cc:86-108`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/data_utils.cc#L86-L108) |
| Tool declaration JSON (Swift) | `Tool.getSchema` | [`swift/Tool.swift:224-273`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Tool.swift#L224-L273) |
| Tool response message (Swift) | `handleToolCalls` | [`swift/Conversation.swift:397-425`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L397-L425) |
| Render difference per turn | `GetSingleTurnTextFromFullHistory` | [`runtime/conversation/conversation.cc:292-315`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/conversation.cc#L292-L315) |
| Repository templates and their tests | `chat_template_e2b_e4b.jinja`, `chat_template_test` | [`models/gemma4/chat_template_e2b_e4b.jinja:123-199`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/models/gemma4/chat_template_e2b_e4b.jinja#L123-L199), [`models/gemma4/BUILD:27-60`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/models/gemma4/BUILD#L27-L60) |
| Template text in each file | `jinja_prompt_template` | [e4b-ours](../generated/models/e4b-ours.md), [e4b-gpu](../generated/models/e4b-gpu.md), section 0 |

## Open questions

- **Generated vs encoded tokens.** The worked example encodes text; a token dump of a real E4B run
  (two turns, a tool call) would show the IDs the model actually emits and confirm what the code trace
  says the KV cache holds at each boundary
  ([../litert-lm/conversation.md](../litert-lm/conversation.md#what-the-kv-cache-holds-at-a-turn-boundary)).
- **Does the file template hurt tool use?** Compare E4B with the file template and with LiteRT-LM's
  repository template (which matches Google's turn structure and reads LiteRT-LM's message shapes)
  on Agello's tool scenarios. Not done.
- **Thoughts within a tool-calling turn.** Google says to keep them; LiteRT-LM removes them. The filter
  is set per conversation (C++) or process-wide (Swift `ExperimentalFlags`), with no per-turn switch.
  Effect not measured.
- **Hugging Face tokenizer parity.** `tokenizer.json` wasn't downloaded; that it encodes every text
  like the SentencePiece section is assumed, not tested.
- **Unverified: length of the whole example chat.** The length of the whole example chat isn't
  established: it depends on the tool's description strings and the second answer's text
  (MOD-TOK-052, unverifiable).
- **Report Table 11 vs the prompt page** on what ends a tool-call output (`<turn|>` or
  `<|tool_response>`): the runtime stops on both, but which one E4B emits was not observed. Google's
  E2B example (Transformers, Hugging Face template) emits `<|tool_response>`; E4B with the file's
  template under LiteRT-LM is still unobserved.
