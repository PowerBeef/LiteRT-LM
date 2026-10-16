# Returns: mod-family (B2, models/gemma4-family.md and models/tokenizer-and-chat-format.md)

## Settings rows (for configuration.md)

name | layer | default | where read (citation link) | effect on Apple | version added | Agello sets it? | notes
--- | --- | --- | --- | --- | --- | --- | ---
`ConversationConfig.chatTemplate` | Swift | nil (the file's `jinja_prompt_template`) | [`swift/Engine.swift:325-327`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Engine.swift#L325-L327) | replaces the Gemma 4 chat template | present in v0.18.0 | no | Google's Hugging Face template drops LiteRT-LM tool responses (MOD-TOK-043); LiteRT-LM's `models/gemma4/chat_template_e2b_e4b.jinja` reads LiteRT-LM's shapes and follows Google's turn structure (MOD-TOK-036, MOD-TOK-037)
`LlmMetadata.start_token` / `stop_tokens` (file) → `SessionConfig` start/stop token IDs | C++ (engine settings) | from the file: start 2, stops 1, 50, 106 | [`runtime/engine/engine_settings.cc:749-768`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L749-L768) | same on all platforms | — | no (file defaults) | only stop tokens given as token IDs are copied (MOD-TOK-028)
`LlmMetadata.jinja_prompt_template` (file) | model file | the file's template (`a8c20bf5…` in the published E2B/E4B files) | see LM-CNV-003 | — | — | no | four variants in circulation (MOD-TOK-030)

## Terms (for glossary.md)

term | meaning | prefer / avoid
--- | --- | ---
E2B / E4B | Gemma 4 sizes counted in "effective" parameters, i.e. without the per-layer embedding tables | prefer "E4B"; avoid "4B"
A4B | "active" 4B: the 26B mixture-of-experts model uses about 3.8B parameters per token | prefer "26B-A4B"
Unified (12B) | Gemma 4 12B: encoder-free; image patches and raw audio projected into the decoder | —
Assistant model | a 4-layer MTP drafter for one Gemma 4 target (Hugging Face suffix `-assistant`) | prefer "MTP drafter"; avoid "assistant" alone in user text (confused with the chat role)
Per-layer embeddings (PLE) | a second, per-layer token embedding in E2B/E4B (256 wide per layer) | prefer "per-layer embeddings"
KV sharing | the last `num_kv_shared_layers` layers reuse earlier layers' KV caches (E2B 20, E4B 18) | —
p-RoPE | proportional RoPE on global layers, rotary angles on 25% of each head | —
File template | the `jinja_prompt_template` inside a `.litertlm` file | avoid "the chat template" without saying which
Control token / user-defined symbol | SentencePiece piece types: `<pad>`, `<eos>`, `<bos>` vs the chat markers (`<|turn>`, …) | —
Placeholder | `<|image|>` / `<|audio|>` in rendered text, replaced by media embeddings in LiteRT-LM | —

## Contradictions

- NEW-1 (what ends a tool-call output): the report's Table 11 ends it with `<tool_call|><turn|>`; Google's prompt page and the Hugging Face template follow a tool call with `<|tool_response>`. Both 50 and 106 are stop tokens in the files, so LiteRT-LM stops on either; which one E4B emits is unobserved. Claims MOD-TOK-023, MOD-TOK-024, MOD-TOK-026.
- NEW-2 (thoughts in tool-calling turns): Google says to keep them within a tool-calling turn; agello-next's default channel filter and the file template remove them. Resolution: documented as a departure; quality effect unmeasured. Claims MOD-TOK-021, MOD-TOK-045.
- NEW-3 (KV at 32K for E2B/E4B): the report's Table 3 (+0.05 / +0.14 GB) is half of the config arithmetic (103.8 / 278.9 MB), which matches the exported cache layout; 12B/26B/31B agree. Kept both, page uses the arithmetic. Claims MOD-FAM-043, MOD-FAM-045, MOD-FAM-047.
- NEW-4 (turn structure around tools): the template inside the published E2B/E4B files closes the model turn after a tool call and reopens `<|turn>model\n` for the answer after a tool response; Google's Hugging Face template, its prompt page and LiteRT-LM's own repository template keep one model turn. Resolution: the files' template is what runs (Agello included); recorded as a difference to test. Claims MOD-TOK-030, MOD-TOK-036, MOD-TOK-037, MOD-TOK-050.
- C14 (context 128K vs 32K) is B1's; this page states trained 131,072 vs the files' 32003 window as background (MOD-FAM-013, MOD-FAM-042) without resolving it.
- C1 (app limit 6.4 vs 8.6 GB) is B3's; the 12B verdict uses the 6.4 GB figure from `results/iphone/2026-10-15/README.md` and names both (MOD-FAM-052).

## Notes for other pages

- **litert-lm/conversation.md** (Run A):
  - Line 273-274 ("Generation prompt ... omitted right after a tool response so the model continues its turn", LM-CNV-025): accurate about the generation prompt, but in the file template the model's turn was already closed by `<turn|>\n` after the tool call, and history renders the answer under a new `<|turn>model\n` (MOD-TOK-036, MOD-TOK-037, MOD-TOK-050). Suggest rewording "continues its turn".
  - Its open question "Same question after a tool call" can cite MOD-TOK-048/MOD-TOK-050 for the exact strings and token IDs of the refills.
  - Add that the channel filter also removes thoughts from tool-call replies, against Google's guidance (MOD-TOK-045).
  - Overriding `chatTemplate` with Google's Hugging Face template breaks tool responses (MOD-TOK-043); LiteRT-LM's repository template (`models/gemma4/chat_template_e2b_e4b.jinja`, `a6b935b2`) is a fourth variant not mentioned (MOD-TOK-030, MOD-TOK-031).
  - LM-CNV-027 says the HF template writes `<|think|>\n` where the file writes `<|think|>`; with tools the file template writes `<|think|>\n\n` (MOD-TOK-034). Not wrong, but incomplete.
- **litert-lm/components.md**: the SentencePiece model's own facts (BPE, byte fallback, split digits, identity normalizer, no dummy prefix, 6,410 user-defined pieces) are in MOD-TOK-002/003; marker strings in user text become control tokens (MOD-TOK-016).
- **gemma4-e4b.md / gemma4-e2b.md** (B1): exported KV caches total 917,590,016 bytes for E4B at 32,003 positions (MOD-FAM-047), relevant to C9 (918 MB = 917.6 MB in 10^6 bytes; 875 MiB). Global layers in the files keep separate 512-wide K and V caches (MOD-FAM-047). The trained E4B needs only 278.9 MB at 32K because local layers are bounded by the 512 window (MOD-FAM-043). Google's iPhone 17 Pro figures: E2B 1,450 MB, E4B 3,380 MB (MOD-FAM-046). The E4B drafter (`google/gemma-4-E4B-it-assistant` `8d0031ea`) has width 256 and 2,048 token clusters with top-32 (MOD-FAM-032).
- **embeddinggemma.md** (mod-embedding): the EmbeddingGemma 2 file's `SP_Tokenizer` section is byte-identical to the Gemma 4 E4B one (SHA-256 `e594c8a9…`, MOD-TOK-001).
- **apple-platform.md / memory.md**: phone verdicts per model and the KV arithmetic table (MOD-FAM-043, MOD-FAM-044, MOD-FAM-052).
- **patches.md / contributing.md**: a candidate upstream report: the published E2B/E4B files carry an older template whose tool-turn structure differs from Google's and from LiteRT-LM's own `models/gemma4` template (MOD-TOK-030, MOD-TOK-036, MOD-TOK-037). Needs a quality measurement before posting.
- **facts.md**: vocabulary 262,144; stop tokens 1/50/106 = `<eos>`/`<|tool_response>`/`<turn|>`; trained context 131,072 (E2B/E4B) and 262,144 (others); license Apache 2.0.
- Version note: the technical report text read was arXiv 2607.02770v1 (Hugging Face paper page); v2 (2026-07-24) exists and was not compared.
- Method note: tokenizer facts come from parsing section 1 of the local E4B files with a protobuf wire-format reader and the SentencePiece 0.2.2 Python wheel found in the user's uv cache (not installed anywhere); templates were rendered with Jinja2 from the same cache.

## Open questions

- Which stop token E4B emits after a tool call (`<|tool_response>` or `<turn|>`), and what the KV cache holds at each tool-turn boundary (needs a token dump; no engine runs in this run).
- Whether the file template's tool-turn structure, or removing thoughts within tool turns, hurts E4B's tool use compared with LiteRT-LM's repository template.
- Why the report's Table 3 KV figures for E2B/E4B are half the config arithmetic.
- Whether the Hugging Face `tokenizer.json` encodes all text like the SentencePiece section (not downloaded).
- Base checkpoints of 26B-A4B and 31B have more parameters than their `-it` versions; not investigated.
- Whether the 12B and 31B drafters use their centroid fields (`use_ordered_embeddings` false).
- Whether agello-next can load `gemma-4-12B-it.litertlm` (`gemma4_unified`, `merge_patches`, `skip_mel_spectrogram_extraction`) at all; only memory was considered.
