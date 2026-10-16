# Returns: mod-embedding (B2, models/embeddinggemma.md)

## Settings rows (for configuration.md)

name | layer | default | where read (citation link) | effect on Apple | version added | Agello sets it? | notes
--- | --- | --- | --- | --- | --- | --- | ---
`EmbeddingEngineConfig.backend` | Swift | `.cpu()` | [`swift/EmbeddingEngineConfig.swift:42-58`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/EmbeddingEngineConfig.swift#L42-L58) | `.gpu`: text encoder on Metal through ML Drift; the token embedder stays on the CPU either way | ≤ v0.18.0 | yes: `.gpu`, then `.cpu()` on failure | MOD-EMB-059, MOD-EMB-072
`EmbeddingEngineConfig.cacheDir` | Swift → C | nil (the model file's folder) | [`swift/EmbeddingEngine.swift:67-123`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/EmbeddingEngine.swift#L67-L123) | GPU program and weight caches (0.18 GB on iPhone 17 Pro, our build) | ≤ v0.18.0 | yes: `Caches/com.patricedery.agello.litertlm/embedding[-<build id>]` | MOD-EMB-061, MOD-EMB-070
`EmbeddingEngineConfig.maxInputLength` / `litert_lm_embedding_engine_settings_set_max_input_length` | Swift → C → `EmbeddingEngineSettings` | nil → the file's `EmbeddingMetadata.max_input_length` (1,024 for embeddinggemma-2-text-270m) | [`runtime/engine/embedding_engine_settings.cc:266-298`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/embedding_engine_settings.cc#L266-L298) [`runtime/executor/model_signature_utils.cc:202-236`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/model_signature_utils.cc#L202-L236) | which encoder signatures compile (128-1,024 by default), hence the longest input; > 8,192 fails engine creation | ≤ v0.18.0 | no | MOD-EMB-004, MOD-EMB-050, MOD-EMB-065
`litert_lm_embedding_engine_settings_set_min_input_length` / `EmbeddingMetadata.min_input_length` | C / metadata | unset | same as above | drops signatures shorter than it | ≤ v0.18.0 | no (not in Swift) | 
`litert_lm_embedding_engine_settings_set_activation_data_type` | C | GPU: FLOAT16 (unless the section's `prefer_activation_type` says otherwise); CPU: unset | [`runtime/engine/embedding_engine_settings.cc:183-208`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/embedding_engine_settings.cc#L183-L208) [`runtime/executor/litert_compiled_model_executor_utils.cc:1017-1031`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/litert_compiled_model_executor_utils.cc#L1017-L1031) | GPU precision kFp16 vs kFp32 for the text encoder | ≤ v0.18.0 | no (Swift has no setter) | card says not to run the model in float16: MOD-EMB-049, MOD-EMB-073
`EmbeddingOptions.normalize` | Swift / C / C++ | true | [`runtime/core/embedding_engine_impl.cc:1589-1624`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/embedding_engine_impl.cc#L1589-L1624) | L2 normalization on the host after truncation | ≤ v0.18.0 | yes: true | MOD-EMB-054
`EmbeddingOptions.outputSize` | Swift / C / C++ | none (768) | same as above | Matryoshka truncation before normalization; > 768 or < 0 is an error | ≤ v0.18.0 | yes: 256 | MOD-EMB-005, MOD-EMB-054
`EmbeddingOptions.insertSpecialTokens` | Swift / C / C++ | true | [`runtime/core/embedding_engine_impl.cc:1260-1318`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/embedding_engine_impl.cc#L1260-L1318) | BOS 2 and EOS 1 around the input (+2 tokens) | ≤ v0.18.0 | default | MOD-EMB-051
`input_overflow_strategy` (`litert_lm_embedding_options_set_input_overflow_strategy`) | C / C++ | `kError` | [`runtime/executor/embedding_litert_compiled_model_executor.cc:520-549`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/embedding_litert_compiled_model_executor.cc#L520-L549) | inputs over the longest loaded signature: error, truncate (EOS kept), or chunk and average | ≤ v0.18.0 | no (not in Swift) | MOD-EMB-053, MOD-EMB-075

## Terms (for glossary.md)

term | meaning | prefer / avoid
--- | --- | ---
EmbeddingGemma 2 | Google DeepMind's multimodal embedding model (text, image, video, audio → 768 dimensions), 740M parameters; the text part is 270M | prefer "EmbeddingGemma 2 text 270m" for the file Agello uses; avoid "EmbeddingGemma" alone (also names the first, Gemma 3 based model)
task prompt | the text prefix that tells the model what an embedding is for (`task: search result \| query: `, `title: none \| text: `) | prefer "task prompt" or "prefix"; avoid "instruction" (not an instruction-tuned chat prompt)
Matryoshka dimensions (MRL) | leading sub-vectors (128, 256, 512 of 768) trained to work alone; truncate, then re-normalize | prefer "truncate to N dimensions"; avoid "compress" or "quantize" (different things)
signature (embedding) | a fixed-length entry point of the text encoder (`encoder_1x128` … `encoder_1x8192`) | prefer "signature"; avoid "context size"
`max_input_length` | the bound that decides which encoder signatures load (1,024 from the file by default) | distinguish from the trained context (8,192) and the longest exported signature (8,192)
projection-only per-layer embeddings | EmbeddingGemma 2's per-layer inputs computed from the token embeddings by one linear map, with no per-layer token table | avoid calling it the "per-layer embedder" (the E4B section of that name is a table lookup)
mean pooling | averaging the final token states over real tokens | —

## Contradictions

- NEW-a (sliding window): the google/embeddinggemma-2 card says "Sliding Window 1024 tokens", config.json says `sliding_window: 512`. Resolved: transformers treats 512 as a radius on each side of a bidirectional window (half of the reference's 1024-wide window), so they describe the same window. The export adds a twist: signatures ≤ 1,024 have no window and the longer ones use offsets −512..+511, so the LiteRT file differs from transformers for 513-1,024-token inputs and by one key above. Claims MOD-EMB-016, -017, -043, -044, -045.
- NEW-b (context): card 8,192-token context and `max_position_embeddings` 262,144 vs LiteRT-LM's runtime limit of 1,024. Resolved as three limits: trained 8,192 (card), exported signatures to 8,192 (file), loaded 128-1,024 by default because of the file's `max_input_length: 1024`. Claims MOD-EMB-004, -013, -039, -047, -050.
- NEW-c (float16): the card says "Do not use float16" (activation range exceeds float16) while LiteRT-LM runs the GPU text encoder with FP16 activations by default. Not resolved: recorded with Agello's probe (expected ranking, scores within 0.001 across GPU paths) and an open question; no FP32 comparison exists. Claims MOD-EMB-049, -073, -068, -069.
- NEW-d (130M + 140M): the card's split is reproduced by the config's matrix count only if "embedder" includes the per-layer projection; recorded as our reading. Claims MOD-EMB-023, -024.

## Notes for other pages

- litert-lm/executors.md (Embedding executor, LM-EXE-083 "token (and per-layer) EmbeddingLookupManager instances"): correct in code, but for embeddinggemma-2-text-270m there is no per-layer embedder section, so only the token lookup runs; the per-layer projection is inside the encoder graph (MOD-EMB-036, -052). Also worth adding there: the default longest input is 1,024 tokens for this file, not 8,192, because the metadata's `max_input_length` limits which signatures compile (MOD-EMB-004, -050). LM-EXE-033's "kTruncate cuts to the largest signature" means the largest *loaded* one.
- litert-lm/swift-api.md (EmbeddingEngine) and c-api.md (Embedding engine): the Swift wrapper also lacks the activation data type, so Swift users can't leave FP16 on the GPU (MOD-EMB-057); Swift `EmbeddingResponse` drops input length, truncation and chunk count.
- litert-lm/components.md / tokenizer-and-chat-format.md (mod-family): the EmbeddingGemma 2 file's SP_Tokenizer section is byte-identical to E4B's (sha256 e594c8a9…, MOD-EMB-033).
- litert/ml-drift-delegate.md: its composite table counts `odml.rope` 48 for `encoder_1x1024`; the split is 40 (θ 10,000, sliding) and 8 (θ 1,000,000, full) (MOD-EMB-040).
- memory.md: the embedder's INT4 weights are 135,462,912 bytes in the file (MOD-EMB-074); its GPU cache folder was 179,974,496 bytes on iPhone 17 Pro (MOD-EMB-070); per-signature attention tensors are 4n² values (8 MiB at FP16 for n = 1,024, 512 MiB for 8,192; MOD-EMB-046). Probe 3's footprint is the whole process.
- apple-platform.md: the embedder runs on the CPU and the encoder on the GPU on Apple (MOD-EMB-072); Google's iPhone 18 Pro figures (MOD-EMB-071).
- Agello (not a reference page; leads for the app's docs): the catalog downloads EmbeddingGemma at revision `main` rather than a commit (MOD-EMB-058), unlike E4B; Agello's chunker limits characters while the runtime limit is 1,022 text tokens (MOD-EMB-065); a single over-long passage leaves its whole document on keyword search (MOD-EMB-066).

## Open questions

- Which sliding-window behavior Google's reference implementation has (the export's none-up-to-1,024 and −512..+511, or transformers' ±512); a numeric comparison of the LiteRT file against transformers for a 600-token input would settle whether the export diverges.
- Quality of the FP16 GPU encoder against FP32 (CPU, or the C activation setter with FLOAT32); not measured.
- Actual token counts of Agello's 1,600-character passages per language and whether any exceed 1,022 tokens.
- Whether the Sentence Transformers pipeline adds `<bos>`/`<eos>` like LiteRT-LM (tokenizer config doesn't say).
- No EmbeddingGemma 2 paper found (arXiv, HF Papers, 2026-10-15); training details are only known for the first model (arXiv 2509.20354v3).
- Whether Google's prebuilt v0.18.0 binary behaves as its sources (no embedding probe was run on it).
