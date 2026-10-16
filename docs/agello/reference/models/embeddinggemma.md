# EmbeddingGemma 2 (text 270m)

> **Scope:** EmbeddingGemma 2, the embedding model behind Agello's search by meaning: the variants
> Google publishes; the text 270m model as trained (architecture, task prompts, pooling,
> normalization, Matryoshka dimensions), as exported (`embeddinggemma-2-text-270m.litertlm`: sections,
> graphs, signatures, quantization, metadata), as LiteRT-LM runs it, and as Agello uses it.
> **Out of scope:** the embedding executor's general mechanics ([../litert-lm/executors.md](../litert-lm/executors.md#embedding-executor)),
> the C and Swift embedding APIs ([../litert-lm/c-api.md](../litert-lm/c-api.md#embedding-engine),
> [../litert-lm/swift-api.md](../litert-lm/swift-api.md#embeddingengine)), the container format
> ([../litert-lm/file-format.md](../litert-lm/file-format.md)), CPU embedding lookups
> ([../litert-lm/components.md](../litert-lm/components.md#embedding-lookups)), the ML Drift delegate
> ([../litert/ml-drift-delegate.md](../litert/ml-drift-delegate.md)), the Gemma 4 chat models
> ([gemma4-family.md](gemma4-family.md)) and the tokenizer ([tokenizer-and-chat-format.md](tokenizer-and-chat-format.md)).
> **Pinned to:** LiteRT-LM `7b872544` (shipped wrapper `85b1a776`); model files and configs at the
> revisions in `tools/agello-docs/pins.json` (`google/embeddinggemma-2` `914f7f89`,
> `litert-community/embeddinggemma-2-text-270m-litert-lm` `9be6e8b9`); Agello `93122b647a`.
> **Applies to:** all platforms (model), Apple (Agello). **Last verified:** 2026-10-15

## Summary

- EmbeddingGemma 2 is Google DeepMind's multimodal embedding model: text, images, video and audio
  into one 768-dimensional space, 740M parameters in total, of which the text model is 270M. <!-- c:MOD-EMB-001 -->
  Agello uses only the text part, through litert-community's `embeddinggemma-2-text-270m`
  `.litertlm` file (164,626,432 bytes, INT4 weights), downloaded from the repository's `main` branch:
  Agello pins no revision for it, and `main` was `9be6e8b9`, the revision documented here, on
  2026-10-15. <!-- c:MOD-EMB-002 -->
- The text model is a 24-layer bidirectional encoder (width 512, 4 query heads, 5 sliding layers to
  1 global), with per-layer embeddings computed from the input embeddings rather than looked up and
  a 512 → 768 projection of each token's final state; Sentence Transformers then mean-pools and
  normalizes. <!-- c:MOD-EMB-003 -->
- The file holds the token embedder (run on the CPU on Apple) and one text encoder graph with six
  signatures, 128 to 8,192 tokens; pooling and the projection are inside the graph, the L2
  normalization and Matryoshka truncation are done by LiteRT-LM's engine afterwards.
- Trained context 8,192 tokens; exported signatures up to 8,192; but the file's metadata sets
  `max_input_length: 1024`, so by default LiteRT-LM loads only the 128-1,024 signatures and rejects
  longer inputs. Agello keeps that default. <!-- c:MOD-EMB-004 -->
- Agello embeds passages as `title: {name} | text: {passage}` and queries as
  `task: search result | query: {query}`, keeps 256 of the 768 dimensions, and stores each vector as
  256 float16 values (512 bytes). <!-- c:MOD-EMB-005 -->
- Two things to know: on the GPU, LiteRT-LM runs the encoder with FP16 activations, which the model
  card warns against for the original checkpoint (not measured against an FP32 reference here); and
  the exported graphs' sliding-window masks differ from the Hugging Face implementation for inputs
  of 514 tokens or more, counting BOS and EOS (see [The attention masks](#the-attention-masks)).

## Concepts

| Term | Meaning here |
| --- | --- |
| Embedding | A fixed-length vector for a whole input; similar meanings give vectors with a high cosine similarity. |
| Task prompt | A short text prefix (`task: search result \| query: `, `title: none \| text: `) that tells the model what the vector is for. Text only. |
| Asymmetric / symmetric task | Retrieval uses one prefix for queries and another for documents; classification, clustering and similarity use the same prefix for every input. |
| Mean pooling | The average of the final token states over the real (unmasked) tokens. |
| Matryoshka (MRL) dimensions | Training makes the leading 128, 256 and 512 dimensions usable on their own; truncate, then L2-normalize again. |
| Signature | One fixed-length entry point of the exported text encoder (`encoder_1x128` ... `encoder_1x8192`). An input runs the smallest signature at least as long as its tokens; the rest is padding. |
| `max_input_length` | Metadata (or a setting) that limits which signatures LiteRT-LM loads; with 1,024, signatures over 1,024 tokens are not compiled. |
| Trained / exported / runtime | Three different limits: what the model was trained for (8,192 tokens), what the file contains (signatures to 8,192), what the engine loads (1,024 by default). |

### The variants Google publishes

| Repository | What | Size | Agello |
| --- | --- | --- | --- |
| `google/embeddinggemma-2` (`914f7f89`) | The full model (text, vision, audio), `model.safetensors` in bfloat16 | 1,488,915,288 bytes | — |
| `litert-community/embeddinggemma-2-text-270m-litert-lm` (`9be6e8b9`) | Text only, INT4 per-channel (QAT) for transformer and embeddings; also chip-specific accelerator builds named for Google Tensor G5/G6, Intel PTL, MediaTek MT6991/MT6993 and Qualcomm QCS8275/SM8550/SM8650/SM8750/SM8850 | 164,626,432 bytes (the CPU/GPU file) | Yes |
| `litert-community/embeddinggemma-2-text-vision-440m-litert-lm` (`e301f74d`) | Text and images; vision encoder INT8 per-channel | 387,710,976 bytes | — |
| `litert-community/embeddinggemma-2-740m-litert-lm` (`24d962e9`) | Text, images, video and audio; audio encoder mixed INT2/INT4/INT8 | 484,622,336 bytes | — |
| `google/embeddinggemma-300m` | The first EmbeddingGemma (Gemma 3 based, text only) | — | — |

<!-- c:MOD-EMB-006 --> The card of the full model lists the text model as a 130M "backbone" and a
140M "embedder", with 170M for the vision and 300M for the audio encoder, and offers loading only
the needed encoders (270M text only, 440M with images, 570M with audio, 740M with all). <!-- c:MOD-EMB-007 -->
The safetensors file is consistent with the card's 740M: the Hugging Face API counts 744,371,512
parameters, all bfloat16 (2 bytes each = 1,488,743,024 bytes, plus a 172,264-byte header).
<!-- c:MOD-EMB-008 -->
The litert-community card lists the text 270m file's supported text input sizes as 128, 256, 512,
1,024, 2,048 and 8,192 tokens and its output sizes as 128, 256, 512 and 768 values (512 to 3,072
bytes in float32). <!-- c:MOD-EMB-009 -->

The configuration of `google/embeddinggemma-2` keeps the Gemma 4 vision (16 layers, width 768) and
audio (12 layers, width 1,024) encoder configs and the Gemma 4 media token IDs (image 258,880,
audio 258,881, video 258,884); its `architectures` is `EmbeddingGemma2Model`
([../generated/hf-configs.md](../generated/hf-configs.md)). <!-- c:MOD-EMB-010 --> Only the text
model is covered below; Agello doesn't use the media encoders.

## How it works

### The model as trained

From `config.json` of `google/embeddinggemma-2` at `914f7f89`
([../generated/hf/google__embeddinggemma-2/](../generated/hf/google__embeddinggemma-2/config.json)),
`text_config`:

| Field | Value | Note |
| --- | --- | --- |
| `model_type` | `embedding_gemma2_text` | |
| `num_hidden_layers` | 24 | |
| `hidden_size` | 512 | the model width |
| `intermediate_size` | 2,048 | gated MLP, `gelu_pytorch_tanh` |
| `num_attention_heads` | 4 | |
| `num_key_value_heads`, `head_dim` | 2, 256 | sliding layers |
| `per_layer_config` 05, 11, 17, 23 | `head_dim` 512, `num_key_value_heads` 1 | the four full-attention layers |
| `layer_types` | 20 `sliding_attention`, 4 `full_attention` (every sixth layer) | |
| `sliding_window` | 512 | a radius: see below |
| `rope_parameters` | sliding θ = 10,000; full θ = 1,000,000 (both `default`) | |
| `hidden_size_per_layer_input` | 512 | per-layer embeddings, projection only |
| `embedding_dim` | 768 | the output size |
| `vocab_size` | 262,144 | |
| `max_position_embeddings` | 262,144 | not the usable context (8,192 per the card) |
| `rms_norm_eps` | 1e-6 | |
| `bos_token_id`, `eos_token_id`, `pad_token_id` | 2, 1, 0 | |

<!-- c:MOD-EMB-011 --> <!-- c:MOD-EMB-012 --> <!-- c:MOD-EMB-013 -->

Per layer:

| Layers | Type | Query heads × dim | KV heads × dim | Q width | K, V width | RoPE θ |
| --- | --- | --- | --- | --- | --- | --- |
| 0-4, 6-10, 12-16, 18-22 (20) | sliding | 4 × 256 | 2 × 256 | 1,024 | 512 | 10,000 |
| 5, 11, 17, 23 (4) | full | 4 × 512 | 1 × 512 | 2,048 | 512 | 1,000,000 |

The model card summarizes the same as "# KV-Heads (Local/Global) 2/1", "Local:Global 5:1",
"GQA/MQA", mean pooling and a 512 → 768 projection layer. <!-- c:MOD-EMB-014 -->

What the Hugging Face implementation (`transformers` `modular_embedding_gemma2.py` at `8741ef5a`)
adds to the config:

1. **Bidirectional attention.** Every layer is non-causal (`is_causal = False`), and there is no
   key-value cache between calls. <!-- c:MOD-EMB-015 -->
2. **Sliding window as a radius.** A sliding layer attends to every position with
   `abs(q_idx - kv_idx) <= sliding_window`, so 512 means 512 positions on each side; the docstring
   says this is half of the "1024-wide window" of Google's reference implementation. <!-- c:MOD-EMB-016 -->
   This is why the model card says "Sliding Window 1024 tokens" while the config says 512: the two
   describe the same window. <!-- c:MOD-EMB-017 -->
3. **Attention scale 1.** `scaling = 1.0`, with RMS norms on queries and keys and a scale-free RMS
   norm on values (as in Gemma 4). <!-- c:MOD-EMB-018 -->
4. **Projection-only per-layer embeddings.** There is no per-layer token table
   (`embed_tokens_per_layer` is deleted). A single linear map takes the scaled input embeddings to
   24 × 512 values per token (scaled by 512^-0.5, then each layer's 512-wide slice RMS-normed). In
   each layer, after attention and the MLP, a gated block (512 → 512, GELU, times that layer's slice,
   512 → 512, RMS norm) is added to the residual, and the layer output is then multiplied by the
   layer's `layer_scalar` buffer. <!-- c:MOD-EMB-019 -->
5. **Input scaling.** Token embeddings are multiplied by √512 ≈ 22.627 (22.625 after bfloat16 rounding). <!-- c:MOD-EMB-020 -->
6. **Output.** A final RMS norm, then `embedding_projection` (512 → 768, no bias) per token; the
   comment notes that projecting per token equals projecting after mean pooling. The
   Sentence Transformers modules then mean-pool (`1_Pooling`: `pooling_mode` `mean`,
   `include_prompt` true) and L2-normalize (`2_Normalize`). <!-- c:MOD-EMB-021 -->

Compared with the first EmbeddingGemma (arXiv 2509.20354v3): that model is built on Gemma 3 with
`n = 24` layers, width `d_M = 768`, and two projections after mean pooling (768 → 3,072 → 768). <!-- c:MOD-EMB-022 -->
EmbeddingGemma 2 halves the width to 512, adds per-layer embeddings and keeps one 512 → 768
projection. No EmbeddingGemma 2 paper was found on arXiv or Hugging Face Papers as of 2026-10-15.

#### Where the 270M comes from

Counting only the weight matrices (norm vectors add about 77 thousand values):

| Part | Formula | Values |
| --- | --- | --- |
| Token embeddings | 262,144 × 512 | 134,217,728 |
| Per-layer projection | 512 × (24 × 512) | 6,291,456 |
| One sliding layer | q 512×1,024 + k 512×512 + v 512×512 + o 1,024×512 + MLP 3 × 512×2,048 + per-layer block 2 × 512×512 | 5,242,880 |
| One full layer | q 512×2,048 + k 512×512 + v 512×512 + o 2,048×512 + MLP 3 × 512×2,048 + per-layer block 2 × 512×512 | 6,291,456 |
| 24 layers | 20 × 5,242,880 + 4 × 6,291,456 | 130,023,424 |
| Output projection | 512 × 768 | 393,216 |
| **Total** | | **270,925,824** |

<!-- c:MOD-EMB-023 --> The card's split, 130M "backbone" and 140M "embedder", matches this count if
the embedder is the token table plus the per-layer projection (134,217,728 + 6,291,456 = 140,509,184)
and the backbone the 24 layers plus the output projection (130,416,640). The card doesn't define the
split; this is our reading. <!-- c:MOD-EMB-024 -->

### Task prompts

EmbeddingGemma 2 was trained with short text prefixes; the card says omitting them still works but
reduces quality, that media inputs take no prefix, and that a document with a real title should be
`title: {title} | text: {content}` (`title: none` otherwise). <!-- c:MOD-EMB-025 -->
`config_sentence_transformers.json` names the prefixes (`default_prompt_name` is null, so nothing is
added unless the caller passes a `prompt_name` or a literal `prompt` string; similarity is cosine):
<!-- c:MOD-EMB-026 -->

| Prefix | Prompt names |
| --- | --- |
| `task: search result \| query: ` | `query`, `Retrieval-query`, `Retrieval`, `SearchQuery`, `Reranking`, `BitextMining` |
| `title: none \| text: ` | `document`, `Document`, `Retrieval-document` |
| `task: question answering \| query: ` | `QuestionAnswering` |
| `task: fact checking \| query: ` | `FactChecking` |
| `task: code retrieval \| query: ` | `CodeRetrieval`, `InstructionRetrieval` |
| `task: classification \| query: ` | `Classification`, `MultilabelClassification` |
| `task: clustering \| query: ` | `Clustering` |
| `task: sentence similarity \| query: ` | `STS`, `SentenceSimilarity`, `PairClassification`, `Summarization` |

<!-- c:MOD-EMB-027 --> The same two retrieval formats appear in the first EmbeddingGemma paper
(arXiv 2509.20354v3). LiteRT-LM adds no prefix itself: the caller writes it into the text.

### Matryoshka dimensions

Matryoshka Representation Learning (Kusupati et al., arXiv 2205.13147v4) trains one vector so that
its leading dimensions form usable smaller vectors, at no extra inference cost. <!-- c:MOD-EMB-028 -->
The first EmbeddingGemma applied it by splitting the contrastive and spread-out losses into losses on
overlapping leading sub-vectors, supporting 512, 256 and 128 dimensions besides 768. <!-- c:MOD-EMB-029 -->
EmbeddingGemma 2's card lists the same truncation dimensions (128, 256, 512; native 768) and gives
the rule: keep the leading dimensions, then L2-normalize again, and give queries and documents the
same dimension. <!-- c:MOD-EMB-030 -->

Google reports, for the full-precision checkpoint (card at `914f7f89`, as of 2026-10-15):

| Dimension | Storage vs 768 | MTEB multilingual v2 | MTEB English v2 | MTEB code v1 |
| --- | --- | --- | --- | --- |
| 768 | 1:1 | 61.36 | 68.46 | 78.68 |
| 512 | 1:1.5 | 61.17 | 68.41 | 77.24 |
| 256 | 1:3 | 60.41 | 67.78 | 76.18 |
| 128 | 1:6 | 57.89 | 65.68 | 71.41 |

<!-- c:MOD-EMB-031 --> The card calls quality "close to lossless" down to 256 dimensions and says 128
suits text-only work best. These are the full-precision model's numbers, not the INT4 file's.

### The model as exported

`embeddinggemma-2-text-270m.litertlm` at `9be6e8b9` ([../generated/models/embeddinggemma.md](../generated/models/embeddinggemma.md)):
container 1.7.0, created 2026-09-30, six sections. <!-- c:MOD-EMB-032 -->

| # | Section | Bytes | What |
| --- | --- | --- | --- |
| 0 | `EmbeddingMetadataProto` | 124 | the metadata below |
| 1 | `SP_Tokenizer` | 4,689,013 | SentencePiece model |
| 2 | `TFLiteModel` `tf_lite_embedder` | 3,149,080 | the token lookup graph |
| 3 | `TFLiteModel` `tf_lite_text_encoder` | 21,250,720 | the encoder graph (six signatures) |
| 4 | `TFLiteWeights` `tf_lite_embedder` | 67,108,864 | INT4 token table |
| 5 | `TFLiteWeights` `tf_lite_text_encoder` | 68,354,048 | INT4 encoder matrices |

The tokenizer section has the same offsets as in Agello's Gemma 4 E4B file (bytes 32,768 to
4,721,781), and its bytes are identical (sha256 `e594c8a9…` for both). <!-- c:MOD-EMB-033 --> No
section carries a `backend_constraint` or `prefer_activation_type` item: each has only `model_type`,
so the runtime's own defaults decide the backend's precision. <!-- c:MOD-EMB-034 -->

**The weights add up exactly.** Both weight sections are the trained matrices at 4 bits each, no
padding: <!-- c:MOD-EMB-035 -->

- embedder: 262,144 × 512 values × 0.5 bytes = 67,108,864 bytes (section 4);
- encoder: 136,708,096 values (130,023,424 in the layers + 6,291,456 per-layer projection + 393,216
  output projection) × 0.5 bytes = 68,354,048 bytes (section 5).

So the export keeps the per-layer projection inside the text encoder, not in a separate per-layer
embedder section: the file has none, and the encoder takes only `embeddings` and `input_mask`. <!-- c:MOD-EMB-036 -->

#### The embedder graph

`embedder_1x1` takes `token_ids INT32[1, 1]` and returns `embeddings FLOAT32[1, 1, 512]` in 8 ops:
a range check (`GREATER_EQUAL` 0, `LESS` 262,144, `SELECT_V2` to 0 when out of range), an INT4
`EMBEDDING_LOOKUP` on the [262,144, 512] table with one scale per row, and a `MUL` by
22.627416610717773 (√512 in float32). <!-- c:MOD-EMB-037 --> The table's 262,144 per-row scales
(float32) and zero points (int64) account for most of the 3.1 MB graph: 262,144 × (4 + 8) =
3,145,728 of its 3,149,080 bytes. <!-- c:MOD-EMB-038 -->

#### The text encoder graph

One graph, six signatures, shared weights:

| Signature | Subgraph | Inputs | Output | Ops | Window mask |
| --- | --- | --- | --- | --- | --- |
| `encoder_1x128` | 1 | `embeddings [1,128,512]`, `input_mask [1,128]` | `encodings [1,768]` | 1,011 | no |
| `encoder_1x256` | 3 | `[1,256,512]`, `[1,256]` | `[1,768]` | 1,011 | no |
| `encoder_1x512` | 4 | `[1,512,512]`, `[1,512]` | `[1,768]` | 1,011 | no |
| `encoder_1x1024` | 0 | `[1,1024,512]`, `[1,1024]` | `[1,768]` | 1,011 | no |
| `encoder_1x2048` | 2 | `[1,2048,512]`, `[1,2048]` | `[1,768]` | 1,031 | yes |
| `encoder_1x8192` | 5 | `[1,8192,512]`, `[1,8192]` | `[1,768]` | 1,031 | yes |

<!-- c:MOD-EMB-039 --> All inputs and outputs are FLOAT32.

The op counts follow the architecture one for one (signature `encoder_1x1024`): <!-- c:MOD-EMB-040 -->

| Op | Count | Accounted for by |
| --- | --- | --- |
| `FULLY_CONNECTED` (INT4 weights, per-channel) | 218 | 9 per layer (q, k, v, o, MLP gate, up, down, per-layer gate and projection) × 24 = 216, plus the per-layer projection and the output projection |
| `odml.rms_norm` composite | 194 | 8 per layer (input, post-attention, pre- and post-MLP, post-per-layer, query, key, value) × 24 = 192, plus the per-layer projection norm and the final norm |
| `odml.rope` composite | 48 | queries and keys × 24 layers: 40 with `max_timescale` 10,000 (sliding), 8 with 1,000,000 (full) |
| `BATCH_MATMUL` | 48 | scores and weighted values × 24 |
| `SOFTMAX` | 24 | one per layer |
| `GELU` | 48 | MLP gate and per-layer gate × 24 |
| `SUM` 2, `DIV` 1, `ABS` 1 | | masked pooling: Σ(states × mask) / (Σ mask + \|Σ mask\| + 7e-05) |

The FC weight shapes confirm the per-layer table: 20 [1,024, 512] query and 20 [512, 1,024] output
matrices (sliding), 52 [2,048, 512] (48 MLP gate/up + 4 full-layer query), 28 [512, 2,048] (24 MLP
down + 4 full-layer output), 96 [512, 512] (40 sliding and 8 full key/value + 48 per-layer gate and
projection), one [12,288, 512] per-layer projection and one [768, 512] output projection. <!-- c:MOD-EMB-041 -->

In every signature the pooler sums the final-norm states × `input_mask` over the sequence and
divides by Σ mask + |Σ mask| + 7e-05, that is twice the mask sum plus 7e-05, so the pooled [1, 512]
vector is half the masked mean. The output projection (`FULLY_CONNECTED` [768, 512], no bias) runs
after pooling, so the graph's [1, 768] output is not L2-normalized. <!-- c:MOD-EMB-042 -->

#### The attention masks

The scores of each layer are biased by a mask built from `input_mask`: every real token sees every
real token, padding is hidden with a bias of −45,852.8 (inside the FP16 range, whose largest finite
value is 65,504). In `encoder_1x128` to `encoder_1x1024` that is the only mask, on sliding and full
layers alike. <!-- c:MOD-EMB-043 --> In `encoder_1x2048` and `encoder_1x8192` the sliding layers'
mask is also multiplied by a window built from `i − j` (query row `i`, key column `j`) and the
constants 512 and 513: a key is visible when `0 ≤ i − j < 512` or `0 < j − i < 513`, that is 511
positions before the query, the query itself and 512 after: 1,024 keys. The full layers keep the
padding mask only. <!-- c:MOD-EMB-044 -->

| Sequence length (tokens, with BOS and EOS) | Hugging Face (`abs(i − j) ≤ 512`) | Exported graph |
| --- | --- | --- |
| ≤ 513 | all tokens | all tokens (same) |
| 514-1,024 | 1,025-key window on sliding layers | all tokens on every layer (signatures up to `encoder_1x1024` have no window) |
| > 1,024 (only when `max_input_length` is raised above the metadata's 1,024) | 1,025-key window | 1,024-key window (`i − j` from −512 to 511) |

So for sequences of 514 to 1,024 tokens the exported encoder doesn't compute what the Hugging Face
implementation does, and the `encoder_1x2048` and `encoder_1x8192` signatures differ from it by one
key position. <!-- c:MOD-EMB-045 --> Whether Google's reference
implementation matches the export or `transformers` was not settled (see [Open questions](#open-questions)).
Agello's passages are sized to stay under 512 tokens in English, where all three agree.

**Attention tensor sizes.** Each layer's softmax input holds 4n² values for a signature of n tokens:
[1, 2, 2n, n] on sliding layers (2 key-value groups × 2 queries each) and [1, 1, 4n, n] on full
layers. <!-- c:MOD-EMB-046 --> At 2 bytes (FP16):

| n | 4n² values | FP16 bytes |
| --- | --- | --- |
| 128 | 65,536 | 131,072 (128 KiB) |
| 512 | 1,048,576 | 2,097,152 (2 MiB) |
| 1,024 | 4,194,304 | 8,388,608 (8 MiB) |
| 2,048 | 16,777,216 | 33,554,432 (32 MiB) |
| 8,192 | 268,435,456 | 536,870,912 (512 MiB) |

These are tensor sizes in the graph, not measured allocations; they show why loading only the short
signatures is the default.

#### The metadata

Section 0 (`EmbeddingMetadata`): `embedding_model_type.embedding_gemma_v2` with the image tokens
`<|image>`/`<image|>`, patches 16 × 16, `max_num_patches` 1,260, pooling kernel 3, the audio tokens
`<|audio>`/`<audio|>`; BOS token ID 2; EOS token ID 1; a miniaudio front end (16 kHz mono, frame
320, hop 160, FFT 512, 128 mel bins to 8 kHz); `max_input_length: 1024`; no `min_input_length` and no
`min_runtime_version`. <!-- c:MOD-EMB-047 --> The media fields are unused by this text-only file.
LiteRT-LM's canonical metadata for the family, `models/embeddinggemma2/EmbeddingMetadataProto_vision140.pbtext`,
has the same values (plus some fields spelled out at their defaults) and `max_input_length: 1024` [`models/embeddinggemma2/EmbeddingMetadataProto_vision140.pbtext:56`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/models/embeddinggemma2/EmbeddingMetadataProto_vision140.pbtext#L56); its README describes it as the setting
for 140 visual tokens (1,260 patches). <!-- c:MOD-EMB-048 -->

### How LiteRT-LM runs it

```mermaid
sequenceDiagram
  participant App as Swift EmbeddingEngine
  participant Eng as EmbeddingEngineImpl
  participant Exe as EmbeddingLiteRtCompiledModelExecutor
  participant CPU as Embedder (CPU)
  participant Enc as Text encoder (GPU or CPU)
  App->>Eng: create(settings: backend, cacheDir, maxInputLength?)
  Eng->>Eng: ResolveDefaults: max_input_length 1024 from metadata; GPU -> FLOAT16
  Eng->>Exe: compile encoder_1x128..1x1024 only
  App->>Eng: computeEmbedding([text], normalize, outputSize)
  Eng->>Eng: [BOS 2] + SentencePiece(text) + [EOS 1]
  Eng->>Exe: ComputeEmbedding(tokens, kError)
  Exe->>CPU: one embedder run per token -> [n, 512]
  Exe->>Exe: input_mask 1.0 for tokens, 0.0 for padding
  Exe->>Enc: Run(smallest signature >= n)
  Enc-->>Exe: encodings [1, 768]
  Exe-->>Eng: vector (unnormalized)
  Eng->>Eng: truncate to outputSize, then L2-normalize
  Eng-->>App: [Float] of outputSize
```

1. **Settings.** `EmbeddingEngineSettings::ResolveDefaults` takes the file's metadata unless the
   caller supplied some, and copies its `min_input_length` / `max_input_length` into the settings
   when the caller set none [`runtime/engine/embedding_engine_settings.cc:266-298`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/embedding_engine_settings.cc#L266-L298). The text encoder's activation type comes from the caller,
   else from the section's `prefer_activation_type`, else FLOAT16 when the backend is the GPU
   [`runtime/engine/embedding_engine_settings.cc:183-208`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/embedding_engine_settings.cc#L183-L208); `SetCommonGpuOptions` turns any activation type but FLOAT32 (and no mixed precision)
   into GPU precision `kFp16` [`runtime/executor/litert_compiled_model_executor_utils.cc:1017-1031`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/litert_compiled_model_executor_utils.cc#L1017-L1031). For this file on the GPU that means FP16 activations. <!-- c:MOD-EMB-049 -->
2. **Which signatures load.** With either bound set, `SelectTextEncoderSignatures` keeps the
   signatures sorted by length from the smallest up to the first one at least `max_input_length`
   long (skipping any shorter than `min_input_length`) [`runtime/core/embedding_engine_impl.cc:334-352`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/embedding_engine_impl.cc#L334-L352) [`runtime/executor/model_signature_utils.cc:202-236`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/model_signature_utils.cc#L202-L236), and the executor
   compiles the encoder with only those selected [`runtime/executor/embedding_litert_compiled_model_executor.cc:242-252`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/embedding_litert_compiled_model_executor.cc#L242-L252). With the file's 1,024, that is
   `encoder_1x128`, `1x256`, `1x512` and `1x1024`; a `max_input_length` above 8,192 fails engine
   creation ("exceeds maximum available signature length"). <!-- c:MOD-EMB-050 --> Input and output
   buffers are created for every loaded signature at compile time [`runtime/executor/embedding_litert_compiled_model_executor.cc:273-307`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/embedding_litert_compiled_model_executor.cc#L273-L307).
3. **Tokens.** With `insert_special_tokens` (default true) the engine puts the BOS IDs before the
   contents and the EOS IDs after them [`runtime/core/embedding_engine_impl.cc:1260-1318`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/embedding_engine_impl.cc#L1260-L1318); text is tokenized with the file's
   SentencePiece model, which adds no special tokens of its own [`runtime/core/embedding_engine_impl.cc:1346-1369`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/embedding_engine_impl.cc#L1346-L1369) [`support/tokenizer/sentencepiece_tokenizer.cc:65-73`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/support/tokenizer/sentencepiece_tokenizer.cc#L65-L73). A text input
   of t tokens therefore becomes t + 2 tokens. <!-- c:MOD-EMB-051 -->
4. **Lookups.** The executor creates the token lookup from the embedder section (and a per-layer
   lookup only if the file has a per-layer embedder section, which this one doesn't) [`runtime/executor/embedding_litert_compiled_model_executor.cc:326-333`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/embedding_litert_compiled_model_executor.cc#L326-L333);
   the lookup runs on the CPU on Apple ([../litert-lm/components.md](../litert-lm/components.md#embedding-lookups)).
   It fills the `embeddings` input, padding rows included (with the embedding of token 0), and the
   `input_mask` with 1.0 for the real tokens and 0.0 after them [`runtime/executor/embedding_litert_compiled_model_executor.cc:412-431`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/embedding_litert_compiled_model_executor.cc#L412-L431) [`runtime/executor/embedding_litert_compiled_model_executor.cc:433-452`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/embedding_litert_compiled_model_executor.cc#L433-L452); the per-layer
   input is filled only when the signature has an input named `per_layer` [`runtime/executor/embedding_litert_compiled_model_executor.cc:454-460`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/embedding_litert_compiled_model_executor.cc#L454-L460). <!-- c:MOD-EMB-052 -->
5. **Encoder.** It runs the smallest loaded signature at least as long as the tokens and copies the
   [1, 768] output [`runtime/executor/embedding_litert_compiled_model_executor.cc:383-389`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/embedding_litert_compiled_model_executor.cc#L383-L389) [`runtime/executor/embedding_litert_compiled_model_executor.cc:462-480`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/embedding_litert_compiled_model_executor.cc#L462-L480). Longer inputs follow the overflow strategy: `kError` (the
   default) returns InvalidArgument "Input sequence length (N) exceeds maximum supported signature
   length (M)"; `kTruncate` cuts to M tokens, overwriting the end with the EOS IDs when special
   tokens are on; `kChunkAndAverage` embeds consecutive M-token chunks and averages the raw
   (unnormalized) outputs [`runtime/executor/embedding_litert_compiled_model_executor.cc:520-549`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/embedding_litert_compiled_model_executor.cc#L520-L549) [`runtime/executor/embedding_litert_compiled_model_executor.cc:550-583`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/embedding_litert_compiled_model_executor.cc#L550-L583) [`runtime/executor/embedding/embedding_executor_base.h:31-52`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/embedding/embedding_executor_base.h#L31-L52). <!-- c:MOD-EMB-053 --> M is the
   longest *loaded* signature, so 1,024 by default, not 8,192.
6. **Output.** The engine truncates the vector to `output_size` when set (an error if negative or
   above 768), then L2-normalizes it when `normalize` is on (the default), so a truncated vector
   comes back unit length, as the Matryoshka rule requires [`runtime/core/embedding_engine_impl.cc:1589-1624`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/embedding_engine_impl.cc#L1589-L1624) [`runtime/core/embedding_engine_impl.cc:245-262`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/embedding_engine_impl.cc#L245-L262) [`runtime/engine/embedding_engine.h:31-51`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/embedding_engine.h#L31-L51). <!-- c:MOD-EMB-054 -->
7. **Batches.** `ComputeEmbeddingBatch` loops over the requests one by one, in the engine and in the
   executor ("Batch inference isn't supported yet"); a batch saves call overhead, not GPU work
   [`runtime/core/embedding_engine_impl.cc:1655-1694`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/embedding_engine_impl.cc#L1655-L1694) [`runtime/executor/embedding_litert_compiled_model_executor.cc:598-610`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/embedding_litert_compiled_model_executor.cc#L598-L610). <!-- c:MOD-EMB-055 -->

The shipped v0.18.0 sources have the same defaults: metadata input-length bounds, FLOAT16 on the GPU,
and the `kError` overflow strategy [`runtime/engine/embedding_engine_settings.cc:266-298`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/runtime/engine/embedding_engine_settings.cc#L266-L298) [`runtime/engine/embedding_engine_settings.cc:183-208`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/runtime/engine/embedding_engine_settings.cc#L183-L208) [`runtime/engine/embedding_engine.h:31-51`](https://github.com/google-ai-edge/LiteRT-LM/blob/b2f686e2ed4718fb84ec398a61dd59ca0f0aff27/runtime/engine/embedding_engine.h#L31-L51). <!-- c:MOD-EMB-056 -->

What the Swift API exposes is on [../litert-lm/swift-api.md](../litert-lm/swift-api.md#embeddingengine):
`EmbeddingEngineConfig` has `maxInputLength` (passed to the C setter only when non-nil) but no
activation type, and `EmbeddingOptions` has no overflow strategy [`swift/EmbeddingEngineConfig.swift:42-58`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/EmbeddingEngineConfig.swift#L42-L58) [`swift/EmbeddingEngine.swift:102-107`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/EmbeddingEngine.swift#L102-L107) [`swift/EmbeddingEngineConfig.swift:62-94`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/EmbeddingEngineConfig.swift#L62-L94);
the C API has a setter for the activation type that the Swift wrapper never calls [`c/embedding_engine.h:240-254`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/c/embedding_engine.h#L240-L254). <!-- c:MOD-EMB-057 -->
The Swift `EmbeddingResponse` carries only the vector, not the input length, truncation or chunk
count of the C++ response [`swift/EmbeddingEngineConfig.swift:96-105`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/EmbeddingEngineConfig.swift#L96-L105) [`runtime/engine/embedding_engine.h:53-63`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/embedding_engine.h#L53-L63).

### How Agello uses it

Agello `93122b647a` (`Agello/Knowledge/`, `Agello/ModelHub/ModelCatalog.swift`):

| What | Agello | Where |
| --- | --- | --- |
| File | `litert-community/embeddinggemma-2-text-270m-litert-lm`, `embeddinggemma-2-text-270m.litertlm`, expected size 164,626,432 bytes, revision `main` (not a commit) | `ModelCatalog.swift:44-53`, `:13` |
| Engine | `EmbeddingEngine` with `EmbeddingEngineConfig(modelPath:backend:cacheDir:)`: tries `.gpu`, then `.cpu()`; loads on first use | `Embedding.swift:76-94` |
| Options | `EmbeddingOptions(normalize: true, outputSize: 256)`; special tokens and overflow strategy left at their defaults | `Embedding.swift:23`, `:33` |
| Documents | `title: {document name, or none} \| text: {passage}`, in `computeEmbeddingBatch` calls of 8 | `Embedding.swift:41-53`, `:67-70` |
| Queries | `task: search result \| query: {query}` | `Embedding.swift:55-60`, `:72-74` |
| Index identifier | `embeddinggemma-2-text-270m@256`: a document indexed with a different identifier is re-indexed; chat search vectors store no identifier and are not | `Embedding.swift:24` |
| Storage | 256 float16 values per vector (`VectorIndex.encode`), decoded to float32 for search | `Embedding.swift:137-149` |
| Search | brute-force dot product with `vDSP_dotpr` over unit vectors, sorted | `Embedding.swift:151-173` |
| Passages | about 1,600 characters with 200 characters of overlap ("~400 tokens of English at ~4 characters per token") | `Chunker.swift:12-19` |
| Chats | title and first 6 turns, at most 1,500 characters, embedded with `title: none` | `KnowledgeBase.swift:171-181`, `:195-212` |
| Web passages | embedded with title `none` and ranked against the question | `Web/WebContext.swift:270-275` |
| Related chats | cosine ≥ 0.60 and ≥ 0.08 above the median chat (`sorted[count / 2]`, 3 or more chats); thresholds commented as calibrated on device for the 256-dimension embedder | `Embedding.swift:25-29`, `KnowledgeBase.swift:226-236` |
| Engine cache | `Library/Caches/com.patricedery.agello.litertlm/embedding` (`embedding-<build id>` for builds that report one); the current build's folder is kept when caches are trimmed | `EngineCache.swift:26-38`, `:115-121` |

<!-- c:MOD-EMB-058 --> <!-- c:MOD-EMB-059 --> <!-- c:MOD-EMB-060 --> <!-- c:MOD-EMB-061 -->

**Matches the card.** The two prompt formats are the card's retrieval pair (`SearchQuery` /
`Document` with a title), truncation to 256 is one of the trained Matryoshka dimensions, and the
engine re-normalizes after truncating. Chat titles are not passed as the title field: chats use
`title: none` with the title inside the text. <!-- c:MOD-EMB-062 -->

**Storage arithmetic.** Google's full output is 768 float32 values, 3,072 bytes; Agello keeps 256
values as float16, 512 bytes: a sixth of the 768-dimension float32 size by dimension count alone
(3,072 / 1,024 = 3 for the truncation, × 2 for float16). <!-- c:MOD-EMB-063 --> The float16
rounding of unit-length components moves a cosine score by at most about 2⁻¹¹ ≈ 0.0005 (each
normal float16 value has a relative rounding error ≤ 2⁻¹¹, and Σ|qᵢ dᵢ| ≤ 1 for unit vectors),
far below the 0.08 margin Agello uses for related chats. <!-- c:MOD-EMB-064 -->

**Token budget.** Agello sets no `maxInputLength`, so the file's 1,024 applies: a passage plus its
prompt must stay under 1,023 SentencePiece tokens of text (1,024 with BOS and EOS). The chunker
counts characters, not tokens. <!-- c:MOD-EMB-065 --> At its own estimate of 4 characters per token a
1,600-character passage is about 400 tokens; it would pass 1,022 tokens only at under about 1.6
characters per token (1,600 / 1,022 ≈ 1.57), which dense scripts or code can approach, and a long
file name adds to the title field. The token count of real passages wasn't measured.

**What a failure does.** `KnowledgeBase.index` embeds all of a document's passages in one call
chain; if any batch throws (an over-long passage, for one), the document stays unindexed and the
error is logged; because `search` uses vectors only when every document in scope is indexed with the
current embedder, a single failed document sends that search to the keyword (BM25) ranking, and
each search retries the indexing. <!-- c:MOD-EMB-066 --> A query that fails to embed also falls back
to keywords.

**Measured (Agello probe 3, `ProbeScenarioTests.test03_EmbeddingGemma`).** The probe loads the
file on the GPU with Agello's cache directory, embeds four short passages as
`title: none | text: …` and one query as `task: search result | query: How tall is the tallest
mountain?` at 256 dimensions, and times one filler passage with the `title: none | text: ` prefix:
`CalibrationScenarioTests.filler(bytes: 1_800)`, which comes to 1,827 bytes (at least 1,800; about
400 tokens per the test's comment). <!-- c:MOD-EMB-067 -->

| Device, build | Load | Short passage | ~400-token passage | Scores (Everest, Python, Paris, lighthouse) |
| --- | --- | --- | --- | --- |
| iPhone 17 Pro (A19 Pro), ours `2e54fbdb9ca9` | 0.62 s | 16.8 ms | 57.4 ms | 0.811, 0.639, 0.584, 0.620 |
| Same phone, Apple9 kernel paths simulated, ours `d7fafdca9b7c` | 1.57 s | 29.4 ms | 91.4 ms | 0.811, 0.639, 0.584, 0.619 |
| Mac mini M6, ours `c5460a024e71` | 0.64 s | 11.9 ms | 27.5 ms | 0.811, 0.639, 0.584, 0.620 |

<!-- c:MOD-EMB-068 --> <!-- c:MOD-EMB-069 --> The right passage ranks first everywhere and the
scores agree within 0.001 across the three GPU paths. The footprint the probe records
(1.41-1.50 GB) is the whole test process after loading, not the embedder alone. On the same phone
the embedder's cache folder held 0.18 GB (179,974,496 bytes) in probe 14. <!-- c:MOD-EMB-070 -->
Google's card reports 11.6 ms on the GPU and 41.8 ms on the CPU for the 128-token signature on an
iPhone 18 Pro, with 85 MB of memory on the GPU (as of 2026-10-15). <!-- c:MOD-EMB-071 -->

## On Apple

- **Embedder on the CPU, encoder on the GPU.** LiteRT-LM compiles the token embedder with the CPU
  accelerator on every platform except Android [`runtime/components/embedding_lookup/embedding_lookup_text.cc:292-298`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/embedding_lookup/embedding_lookup_text.cc#L292-L298), so on iOS and macOS each token's
  embedding is one CPU model run; the text encoder runs on the backend the engine was created with,
  the Metal GPU through the ML Drift delegate when `.gpu` succeeds [`runtime/executor/embedding_litert_compiled_model_executor.cc:182-216`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/embedding_litert_compiled_model_executor.cc#L182-L216). <!-- c:MOD-EMB-072 -->
- **FP16 on the GPU.** With no activation type from the caller or the file, the GPU encoder runs in
  FP16 (step 1 above). Google's card for the original checkpoint says: run in bfloat16 or float32,
  not float16, whose range the activations exceed (NaN or silently degraded vectors). <!-- c:MOD-EMB-073 -->
  The exported graph is not the PyTorch checkpoint (it is INT4 QAT, and its mask bias is chosen to
  fit FP16), and Agello's probe gives the expected ranking with consistent scores on two GPU
  families, but no comparison against FP32 output was made. The CPU backend's precision was not
  examined.
- **Rotary positions.** The encoder's RoPE is the `odml.rope` composite, which the ML Drift delegate
  maps to its own RoPE kernel ([../litert/ml-drift-delegate.md](../litert/ml-drift-delegate.md#the-op-selector));
  that kernel computes angles in FP32 ([../ml-drift/kernels.md](../ml-drift/kernels.md#rotary-positions-rope)),
  unlike E4B's elementwise sine and cosine behind LiteRT-LM #3913. Positions here stay ≤ 1,023 at
  the default anyway.
- **Caches.** With a cache directory, the GPU path writes a program cache and a weight cache there
  [`runtime/executor/embedding_litert_compiled_model_executor.cc:182-216`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/embedding_litert_compiled_model_executor.cc#L182-L216); Agello gives them a folder of their own, which its cache trimming keeps (it deletes only other builds' folders).
- **Memory.** The two INT4 weight sections are 135.5 MB in the file (67,108,864 + 68,354,048
  bytes). <!-- c:MOD-EMB-074 --> How much of that the GPU path keeps resident, and the activation
  memory per signature, were not measured in isolation; see [../memory.md](../memory.md).

## Settings

Rows for [../configuration.md](../configuration.md):

| Setting | Layer | Default | Effect | Agello |
| --- | --- | --- | --- | --- |
| `EmbeddingEngineConfig.backend` | Swift | `.cpu()` | backend of the text encoder | `.gpu`, then `.cpu()` |
| `EmbeddingEngineConfig.cacheDir` | Swift | the model's folder | program and weight caches | its own `embedding` folder |
| `EmbeddingEngineConfig.maxInputLength` | Swift → C → settings | nil: the file's `max_input_length` (1,024) | which signatures load, so the longest input | not set |
| `min_input_length` | C / settings, metadata | unset | drops signatures shorter than it | — |
| activation data type | C / settings | GPU: FLOAT16; else unset | GPU precision | not settable from Swift |
| `EmbeddingOptions.normalize` | Swift / C / C++ | true | L2 normalization after truncation | true |
| `EmbeddingOptions.outputSize` | Swift / C / C++ | none (768) | Matryoshka truncation | 256 |
| `EmbeddingOptions.insertSpecialTokens` | Swift / C / C++ | true | BOS 2 and EOS 1 around the input | default |
| `input_overflow_strategy` | C / C++ | `kError` | longer inputs: error, truncate, or chunk and average | not settable from Swift |

## Limits and failure modes

| Situation | What happens | Evidence |
| --- | --- | --- |
| Input over 1,024 tokens (with BOS/EOS) at the default | InvalidArgument from the executor; in Agello the document isn't indexed and search falls back to keywords | steps 2 and 5; `KnowledgeBase.swift:114-142`, `:261-293` |
| `maxInputLength` set above 8,192 | engine creation fails | [`runtime/executor/model_signature_utils.cc:202-236`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/model_signature_utils.cc#L202-L236) |
| `outputSize` above 768 or negative | InvalidArgument | [`runtime/core/embedding_engine_impl.cc:1589-1624`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/embedding_engine_impl.cc#L1589-L1624) |
| Chunk-and-average on long inputs (C/C++ only) | the average of unnormalized chunk vectors, normalized afterwards; chunks are hard cuts, BOS only on the first and EOS only on the last | [`runtime/executor/embedding_litert_compiled_model_executor.cc:550-583`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/embedding_litert_compiled_model_executor.cc#L550-L583) <!-- c:MOD-EMB-075 --> |
| Inputs of 514-1,024 tokens (with BOS and EOS) | sliding layers see the whole input, unlike the `transformers` implementation | [The attention masks](#the-attention-masks) |
| Changing the dimension or the model | vectors aren't comparable across dimensions; Agello's identifier includes `@256`, so a change re-indexes documents (chat search vectors carry no identifier); its 0.60/0.08 thresholds are commented as calibrated on device for 256 dimensions | `Embedding.swift:20-29` <!-- c:MOD-EMB-076 --> |
| Model file at `main` changes on Hugging Face | Agello's download expects 164,626,432 bytes; a file of a different size fails `ModelStore`'s size check; a same-size replacement would be accepted, since only the size is checked. Probe 3 and the `KnowledgeScenarioTests` threshold calibration don't record the model revision they ran with. | `ModelCatalog.swift:13`, `:44-53` <!-- c:MOD-EMB-058 --> <!-- c:MOD-EMB-002 --> <!-- c:MOD-EMB-068 --> <!-- c:MOD-EMB-076 --> |
| Prompts omitted | still works, lower quality (card) | card |

## Where in the code

| Concept | Symbol | Citation |
| --- | --- | --- |
| Options and defaults | `EmbeddingOptions` | [`runtime/engine/embedding_engine.h:31-51`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/embedding_engine.h#L31-L51) |
| Overflow strategies | `InputOverflowStrategy`, `ComputeEmbeddingOptions` | [`runtime/executor/embedding/embedding_executor_base.h:31-52`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/embedding/embedding_executor_base.h#L31-L52) |
| Metadata input-length bounds | `EmbeddingMetadata.min_input_length`, `max_input_length` | [`runtime/proto/embedding_metadata.proto:114-124`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/proto/embedding_metadata.proto#L114-L124) |
| EmbeddingGemma 2 model type | `EmbeddingGemmaV2` | [`runtime/proto/embedding_model_type.proto:38-69`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/proto/embedding_model_type.proto#L38-L69) |
| Defaults from metadata, activation type | `EmbeddingEngineSettings::ResolveDefaults`, `ResolveActivationDataType` | [`runtime/engine/embedding_engine_settings.cc:266-298`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/embedding_engine_settings.cc#L266-L298) [`runtime/engine/embedding_engine_settings.cc:183-208`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/embedding_engine_settings.cc#L183-L208) |
| Signature selection | `SetTextEncoderSignaturesFromSettings`, `SelectSignaturesByCapacity` | [`runtime/core/embedding_engine_impl.cc:334-352`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/embedding_engine_impl.cc#L334-L352) [`runtime/executor/model_signature_utils.cc:202-236`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/model_signature_utils.cc#L202-L236) |
| Encoder compile per backend | `CompileTextEncoder` | [`runtime/executor/embedding_litert_compiled_model_executor.cc:182-216`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/embedding_litert_compiled_model_executor.cc#L182-L216) |
| Encoder signature map | `GetTextEncoderSignatureMap` | [`runtime/executor/embedding_litert_compiled_model_executor.cc:111-168`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/embedding_litert_compiled_model_executor.cc#L111-L168) |
| One run | `RunEncoderForTokens` | [`runtime/executor/embedding_litert_compiled_model_executor.cc:383-389`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/embedding_litert_compiled_model_executor.cc#L383-L389) [`runtime/executor/embedding_litert_compiled_model_executor.cc:462-480`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/embedding_litert_compiled_model_executor.cc#L462-L480) |
| Overflow handling | `ComputeEmbedding` | [`runtime/executor/embedding_litert_compiled_model_executor.cc:520-549`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/embedding_litert_compiled_model_executor.cc#L520-L549) [`runtime/executor/embedding_litert_compiled_model_executor.cc:550-583`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/embedding_litert_compiled_model_executor.cc#L550-L583) |
| Special tokens | `EmbeddingEngineImpl::InsertSpecialTokens` | [`runtime/core/embedding_engine_impl.cc:1260-1318`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/embedding_engine_impl.cc#L1260-L1318) |
| Truncation and normalization | `ComputeEmbeddingInternal`, `L2Norm` | [`runtime/core/embedding_engine_impl.cc:1589-1624`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/embedding_engine_impl.cc#L1589-L1624) [`runtime/core/embedding_engine_impl.cc:245-262`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/embedding_engine_impl.cc#L245-L262) |
| GPU precision | `SetCommonGpuOptions` | [`runtime/executor/litert_compiled_model_executor_utils.cc:1017-1031`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/litert_compiled_model_executor_utils.cc#L1017-L1031) |
| Section hints | `LitertLmLoader` section items | [`runtime/util/litert_lm_loader.cc:164-176`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.cc#L164-L176) |
| Swift configuration | `EmbeddingEngineConfig`, `EmbeddingOptions` | [`swift/EmbeddingEngineConfig.swift:42-58`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/EmbeddingEngineConfig.swift#L42-L58) [`swift/EmbeddingEngineConfig.swift:62-94`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/EmbeddingEngineConfig.swift#L62-L94) |
| Swift initialization | `EmbeddingEngine.initialize` | [`swift/EmbeddingEngine.swift:67-123`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/EmbeddingEngine.swift#L67-L123) |
| C activation setter | `litert_lm_embedding_engine_settings_set_activation_data_type` | [`c/embedding_engine.h:240-254`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/c/embedding_engine.h#L240-L254) |
| C input-length setters | `litert_lm_embedding_engine_settings_set_max_input_length` / `_min_` | [`c/embedding_engine.h:196-222`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/c/embedding_engine.h#L196-L222) |
| Agello embedder | `LiteRTEmbedder`, `VectorIndex` | Agello `Agello/Knowledge/Embedding.swift` at `93122b647a` |
| Agello indexing and search | `KnowledgeBase.index`, `.search`, `.indexChats` | Agello `Agello/Knowledge/KnowledgeBase.swift` at `93122b647a` |
| Agello passages | `Chunker` | Agello `Agello/Knowledge/Chunker.swift` at `93122b647a` |

## Open questions

- **Sliding window in the export.** The `encoder_1x128`-`1x1024` graphs have no window and the
  longer ones use offsets −512 to +511, while `transformers` uses ±512. Which matches Google's
  reference implementation (the card's "1024 tokens"), and does it matter for 514-1,024-token
  sequences? Not verified; a comparison against `transformers` output would settle it.
- **FP16 on the GPU.** Does the GPU path's FP16 encoder lose quality compared with FP32 (the CPU, or
  `litert_lm_embedding_engine_settings_set_activation_data_type` with FLOAT32)? Not measured.
- **Tokens per passage.** How many tokens Agello's 1,600-character passages actually take, by
  language, and whether any real document exceeds 1,022. Not measured.
- **BOS/EOS in the reference.** Whether the Sentence Transformers pipeline also wraps the input in
  `<bos>` … `<eos>` as LiteRT-LM does was not checked (the tokenizer config doesn't say).
- **The 130M / 140M split.** The card doesn't define it; the reading above is ours.
- **No EmbeddingGemma 2 paper.** Training details (Matryoshka losses, distillation, data) are known
  only for the first model (arXiv 2509.20354v3).
- **Shipped binary.** The defaults above come from the v0.18.0 sources; Agello's probe ran on our
  builds only, not on Google's prebuilt binary.
