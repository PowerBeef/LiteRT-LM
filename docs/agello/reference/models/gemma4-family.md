# Gemma 4: the model family

> **Scope:** the Gemma 4 models as Google publishes them: E2B, E4B, 12B, 26B-A4B (mixture of experts),
> 31B and the "assistant" drafters; base and instruction-tuned checkpoints; the design they share
> (local and global attention, head sizes, RoPE, per-layer embeddings, KV sharing); modalities;
> trained context; license; which ones have LiteRT-LM files and which could run on a phone.
> **Out of scope:** E4B and E2B in depth, including their `.litertlm` exports
> ([gemma4-e4b.md](gemma4-e4b.md), [gemma4-e2b.md](gemma4-e2b.md)); the tokenizer and chat format
> ([tokenizer-and-chat-format.md](tokenizer-and-chat-format.md)); EmbeddingGemma
> ([embeddinggemma.md](embeddinggemma.md)); how the runtime loads a file
> ([../litert-lm/file-format.md](../litert-lm/file-format.md)); memory on the device
> ([../memory.md](../memory.md)).
> **Pinned to:** Hugging Face revisions in `tools/agello-docs/pins.json` (`generated/hf/*/SOURCE.txt`);
> LiteRT-LM `7b872544` for the few code facts.
> **Applies to:** all platforms (model facts); phone estimates are for Apple. **Last verified:** 2026-10-15

## Summary

Gemma 4 is Google DeepMind's open-weight model family: five instruction-tuned sizes (E2B,
E4B, 12B, 26B-A4B, 31B), each with a pre-trained base, plus small "assistant" models that are
multi-token-prediction drafters for speculative decoding. All five share one tokenizer (262,144
entries), a 5:1 or 4:1 mix of sliding-window and global attention with the last layer global,
256-wide local heads and 512-wide global heads, and two RoPE settings. E2B and E4B are the on-device
models: per-layer embeddings, KV sharing across their last layers, a 128K trained context, and
vision and audio encoders. 12B is encoder-free; 26B-A4B and 31B have no audio. Everything is Apache
2.0. Only E2B and E4B have `.litertlm` files that fit an iPhone with Agello's build; Agello ships
E4B. 26B-A4B and 31B exist on `litert-community` only as `-gpu` and `-web` files; if those are
"artisan" files like E2B's and E4B's `-gpu` files, the documented build can't run them. 12B's 6.9 GB
file is close to the iPhone 17 Pro's app limit, which is 8 GiB (8,589,934,592 bytes, printed
"8.6 GB"; an early reading can show 6 GiB). <!-- c:APL-001 --> <!-- c:APL-004 -->

## Concepts

| Term | Meaning here |
| --- | --- |
| **E2B, E4B** | "Effective" 2B and 4B: parameter counts that leave out the large per-layer embedding tables, which are only looked up |
| **A4B** | "Active" 4B: the card gives 3.8B active of 25.2B total parameters (MOD-FAM-057); the checkpoint holds 25,805,936,206 BF16 parameters (MOD-FAM-021) |
| **Unified** | 12B: no separate vision or audio encoder; image patches and raw audio chunks are projected straight into the decoder |
| **Assistant** | a 4-layer drafter for one target model, used for speculative decoding (Google calls it the MTP drafter) |
| **Local / global layer** | `sliding_attention` (attends to the last `sliding_window` tokens) / `full_attention` (attends to everything) in `layer_types` |
| **Per-layer embeddings (PLE)** | a second token embedding, `hidden_size_per_layer_input` wide per decoder layer, fed to every layer (E models only) |
| **KV sharing** | the last `num_kv_shared_layers` layers reuse key/value caches of earlier layers and keep none of their own |
| **p-RoPE** | proportional RoPE on global layers: rotary angles on a fraction (`partial_rotary_factor` 0.25) of each head |
| **Trained context** | `max_position_embeddings` in the config; distinct from a file's export window and from what the runtime allocates |

Three different things are described on this page and kept apart: the model **as trained** (config,
card, technical report), the model **as exported** (a `.litertlm` file), and what a runtime
**allocates**. Model-card and report facts come from Google and are labeled as such.

## How it works

### The family at a glance

From the official `config.json` files at the pinned revisions ([hf-configs](../generated/hf-configs.md)):

| | E2B | E4B | 12B | 26B-A4B | 31B |
| --- | --- | --- | --- | --- | --- |
| Hugging Face repo (instruction-tuned) | `google/gemma-4-E2B-it` | `google/gemma-4-E4B-it` | `google/gemma-4-12B-it` | `google/gemma-4-26B-A4B-it` | `google/gemma-4-31B-it` |
| Revision | `3e22461f` | `ee0ef602` | `707f0a3b` | `4d7ae498` | `842da379` |
| `architectures` | `Gemma4ForConditionalGeneration` | same | `Gemma4UnifiedForConditionalGeneration` | `Gemma4ForConditionalGeneration` | same |
| `model_type` | `gemma4` | `gemma4` | `gemma4_unified` | `gemma4` | `gemma4` |
| Decoder layers | 35 | 42 | 48 | 30 | 60 |
| `hidden_size` | 1,536 | 2,560 | 3,840 | 2,816 | 5,376 |
| `intermediate_size` | 6,144 (double-wide MLP) | 10,240 | 15,360 | 2,112 (plus experts) | 21,504 |
| Query heads | 8 | 8 | 16 | 16 | 32 |
| KV heads, local layers | 1 | 2 | 8 | 8 | 16 |
| `num_global_key_value_heads` | unset | unset | 1 | 2 | 4 |
| Global layers share K and V (`attention_k_eq_v`) | no | no | yes | yes | yes |
| `sliding_window` | 512 | 512 | 1,024 | 1,024 | 1,024 |
| Global layers (count, indices) | 7: 4, 9, …, 34 | 7: 5, 11, …, 41 | 8: 5, 11, …, 47 | 5: 5, 11, …, 29 | 10: 5, 11, …, 59 |
| `num_kv_shared_layers` | 20 | 18 | 0 | 0 | 0 |
| `hidden_size_per_layer_input` (PLE) | 256 | 256 | 0 | 0 | 0 |
| `max_position_embeddings` | 131,072 | 131,072 | 262,144 | 262,144 | 262,144 |
| Parameters (BF16 checkpoint) | 5,123,178,051 | 7,996,156,490 | 11,959,730,224 | 25,805,936,206 | 31,273,088,876 |

<!-- c:MOD-FAM-001 --> <!-- c:MOD-FAM-002 --> <!-- c:MOD-FAM-003 --> <!-- c:MOD-FAM-004 -->
<!-- c:MOD-FAM-007 --> <!-- c:MOD-FAM-008 --> <!-- c:MOD-FAM-009 --> <!-- c:MOD-FAM-010 -->
<!-- c:MOD-FAM-013 --> <!-- c:MOD-FAM-021 -->

The parameter counts are Hugging Face's safetensors metadata for each repository at its pinned
revision (all weights are BF16). <!-- c:MOD-FAM-021 --> Every model ties its output projection to the
input embedding (`tie_word_embeddings` true) and has a 262,144-token vocabulary. <!-- c:MOD-FAM-014 -->

### Shared design

**Decoder.** Gemma 4 is a decoder-only Transformer with RMSNorm before and after each block and
QK-norm, according to the technical report (arXiv 2607.02770v1, §2). <!-- c:MOD-FAM-026 --> The
configs give every text model the `gelu_pytorch_tanh` activation, `rms_norm_eps` 1e-6 and a final
logit soft cap of 30.0 (`final_logit_softcapping`). <!-- c:MOD-FAM-012 -->

**Local and global attention.** Layers alternate between sliding-window ("local") and full
("global") attention. E2B has a global layer every fifth layer (4 local : 1 global); E4B, 12B,
26B-A4B and 31B every sixth (5 : 1). In all five the last layer is global (index 34, 41, 47, 29 and
59). <!-- c:MOD-FAM-007 --> The report states the same ratios and that the pattern follows Gemma 3.
<!-- c:MOD-FAM-026 --> Local windows are 512 tokens on E2B and E4B and 1,024 on the larger models.
<!-- c:MOD-FAM-006 -->

**Head sizes.** Every model uses 256-wide heads (`head_dim`) on local layers and 512-wide heads
(`global_head_dim`) on global layers. <!-- c:MOD-FAM-005 --> 12B, 26B-A4B and 31B also cut the number
of KV heads on global layers (`num_global_key_value_heads` 1, 2 and 4 against 8, 8 and 16 locally)
and set `attention_k_eq_v`, which the report describes as reusing keys as values on global layers,
"except in E2B and E4B". <!-- c:MOD-FAM-004 --> <!-- c:MOD-FAM-008 --> E2B and E4B leave
`num_global_key_value_heads` unset; their exported files give global layers the same KV head count
as local ones (1 and 2) with 512-wide heads and separate K and V caches
([e4b-ours](../generated/models/e4b-ours.md), [e2b-google](../generated/models/e2b-google.md), `decode`
inputs `kv_cache_k_5` and `kv_cache_k_4`). <!-- c:MOD-FAM-047 --> gemma.ai's visual guide to Gemma 4
(a repost from the Exploring Language Models blog) lists K=V among changes for all sizes and gives 2
query heads per KV head on local layers and 8 on global ones. For E2B and E4B neither K=V nor the
2:1 local grouping holds: their configs have no K=V and 8 query heads on 1 (E2B) or 2 (E4B) KV heads
in every layer, so E4B is 4:1 throughout and only E2B's 8:1 matches the guide's global figure. Take
E2B/E4B specifics from the configs and the report, not the guide. <!-- c:X-GAI-008 -->

**RoPE.** Global layers use `proportional` RoPE with `rope_theta` 1,000,000 and
`partial_rotary_factor` 0.25; local layers use the default RoPE with `rope_theta` 10,000.
<!-- c:MOD-FAM-011 --> The report calls the global variant p-RoPE with p = 0.25.
<!-- c:MOD-FAM-026 --> How the E4B file computes these angles, and why Agello's file splits them into
512-token blocks, is on [gemma4-e4b.md](gemma4-e4b.md).

**Per-layer embeddings (E models only).** E2B and E4B add a per-layer embedding: each token gets a
256-wide vector per decoder layer from a table of `vocab_size_per_layer_input` (262,144) rows.
<!-- c:MOD-FAM-010 --> The table is large but only looked up, which is why the card counts
"effective" parameters without it. The arithmetic, from the configs:

| | E2B | E4B | 12B | 26B-A4B | 31B |
| --- | --- | --- | --- | --- | --- |
| Main embedding, 262,144 × `hidden_size` | 402,653,184 | 671,088,640 | 1,006,632,960 | 738,197,504 | 1,409,286,144 |
| Per-layer embedding, layers × 256 × 262,144 | 2,348,810,240 | 2,818,572,288 | none | none | none |
| Report, Table 1 ("Embedder") | 400M + 2,340M | 670M + 2,820M | 1,000M | 740M | 1,410M |

The arithmetic agrees with the report after rounding, except E2B's per-layer figure: 2,348.8M would
round to 2,350M, and the report gives 2,340M (0.4% lower). <!-- c:MOD-FAM-022 -->

Subtracting both embedding tables from the checkpoint's parameter count gives 2.37B for E2B and
4.51B for E4B, close to the card's "2.3B effective" and "4.5B effective"; Google doesn't publish the
exact formula. <!-- c:MOD-FAM-023 -->

**KV sharing (E models only).** E2B shares KV caches across its last 20 of 35 layers and E4B across
its last 18 of 42, so only 15 and 24 layers keep caches of their own; the report gives the same
ratios (20/35, 18/42). <!-- c:MOD-FAM-009 --> In the exported files the shared layers read the caches
of layers 13 and 14 (E2B) or 22 and 23 (E4B), the last local and last global layer before the
shared block ([e4b-ours](../generated/models/e4b-ours.md), [e2b-google](../generated/models/e2b-google.md),
"Attention per layer"). <!-- c:MOD-FAM-048 -->

**Other differences.** E2B alone sets `use_double_wide_mlp`. <!-- c:MOD-FAM-016 --> 12B, 26B-A4B and
31B set `use_bidirectional_attention` to `vision` (E2B and E4B leave it unset).
<!-- c:MOD-FAM-058 --> The configs' special-token IDs are the same in every model (`<|image>` 255999,
`<|audio>` 256000, `<|image|>` 258880, `<|audio|>` 258881, `<image|>` 258882, `<audio|>` 258883);
see [tokenizer-and-chat-format.md](tokenizer-and-chat-format.md).

### The mixture-of-experts model (26B-A4B)

`gemma-4-26B-A4B-it` sets `enable_moe_block`, `num_experts` 128, `top_k_experts` 8 and
`moe_intermediate_size` 704, next to a dense `intermediate_size` of 2,112. <!-- c:MOD-FAM-015 --> The
card describes it as 25.2B total and 3.8B active parameters with "8 active / 128 total and 1 shared"
experts; the safetensors metadata counts 25,805,936,206 parameters. <!-- c:MOD-FAM-057 -->
<!-- c:MOD-FAM-021 --> Every expert's weights must still be resident or mapped, so the memory cost
follows the total, not the active count.

### Modalities

| | Text | Image | Audio | Video | How images and audio enter |
| --- | --- | --- | --- | --- | --- |
| E2B, E4B | yes | yes | yes | as frames | 150M vision encoder (16 layers, width 768); 305M audio encoder (12 Conformer layers) |
| 12B | yes | yes | yes | as frames | encoder-free: 48 × 48 RGB patches through one projection; raw 40 ms audio chunks (640 samples) projected |
| 26B-A4B, 31B | yes | yes | no | as frames | 550M vision encoder (27 layers, width 1,152); `audio_config` null |

<!-- c:MOD-FAM-017 --> <!-- c:MOD-FAM-018 --> <!-- c:MOD-FAM-019 --> <!-- c:MOD-FAM-020 -->
<!-- c:MOD-FAM-024 -->

The vision encoders use 16-pixel patches pooled 3 × 3, and the token budget per image is one of 70,
140, 280, 560 or 1,120; the configs default to 280 (`vision_soft_tokens_per_image`).
<!-- c:MOD-FAM-054 --> The card limits audio to 30 seconds and video to 60 seconds at one frame per
second; the processors produce one audio token per 40 ms (`audio_ms_per_token`) and at most 750
(`audio_seq_length`), which is 30 s. <!-- c:MOD-FAM-053 --> What the LiteRT-LM files do with images
(which budgets they export) is on [gemma4-e4b.md](gemma4-e4b.md).

### Trained context

| | Trained (`max_position_embeddings`) | Card | In LiteRT-LM files |
| --- | --- | --- | --- |
| E2B, E4B | 131,072 | 128K | export window 32003 (the magic number); Google's file pages say "up to 32k" |
| 12B | 262,144 | 256K | the file page says "up to 128k"; the metadata template in LiteRT-LM sets `max_num_tokens` 131071 |
| 26B-A4B, 31B | 262,144 | 256K | GPU and web files only; the web figures are "up to 195k" and "80k" |

<!-- c:MOD-FAM-013 --> <!-- c:MOD-FAM-025 --> <!-- c:MOD-FAM-042 --> <!-- c:MOD-FAM-059 -->

The trained context is what the model saw in training. A file's export window (32003 for E2B/E4B,
rewritten at load) and the runtime's allocated context (`maxNumTokens`) are separate limits; see
[gemma4-e4b.md](gemma4-e4b.md) and [../litert-lm/executors.md](../litert-lm/executors.md).

### Base and instruction-tuned checkpoints

Each size has a pre-trained base repository (`google/gemma-4-E2B`, `-E4B`, `-12B`, `-26B-A4B`, `-31B`)
and an instruction-tuned one with the `-it` suffix. <!-- c:MOD-FAM-034 --> All models share the
tokenizer; base models end generation with `<eos>` (ID 1), instruction-tuned models with `<turn|>`
(ID 106), according to the report (§3). <!-- c:MOD-FAM-035 --> The thinking mode, the chat turns and
tool calling are instruction-tuned behavior ([tokenizer-and-chat-format.md](tokenizer-and-chat-format.md)).
Every `litert-community` LiteRT-LM repository names an `-it` model as its `base_model`.
<!-- c:MOD-FAM-037 -->

The instruction-tuned `generation_config.json` files set sampling to temperature 1.0, top-p 0.95,
top-k 64, which the cards also recommend, and stop on token IDs 1, 106 and 50.
<!-- c:MOD-FAM-049 --> <!-- c:MOD-FAM-050 --> 12B's (and 12B-assistant's) also suppress tokens 258883
(`<audio|>`) and 258882 (`<image|>`). <!-- c:MOD-FAM-050 -->

### The assistant models (MTP drafters)

`google/gemma-4-12B-it-assistant` and `google/gemma-4-31B-it-assistant` (pinned) are drafters: each
card says the model is a multi-token prediction (MTP) drafter that "predicts several tokens ahead,
which the target model then verifies in parallel". <!-- c:MOD-FAM-030 --> Their configs:

| | 12B-it-assistant (`46d4c6f1`) | 31B-it-assistant (`627c5ec1`) |
| --- | --- | --- |
| `architectures` | `Gemma4UnifiedAssistantForCausalLM` | `Gemma4AssistantForCausalLM` |
| Layers (`layer_types`) | 4: local, local, local, global | same |
| `hidden_size` / `intermediate_size` | 1,024 / 8,192 | 1,024 / 8,192 |
| Heads: query / local KV / global KV | 16 / 8 / 1 | 32 / 16 / 4 |
| `num_kv_shared_layers` | 4 (all layers) | 4 (all layers) |
| `backbone_hidden_size` | 3,840 (= 12B's `hidden_size`) | 5,376 (= 31B's) |
| `num_centroids` / `centroid_intermediate_top_k` / `use_ordered_embeddings` | 2,048 / 32 / false | 2,048 / 32 / false |
| `final_logit_softcapping` | null | null |
| Parameters (BF16) | 422,856,964 | 469,518,596 |

<!-- c:MOD-FAM-028 --> <!-- c:MOD-FAM-033 -->

What the sources show, and no more:

- The local and global head sizes, RoPE and window match the target model's config, and
  `num_kv_shared_layers` equals the layer count, so no drafter layer has a KV cache of its own.
  <!-- c:MOD-FAM-028 --> The report says the drafter is "a 4-layer Transformer block that
  cross-attends to the KVs of the main model", fed with the main model's last-layer activations and
  token embeddings, which removes the need for an MTP prefill (§2.6). <!-- c:MOD-FAM-031 -->
- `backbone_hidden_size` is the target's width, which fits the report's statement that the target's
  activations are fed into the drafter. <!-- c:MOD-FAM-028 -->
- The report gives the drafter width as 256 for E2B and E4B and 1,024 for 26B-A4B and 31B, and says
  the E2B and E4B drafters replace the full-vocabulary projection with a top-k over token clusters,
  shrinking the last matrix multiplication to d × 4,096. <!-- c:MOD-FAM-031 --> The unpinned
  `google/gemma-4-E4B-it-assistant` (`8d0031ea`) has `hidden_size` 256, 2,048 centroids, top-k 32 and
  `use_ordered_embeddings` true; 262,144 / 2,048 = 128 tokens per centroid and 32 × 128 = 4,096,
  which matches. <!-- c:MOD-FAM-032 --> The pinned 12B and 31B drafters carry the same centroid
  fields with `use_ordered_embeddings` false; whether their heads use the clusters is not stated.
- `gemma-4-31B-it-assistant`'s `generation_config.json` sets `is_assistant` true and
  `num_assistant_tokens` 6 with a constant schedule; `gemma-4-12B-it-assistant`'s has neither.
  <!-- c:MOD-FAM-029 -->
- Google's MTP page (unversioned, *volatile*) lists drafters for all five sizes, each named with the
  `-assistant` suffix. <!-- c:MOD-FAM-030 --> The E2B, E4B and 26B-A4B drafters exist on Hugging Face
  but aren't pinned here.

The report's parameter table lists drafters of 76M (E2B), 77M (E4B), 400M (12B), 430M (26B-A4B) and
500M (31B). <!-- c:MOD-FAM-033 --> In the E2B and E4B `.litertlm` files the drafter is a section,
`tf_lite_mtp_drafter` (44.3 MB and 45.3 MB), not a separate download
([e4b-ours](../generated/models/e4b-ours.md), [e2b-google](../generated/models/e2b-google.md));
<!-- c:MOD-FAM-041 --> how LiteRT-LM drafts and verifies is on
[../litert-lm/executors.md](../litert-lm/executors.md).

### Google's documentation site

Google DeepMind's Gemma documentation is at [gemma.ai](https://gemma.ai/) (unversioned, *volatile*,
read 2026-10-15): model cards, prompt formatting, capability guides (thinking, tool calling, vision,
audio, MTP), deployment pages (LiteRT-LM among them) and a release log. The `ai.google.dev/gemma/docs`
pages this reference cites still resolve, and the Gemma 4 prompt-formatting text is nearly the same on
both sites (gemma.ai rewords the opening, adds a Gemma 1-3 section, and its function-calling table spells
the tool-definition token `<|tool|>`). <!-- c:X-GAI-001 --> The release log dates Gemma 4 E2B, E4B, 31B and 26B-A4B to 2026-03-31,
their MTP drafters to 2026-04-16, 12B Unified to 2026-06-03 and EmbeddingGemma 2 to 2026-10-06.
<!-- c:X-GAI-002 -->

### License

Every pinned model card declares `license: apache-2.0` with a link to Google's Gemma 4 license page,
whose heading is "Apache License 2.0"; the report says the models are released under Apache 2.0.
The `litert-community` LiteRT-LM repositories also declare `apache-2.0`. <!-- c:MOD-FAM-036 -->

### LiteRT-LM files on Hugging Face

The `litert-community` organization publishes these Gemma 4 repositories (revisions and sizes from
the Hugging Face API, 2026-10-15): <!-- c:MOD-FAM-037 -->

| Repository @ revision | Files (bytes) | What they are |
| --- | --- | --- |
| `gemma-4-E2B-it-litert-lm` @ `b3ca0d2f` | `gemma-4-E2B-it.litertlm` (2,588,147,712); `-gpu.litertlm` and `-web.litertlm` (2,008,432,640 each); `-web.task`; NPU files for Google Tensor G5/G6, Qualcomm and Intel | standard file: text, vision, audio, MTP drafter |
| `gemma-4-E4B-it-litert-lm` @ `2eee7ac3` | `gemma-4-E4B-it.litertlm` (3,659,530,240); `-gpu.litertlm` and `-web.litertlm` (2,969,059,328 each); `-web.task` | standard file: text, vision, audio, MTP drafter |
| `gemma-4-12B-it-litert-lm` @ `7a0b1ce0` | `gemma-4-12B-it.litertlm` (6,883,278,368); `-gpu.litertlm` and `-web.litertlm` (5,986,074,624 each) | the card: text, vision, audio and MTP; LiteRT-LM v0.17 or later |
| `gemma-4-26B-A4B-it-litert-lm` @ `a6268756` | `-gpu.litertlm` and `-web.litertlm` (15,786,524,672 each) | the card: text only for now, blockwise int4, LiteRT-LM v0.17.1 or later |
| `gemma-4-31B-it-litert-lm` @ `99617231` | `-gpu.litertlm` and `-web.litertlm` (19,217,989,632 each) | the card: text only for now, blockwise int4, LiteRT-LM v0.17.1 or later |

<!-- c:MOD-FAM-037 --> <!-- c:MOD-FAM-038 --> <!-- c:MOD-FAM-039 -->

The `-gpu` files of E2B and E4B are three-section "artisan" files (metadata, tokenizer, one
`tf_lite_artisan_text_decoder`). That section makes `EngineSettings` switch a GPU engine to
`Backend::GPU_ARTISAN`, and the documented build's `CreateLlmLiteRtCompiledModelExecutor` rejects any
backend other than CPU, GPU and NPU ([../litert-lm/file-format.md](../litert-lm/file-format.md), claims
LM-FMT-017 and LM-FMT-018). 26B-A4B and 31B have only `-gpu` and `-web` files; if those are built the
same way (their sections were not inspected), the documented build can't run them.
<!-- c:MOD-FAM-040 --> The `-gpu` and `-web` files of each model have the same size in bytes; whether they
are the same file was not checked.

The cards add memory figures. Google reports, for an iPhone 17 Pro with the GPU backend (1,024
prefill and 256 decode tokens, 2,048-token context), 1,450 MB for E2B and 3,380 MB for E4B ("CPU/GPU
Memory"), as of 2026-10-15. The tables don't name the file; their "Model size" column (2,583 MB and
3,654 MB) points to the plain `gemma-4-E2B-it.litertlm` and `gemma-4-E4B-it.litertlm`, which at the
pinned revisions are 2,588,147,712 and 3,659,530,240 bytes, so the benchmarked files may be earlier
builds. <!-- c:MOD-FAM-046 --> gemma.ai's LiteRT-LM page gives 2.58 GB and 3.65 GB, matching the cards'
figures (it names no source), and says LiteRT-LM ships E2B and E4B, with larger models to come.
<!-- c:X-GAI-004 -->

The same page documents Google's export path from a Hugging Face checkpoint to `.litertlm`:
`litert-torch export_hf` from the `litert-torch-nightly` package, with `--externalize_embedder` and
`--jinja_chat_template_override=litert-community/gemma-4-E2B-it-litert-lm`; the page doesn't explain
that flag beyond its name, which suggests the exported file takes its chat template from the
litert-community repository. <!-- c:X-GAI-005 --> Whether that path
reproduces the published files byte for byte, and whether it could emit E4B with exact FP16 rotary
positions instead of patching the graph afterwards ([gemma4-e4b.md](gemma4-e4b.md)), wasn't tried.

## On Apple

**What runs where.** Agello downloads an E4B file (Google's, with 512-token rotary blocks: ours) and
loads it with the GPU backend; E2B's standard file has the same sections and would load the same way
([gemma4-e2b.md](gemma4-e2b.md)). 12B's standard file targets "macOS, Linux and Windows" on its card,
and 26B-A4B and 31B have only artisan and web files.

**Could a phone run them? (estimates)** The inputs: the file sizes above; the KV arithmetic below;
and the iPhone 17 Pro of Agello's results (12 GB of RAM), whose app limit is 8 GiB (8,589,934,592 bytes, printed
"8.6 GB"; an early reading can show 6 GiB) <!-- c:APL-001 --> <!-- c:APL-004 -->
([../apple-platform.md](../apple-platform.md#c1-86-gb-or-64-gb)).
These are estimates from configs and file sizes, not measurements.

**KV cache per model (trained model, estimate).** A local layer stores K and V for at most
`sliding_window` positions; a global layer stores K and V (or only K when `attention_k_eq_v`) for every
position; shared layers store nothing. With one byte per element (INT8, as the E2B/E4B files store
their caches, scales ignored):

```
bytes(L) = n_local × (2 × kv_heads × 256) × min(L, window)
         + n_global × (k_or_kv × global_kv_heads × 512) × L
```

| | KV layers (local + global) | Bytes per token past the window | At 4,096 | At 32,768 | At 131,072 |
| --- | --- | --- | --- | --- | --- |
| E2B | 12 + 3 | 3,072 | 15.7 MB | 103.8 MB | 405.8 MB |
| E4B | 20 + 4 | 8,192 | 44.0 MB | 278.9 MB | 1,084.2 MB |
| 12B | 40 + 8 | 4,096 | 184.5 MB | 302.0 MB | 704.6 MB |
| 26B-A4B | 25 + 5 | 5,120 | 125.8 MB | 272.6 MB | 775.9 MB |
| 31B | 50 + 10 | 20,480 | 503.3 MB | 1,090.5 MB | 3,103.8 MB |

<!-- c:MOD-FAM-043 --> <!-- c:MOD-FAM-044 -->

BF16 caches are twice these. The report's Table 3 (arXiv 2607.02770v1, unit labelled "Gb") gives
INT8 KV at 32K as +0.05, +0.14, +0.28, +0.28 and +1.10; the arithmetic matches 12B, 26B-A4B and 31B
within rounding, but the report's E2B and E4B figures are about half of it (0.48x and 0.50x).
Halving only the global-layer term gives 53.5 MB and 144.7 MB, which round to the report's 0.05 and
0.14. <!-- c:MOD-FAM-045 --> The configs (`attention_k_eq_v` false) and the exported files (separate
512-wide K and V caches) both say E2B and E4B store values on global layers, so this page keeps its
own figure and lists the report's as an open question.

The exported E4B file allocates more than the trained model needs: its local caches are as long as
the export window, not 512. 24 caches at 32,003 positions come to 917,590,016 bytes
(20 × 2 × 2 × 256 × 32,003 + 4 × 2 × 2 × 512 × 32,003). <!-- c:MOD-FAM-047 --> The E4B page owns that
figure and contradiction C9 ([gemma4-e4b.md](gemma4-e4b.md)).

**Verdict per model (estimates):**

| | Weights on disk | Phone? | Why |
| --- | --- | --- | --- |
| E2B | 2.59 GB file | yes | Google ships it for iOS; 1,450 MB reported on an iPhone 17 Pro |
| E4B | 3.66 GB file | yes (Agello) | 3,380 MB reported by Google; Agello's own figures are on [../memory.md](../memory.md) |
| 12B | 6.88 GB file | no | its 6,883,278,368-byte file (MOD-FAM-037) is about 1.7 GB under the 8 GiB limit, leaving little for KV, activations and the app; the card targets desktops and reports about 7.9 GB of GPU memory on Macs |
| 26B-A4B | 15.79 GB (int4) | no | larger than the device's 12 GB of RAM; artisan/web files only |
| 31B | 19.22 GB (int4) | no | larger than the device's RAM; artisan/web files only |

<!-- c:MOD-FAM-046 --> <!-- c:MOD-FAM-052 -->

The 12B verdict assumes the whole file must be resident. Part of a `.litertlm` file can stay
memory-mapped (the E4B card says the embedding parameters are), so a smaller footprint is possible in
principle; nobody has measured 12B on an iPhone here.

## Limits and failure modes

- **"128K" is not what the E files run.** E2B and E4B were trained to 131,072 positions, but the
  published files export a 32003-position window; a longer context needs a re-export. The 12B
  metadata in LiteRT-LM caps it at 131071 tokens although the model was trained to 262,144.
- **Artisan files need Google's binary.** An artisan section selects the `GPU_ARTISAN` backend,
  which the open-source executor rejects (LM-FMT-017). 26B-A4B and 31B have only `-gpu`/`-web` files
  on `litert-community`; whether they carry that section wasn't inspected.
- **12B is a different processor path.** Its config is `gemma4_unified`; LiteRT-LM's metadata for it
  sets `merge_patches` true, 10,080 patches and `skip_mel_spectrogram_extraction` true, unlike E2B/E4B
  ([`models/gemma4/LlmMetadataProto_12b.pbtext:1-60`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/models/gemma4/LlmMetadataProto_12b.pbtext#L1-L60)).
  <!-- c:MOD-FAM-059 -->
- **The MoE model's memory follows its total size.** "A4B" describes compute per token; all 128
  experts per layer must be available.
- **Drafters are per target.** Each assistant's `backbone_hidden_size` equals its target's width, so
  a drafter can't be paired with another size. <!-- c:MOD-FAM-028 -->
- **Base models don't speak the chat format.** They end with `<eos>`, not `<turn|>`, and the
  thinking mode is instruction-tuned behavior.

## Where in the code

| Concept | Symbol | Citation |
| --- | --- | --- |
| Gemma 4 metadata fields (`merge_patches` "Only used for Gemma4 12B model") | `proto::Gemma4` | [`runtime/proto/llm_model_type.proto:247-300`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/proto/llm_model_type.proto#L247-L300) |
| Canonical metadata for 12B | `LlmMetadataProto_12b.pbtext` | [`models/gemma4/LlmMetadataProto_12b.pbtext:1-60`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/models/gemma4/LlmMetadataProto_12b.pbtext#L1-L60) |
| Canonical metadata for E2B/E4B | `LlmMetadataProto.pbtext` | [`models/gemma4/LlmMetadataProto.pbtext:1-62`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/models/gemma4/LlmMetadataProto.pbtext#L1-L62) |
| Official configs | `config.json`, `generation_config.json` | [hf-configs](../generated/hf-configs.md), [hf/](../generated/hf/google__gemma-4-E4B-it/config.json) |
| What each E2B/E4B file contains | sections, signatures, per-layer attention | [e4b-ours](../generated/models/e4b-ours.md), [e2b-google](../generated/models/e2b-google.md), [e4b-gpu](../generated/models/e4b-gpu.md) |

## Open questions

- **KV at 32K for E2B/E4B.** The report's Table 3 (+0.05 and +0.14) is about half of what the configs and
  the exported caches imply (103.8 MB and 278.9 MB). Not resolved.
- **Effective parameters.** The formula behind "2.3B / 4.5B effective" isn't published; the
  subtraction above is a match, not a definition.
- **12B and 31B drafter heads.** Their configs carry 2,048 centroids with `use_ordered_embeddings`
  false; whether their output heads use the clusters wasn't established.
- **Base checkpoint sizes.** Hugging Face counts 26,544,131,376 parameters for `google/gemma-4-26B-A4B`
  and 32,682,372,656 for `google/gemma-4-31B`, more than their `-it` versions; the base configs weren't
  pinned or read.
- **12B on an iPhone.** Never tried; the verdict above is arithmetic.
- **Are `-gpu` and `-web` the same file?** Equal sizes on every repository; not compared.
- **Are the `.litertlm` files built from the QAT mobile checkpoints?** gemma.ai's QAT page describes
  E2B and E4B `-qat-mobile-*` checkpoints on a wNa8o8 scheme "with 2-bit decoding layers and
  optimized KV caches" <!-- c:X-GAI-009 -->; the files' INT2 tensors would fit, but no source links
  the two.
- **26B-A4B and 31B file sections.** Only the E2B and E4B `-gpu` files were inspected; whether the
  26B-A4B and 31B files carry a `tf_lite_artisan_text_decoder` section is inferred, not checked.
