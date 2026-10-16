# Returns: mod-e4b (B1, models/gemma4-e4b.md)

## Settings rows (for configuration.md)

| name | layer | default | where read (citation link) | effect on Apple | version added | Agello sets it? | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `maxNumTokens` / `max_num_tokens` | Swift `EngineConfig`, C `litert_lm_engine_settings_set_max_num_tokens`, `LlmExecutorSettings` | 0 (E4B then gets 4,096: no `max_num_tokens` in its metadata) | [`runtime/executor/magic_number_configs_helper.cc:243-271`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/magic_number_configs_helper.cc#L243-L271) | allocated context for E4B: the value if below 32,003, else 32,000; KV = 28,672 bytes per position (917.5 MB at 32,000) | before v0.18.0 | yes (planner, 4K–32K windows) | MOD-E4B-038, -039, -056; executor side LM-EXE-003/005 |
| visual token budget (`visual_token_budget`, Swift `ConversationConfig.visualTokenBudget`) | conversation optional args (`Gemma4DataProcessorArguments`) | unset: resize to the metadata's `max_num_patches` (2520 = 280 tokens) | [`runtime/conversation/model_data_processor/multimodal_processor_helper.cc:91-103`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/multimodal_processor_helper.cc#L91-L103), [`runtime/conversation/conversation.cc:556-565`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/conversation.cc#L556-L565) | budget b → at most 9b patches; with E4B any b > 280 is InvalidArgument | v0.18.0 (same code) | yes: 140 / 280 / 560, clamped to `ModelInfo.maxVisionTokenBudget` (280) | MOD-E4B-050, -051 |
| `max_vision_tokens_per_image` | `EngineSettings` (C `litert_lm_engine_settings_set_max_vision_tokens_per_image`) | from metadata: `max_num_patches / pooling²` = 280 for E4B | [`runtime/core/engine_advanced_impl.cc:457-476`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/engine_advanced_impl.cc#L457-L476) | limits which `vision_*` signatures are selected and the largest accepted budget | v0.18.0 (same code) | no | MOD-E4B-049, -050 |
| activation data type | `EngineConfig` / executor settings | unset → section hint `prefer_activation_type=fp16` → FP16 on the GPU | see [../../litert-lm/file-format.md#what-the-hints-do](../../litert-lm/file-format.md#what-the-hints-do) (LM-FMT-013) | FP16 rotary positions: Google's E4B file corrupts verbatim text past 2,048 (#3913); our file is exact; FLOAT32 avoids it at ~2× decode time | before v0.18.0 | no (default) | MOD-E4B-075, -082 |
| speculative decoding | `EngineConfig` / session | off | executors page (LM-EXE-071/072) | loads `tf_lite_mtp_drafter` (45.3 MB), reads caches 22/23 | before v0.18.0 | yes (user option, `caps.mtp`) | MOD-E4B-041..044 |
| vision / audio backend | `EngineConfig.visionBackend`, `.audioBackend` | none | executors page (LM-EXE-028/029/031) | adapters always CPU; audio encoder must be CPU; vision encoder GPU or CPU | before v0.18.0 | yes: `.cpu()` for both | MOD-E4B-075 |

## Terms (for glossary.md)

| term | meaning | prefer / avoid |
| --- | --- | --- |
| effective parameters | Google's headline size for E2B/E4B (4.5B for E4B) that leaves out the per-layer embedding tables (8B total) | prefer "4.5B effective (8B total)"; avoid "4B model" without qualification |
| per-layer embeddings (PLE) | a second embedding table giving each token a 256-value vector per layer (42 × 256 for E4B), added after each layer's MLP through a gated bottleneck | prefer "per-layer embeddings"; avoid "PLE" without expanding it once |
| KV sharing | layers 24–41 of E4B compute no K/V and read layer 22's (local) or 23's (global) caches | prefer "KV sharing", "KV-shared layers" |
| local / global layer | sliding-window (512 positions, head 256) / full attention (head 512) | prefer "local"/"global"; config names `sliding_attention`/`full_attention` |
| p-RoPE (proportional RoPE) | RoPE on a fraction of the head (E4B global: 128 of 512 dims, θ 1e6) | spell out once |
| trained context | 131,072 positions (`max_position_embeddings`) | never "the model's context" when the file's is meant |
| export window | the file's 32003 magic number | avoid "max context" for it |
| allocated context | `maxNumTokens` as rewritten at load (≤ 32,000 for E4B) | |
| rotary block (512-token block) | our file's split position = 512·h + l | prefer "512-token rotary blocks"; avoid "RoPE fix" alone |
| vision token budget | soft tokens per image (70/140/280 in the E4B file) | avoid quoting 560/1120 for the phone |
| soft token | one pooled vision (or audio) embedding row fed to the decoder | |
| MTP drafter | the 4-block speculative-decoding head in `tf_lite_mtp_drafter` | avoid "draft model" (it has no KV cache of its own) |

## Contradictions

- **C2** (vision tokens per image): the trained model supports 70/140/280/560/1120 (card, report); the E4B file exports only `vision_70/140/280` and `max_num_patches` 2520 = 280 × 9; LiteRT-LM (agello-next and v0.18.0) caps the budget at 280 and rejects more with InvalidArgument; Agello's "Detailed (560)" preset is clamped to 280, so USER_GUIDE's 560 isn't what runs. Claims MOD-E4B-019, -046, -049, -050, -051, -052.
- **C9** (E4B KV at 32K: 918 vs 940 MB): 28,672 bytes per position; a 32K request (`maxNumTokens` 32,768) allocates 32,000 positions = 917,504,000 bytes (AUDIT's 918 MB, decimal); 940 MB is 32,768 positions (939,524,096 bytes), which this file never gets. MiB: 875 / 896. Claims MOD-E4B-037, -038, -039.
- **C14** (context length): trained context 131,072 ("128K", config and card); export window 32003 ("up to 32k", litert-community card); allocated context = `maxNumTokens` below 32,003, else 32,000 (4,096 if unset). The archived plan's "128K" is the trained model only. Claims MOD-E4B-056, -057, -073.
- **NEW** (drafter head): the Gemma 4 report v2 says the E2B/E4B drafters use a clustered top-k head (d × 4096); the exported E4B drafter has a full-vocabulary INT4 [262144, 256] head and outputs 262,144 logits. Both stand; they describe different artifacts. MOD-E4B-023, -043.
- **NEW** (drafter key scales, extends E4B-GRAPH): E4B-GRAPH notes only the local mismatch (layer 4's 0.005961 vs cache 22's 0.005955); the global drafter block also uses layer 5's 0.0010908608 instead of cache 23's 0.0010912281. MOD-E4B-044.
- **NEW** (E4B-GRAPH / AUDIT §5.2 "Prefill runs layers 0–23"): more precisely, prefill runs attention for layers 0–22 and only the K/V projection (with its input norm) of layer 23 (23 SOFTMAX, 210 FULLY_CONNECTED, 188 RMS norms). MOD-E4B-029, -007.

## Notes for other pages

- **memory.md:** KV table (MOD-E4B-038), C9 (-039), local vs global split at 32,000 positions: 655,360,000 + 262,144,000 bytes (-081); weights by section and type (page table "Weights and quantization", claims -010, -030, -032, -047, -053); the tied embedding is stored twice (335,544,320 bytes of INT2, -032); the per-layer embedder section is 836,778,512 bytes of which ~132 MB are per-row quantization parameters (-011). litert-community's card (2eee7ac3) says "LiteRT-LM framework always keeps main weights in memory, while the embedding parameters are memory mapped" (not verified here; worth checking against the loader).
- **patches.md / contributing.md:** our model file facts (MOD-E4B-024, -026, -065..-073); reproducibility (-069); results on M6, iPhone and Google's v0.18.0 WebGPU (-070, -071, -072). `patch_rope.py` is not in any pinned tree (it's on the `agello` branch at `f2e7afa0`), so it can't be cited with cite.py; consider pinning a commit that has it or adding it to agello-next.
- **apple-platform.md:** the #3913 M1 Max report places M1 in "a family with A14 and A15 iPhones"; Metal's families put the A14 and M1 in Apple7 and the A15 in Apple8, so only A14-class iPhones share M1's family. The M1 Max loss (MOD-E4B-063) has no iPhone confirmation.
- **gemma4-family.md:** E4B rows (MOD-E4B-001, -002, -003, -005, -013, -015, -016, -018, -020); MatFormer is not mentioned for Gemma 4 (MOD-E4B-017).
- **tokenizer-and-chat-format.md:** special token ids and generation defaults from the config (MOD-E4B-022).
- **troubleshooting.md:** "Visual token budget (N) cannot be larger than the engine's max vision tokens per image (280)" for E4B budgets above 280 (MOD-E4B-050).
- **Run A pages:** consistent with this page; no disagreement found. ml-drift/kernels.md (MD-KRN-073..076) and executors.md (LM-EXE-003/006/022/027/030/073) match the file facts here.
- **Tooling (for maintenance.md / the main loop):**
  - `claims.py`'s `ID_RE` is `^([A-Z]+(?:-[A-Z]+)*)-(\d{3})$`, which can't match the prefixes `MOD-E4B` and `MOD-E2B` (digits). `claims.py check mod-e4b` therefore reports "id must look like LM-ENG-001" for every claim. With the regex relaxed in memory to `[A-Z0-9]+` for the later parts, the check passes with 0 errors (82 claims, 73 tier A, every tier-A claim tagged). The fix is one character class in `tools/agello-docs/claims.py`; I didn't change it (out of bounds).
  - `generated/models/e4b-ours.md` (and e4b-google.md) render the "Attention per layer (`decode`)" block with its header only; the rows are in `e4b-ours.json` (`tflite['10'].layer_table`). The md generator drops them.
  - `gen_model_tables.py` can't parse `tf_lite_per_layer_embedder` ("unpack_from requires a buffer of at least 704644672 bytes … actual buffer size is 268435456"): it reads only 256 MiB of an 836.8 MB inline section. Facts about that section were read directly (MOD-E4B-010, -011).

## Open questions

- Report Table 3 lists E4B at "+0.14" (unit "Gb") for an int8 KV cache at 32k; the export allocates 917.5 MB at 32,000 positions. Only window-sized local caches plus keys-as-values in the global layers would come near 0.14 GB, and E4B doesn't use keys-as-values. Basis unknown.
- How the 4.5B "effective" figure is derived isn't in the report (Table 1 without per-layer embeddings: 5.1B, or ~4.6B without encoders and drafter).
- Whether the drafter's mismatched key scales lower MTP acceptance measurably (untested).
- AUDIT §5.2 says the main graph's activation scales are "static per-tensor"; this page only shows QUANTIZE/DEQUANTIZE counts, not the scale granularity.
- No A14-class iPhone run of the 2048-block or 512-block file exists; the Apple7 loss is known only from an M1 Max.
