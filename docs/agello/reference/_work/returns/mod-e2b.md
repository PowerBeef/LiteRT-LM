# Returns: mod-e2b (B1, models/gemma4-e2b.md)

## Settings rows (for configuration.md)

none (E2B adds no setting; the E4B rows apply: `maxNumTokens`, activation type via `prefer_activation_type`, vision/audio backends, `enable_speculative_decoding`).

## Terms (for glossary.md)

| term | meaning | prefer / avoid |
| --- | --- | --- |
| KV-shared layer | a decoder layer that computes queries only and reads the K/V cache of the last non-shared layer of its attention type (E2B: layers 15–34 read caches 13 and 14; E4B: 24–41 read 22 and 23) | prefer "KV-shared layer"; avoid "cross-layer attention" |
| double-wide MLP | `use_double_wide_mlp`: KV-shared layers get twice the feed-forward width (E2B: 12,288 instead of 6,144); E4B doesn't use it | prefer "double-wide feed-forward"; avoid "2x MLP" without the layer range |
| effective parameters | Google's "E" count for E2B/E4B: by our arithmetic, decoder plus token embedding, without per-layer tables and encoders (E2B 2.28 B ≈ card's 2.3B) | say "effective (Google's count)" |
| artisan file (`-gpu`) | a `.litertlm` with one `tf_lite_artisan_text_decoder` section (`backend_constraint=gpu_artisan`) for Google's closed `GPU_ARTISAN` executor | prefer "artisan `-gpu` file"; avoid "GPU version of the model" (the regular file runs on the GPU too) |
| `GPU_ARTISAN` | LiteRT-LM backend selected automatically for artisan files; served only by closed legacy engines | |
| inline weights | weights stored in the TFLite flatbuffer's buffers (every E2B section) as opposed to after the graph in the same section (E4B main graph) | |

## Contradictions

- **NEW (card vs E2B config/file):** the google/gemma-4-E2B-it card says global layers "feature unified Keys and Values"; E2B's config has `attention_k_eq_v: false` and its file keeps separate K and V caches for layers 4, 9, 14. Resolution: the statement applies to the 12B/26B-A4B/31B configs (`attention_k_eq_v: true`), not to E2B (or E4B). Claim: MOD-E2B-011. (Belongs on gemma4-family.md too.)
- **NEW (C14 analog for E2B):** Agello's `InferencePlanner.swift` comment says "Gemma 4 E2B/E4B support 128K"; the trained context is 131,072, but the E2B export window is 32,003 and LiteRT-LM allocates at most 32,000. Resolution: 128K is the trained model, 32K the exported file; both are true of different things. Claim: MOD-E2B-034.
- **NEW (lead E4B-GRAPH.md vs file bytes):** E4B-GRAPH says only the drafter's local blocks use a mismatched K scale (0.005961); the generated JSON shows the global block also differs (0.00109086 vs cache 23's 0.001091228). E2B's drafter has no mismatch. Claim: MOD-E2B-037. Owner: mod-e4b (gemma4-e4b.md).

## Notes for other pages

- **gemma4-e4b.md (mod-e4b):**
  - E4B's per-layer embedder (unparsed by the generator, "LM-FMT" open question) is 42 per-channel INT2 tensors `[262144, 256]`, 704,643,072 bytes, plus ~12 bytes of quantization parameters per row (836,778,512-byte section). I read the tensor table from the local `~/Models/agello/gemma-4-E4B-it.litertlm` with the tools venv's flatbuffer schema (read-only). Claim MOD-E2B-021.
  - E4B's prefill signatures compute full blocks for layers 0–22 and only the K/V of layer 23 (rms_norm 188 = 23×8 + 3 + 1; runtime_bmm 46 = 23×2; SOFTMAX 23; FC 210 = 23×9 + 2 + 1). E4B-GRAPH.md says "Prefill runs layers 0–23", slightly imprecise. Same pattern as MOD-E2B-028.
  - E4B quantization breakdown by counts: INT4 258 = 24×7 + 18×5, INT8 85 = 42×2 + 1, INT2 1 = output projection (167,772,160 = 262,144 × 2,560 / 4). MOD-E2B-019.
  - E4B drafter global scale mismatch (above).
- **memory.md:** E2B KV = 9,216 B/position (MOD-E2B-031..033); E2B main-graph weights 777,983,264 bytes inline (MOD-E2B-070); Google's iPhone 17 Pro E2B figures (MOD-E2B-065). `ModelInfo` on E2B allocates ~818 MB transiently to parse the main section (MOD-E2B-049); Agello calls `ModelInfo` on every engine load (`ChatEngine.inspect`).
- **litert-lm/file-format.md (Run A, `ModelInfo` section, LM-FMT-037):** add that the context fallback reads the entire main section into memory and refuses sections over 2 GiB, so E4B's file reports `maxContextTokens` 0 while E2B's reports 32,003 (MOD-E2B-049). Also its "Limits" bullet "`-gpu` files in the open build: `Unsupported backend: GPU_ARTISAN`" vs engine-and-sessions.md LM-ENG-002 ("fails with NotFound"): through `EngineFactory::CreateDefault` (the C API path) the NotFound comes first, because no engine type is registered for `GPU_ARTISAN` before any executor is built; the executor-factory error applies only if something constructs the executor directly. Not a wrong claim (LM-FMT-017 says the factory "returns" it), but the user-visible error is NotFound.
- **gemma4-family.md (mod-family):** the card's "unified Keys and Values" statement (MOD-E2B-011); E2B is the only Gemma 4 config with `use_double_wide_mlp: true` (hf-configs.md).
- **patches.md / contributing.md (ours):** third-party #3913 comments confirm `patch_rope.py --split` works on Google's E2B file (b3ca0d2f) and that block 512 fixes E2B on M1 Max and M4 Max (WebGPU); MOD-E2B-061..063. `patch_rope.py` is at agello-litert-lm commit `f2e7afa0`, not at either pin, so it was cited as a web source (GitHub permalink).
- **apple-platform.md / stack.md:** Google's v0.18.0 iOS binary contains `odml::infra::LlmGpuArtisanExecutor` (constructed with an `ml_drift::TransformerConfig` and an `odml::infra::gpu::LlmGpuRunner`) and strings "Constructing legacy EngineImpl...", "EngineAdvancedLegacyImpl", "Loading ArtisanTextDecoder model." (read with `nm -U` and `strings`, 2026-10-15). So the shipped binary appears to include a legacy engine for `GPU_ARTISAN`; not run. MOD-E2B-057.
- **Tooling (claims.py):** `ID_RE = ^([A-Z]+(?:-[A-Z]+)*)-(\d{3})$` rejects prefixes containing digits, so every `MOD-E2B-*` (and `MOD-E4B-*`) ID fails "id must look like LM-ENG-001". With a digit-tolerant regex applied in memory (`[A-Z][A-Z0-9]*` per segment, tool not edited) `claims.py check mod-e2b` reports 0 errors. The tool needs that one-line fix before merge.

## Open questions

- E2B has never run on our build or in Agello: speed, resident memory and correctness on the iOS Metal path are unmeasured; in particular the INT2 decode weights (61 tensors) and inline (non-external) weights through the ML Drift delegate and its weight cache.
- The INT2/INT4/INT8 assignment to individual E2B weights is inferred from exact counts and byte totals; tensor names weren't read (the E2B file isn't local and wasn't downloaded).
- The artisan section's layout, quantization and rotary computation; whether Google's v0.18.0 binary runs `-gpu` files on iOS; what `gpu_artisan_weights_version` does; whether `-web.litertlm` equals `-gpu.litertlm` (same size).
- Google's E2B benchmark table lists a 2,583 MB model while the pinned file is 2,588.1 MB; which revision was benchmarked is unknown.
- No Gemma 4 technical report was found to cite (the Hugging Face paper search was rate-limited); MatFormer nesting of E2B in E4B is not stated in the Gemma 4 config or card.
