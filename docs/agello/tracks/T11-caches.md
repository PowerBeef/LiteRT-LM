# T11: one weight copy for every context size

ML Drift's serialization cache keeps the GPU-format weights of a model in one file
(`<model>_…_mldrift_weight_cache.bin`), one entry per delegated subgraph, and the compiled
programs in another (`…_mldrift_program_cache.bin`), one entry per program fingerprint. Agello's
probe 14 found the weight file growing by 2.2 GB for every context size an engine was created
with, and never trimmed, so Agello keeps one cache folder per size (the two most recent).

## P5.1 Why the key depends on the context size

The weight cache's subgraph key (`SerializationWeightCache::GenerateUniqueModelIdentifier`) ends
in TFLite's `Serialization::GetFingerprint(model_token, custom_key, context, delegate_params)`.
That fingerprint combines the model token and the options with two size lists:

- the byte sizes of the subgraph's first 100 tensors, and
- the delegated partition's node indices and the byte sizes of its input and output tensors.

Both lists include the KV caches, whose sizes scale with the window. So 4K, 8K, 16K and 32K
engines get four keys and write four copies of identical weights.

The converted weights depend only on the model, the delegate options (precision, texture
weights, quantized ops, batching, alignment), and each delegated node with its constant inputs.
The program cache, keyed by program fingerprint, already holds an entry per size in one file.

**Decision:** key the weight cache on those inputs only, keep the program cache as it is, and
give Agello one cache folder per runtime build instead of one per size.

## P5.2 Weights keyed by content (LiteRT `agello-next` `97e06d36`)

`SerializationWeightCache::GenerateContentModelIdentifier` keys a subgraph on the model token,
the options (the same custom key as upstream), and each delegated node: its index, its builtin op,
and each constant input's index, size and type (`kTfLiteMmapRo` tensors, the model's weights).
Tensors that aren't constants (KV caches, masks, activations) don't enter the key.
`AGELLO_SHARED_WEIGHT_CACHE=0` brings the upstream key back for A/B runs.
`content_identifier_test`: the key ignores a KV cache's size, and follows the weights' sizes and
the op.

Bench (E4B FP16, M6), one cache folder from empty, `suites/short.json` (15 digit copies) at each
size in turn; `results/mac/2026-10-15/bench/t11-{shared,perkv}.jsonl`:

| Load | Content key: weights / load | Upstream key: weights / load |
| --- | --- | --- |
| 8K (cold) | 2.20 GB / 6.5 s | 2.20 GB / 4.3 s |
| 4K | 2.20 GB / 1.41 s | 4.40 GB / 13.95 s |
| 16K | 2.20 GB / 1.23 s | 6.60 GB / 3.63 s |
| 8K again | 2.20 GB / 0.73 s | 6.60 GB / 0.74 s |

The program cache grows by about 43 MB per new size either way (one entry per program). Every
digit copy is exact at every size with both keys.

Exit criteria: 8K → 4K → 16K → 8K leaves one weight copy (yes); a context-size switch ≤ 1.5 s
(1.2–1.4 s for a size never loaded, 0.7 s for one loaded before).

## P5.3 Agello: one folder

With our runtime (a build id), `EngineCache.directory(maxNumTokens:)` returns one folder for
every size, `kv-all-<id>`, and deletes that build's older per-size folders; with Google's binary
it keeps one folder per size, the two most recently used. Probe 14 still bounds the folders:
at most `keptContextSizes`, each under one weight copy.

Agello on the Mac (Agello `main`, runtime `d1c570842441`): unit 227 (1 skipped), inference 21/21,
probes 11/11, thinking 2/2, files 3/3. Probe 14 (8K → 4K → 16K → 8K from an empty cache): one
folder, `kv-all-d1c570842441`, at 2.47, 2.51, 2.56 and 2.56 GB (one weight copy plus each size's
programs), loads 4.6, 1.3, 1.3 and 0.7 s (`results/mac/2026-10-15/agello/probes-t11.json`).

Still to check on the iPhone: the same probe, and a load at each size after a reboot (the
program cache is per GPU family as well as per size).
