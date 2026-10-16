# Draft pull request (LiteRT): one weight cache entry for every KV size

**Title:** ML Drift delegate: key the serialization weight cache by content, not tensor sizes

**Body:**

`SerializationWeightCache::GenerateUniqueModelIdentifier` ends in TFLite's
`Serialization::GetFingerprint`, which mixes in the byte sizes of the subgraph's first 100 tensors
and of the delegated partition's inputs and outputs. For an LLM those include the KV caches, so
every `maxNumTokens` an engine is created with gets a new key and writes another copy of the same
converted weights into the model's cache file, which is never trimmed. For Gemma 4 E4B, a cache
folder per context size held about 2.47 GB each: 7.41 GB after loading 8K, 4K and 16K (Agello's
probe 14). An app that sizes the window per
request fills the device.

This PR adds `GenerateContentModelIdentifier`. It keys a subgraph on the model token, the same
options as today, and each delegated node: its index, its builtin op, and each constant input's
index, size and type (`kTfLiteMmapRo`). Those are what the converted weights depend on; KV caches,
masks and activations don't enter the key. The delegate uses it unless
`AGELLO_SHARED_WEIGHT_CACHE=0` (a name to change for upstream; or make it an option). The program
cache is unchanged: it already keeps one entry per program in one file.

Results (Gemma 4 E4B, FP16, 8K → 4K → 16K → 8K; load times on an M6 Mac, cache sizes from
Agello's probe 14):

| | Today | This PR |
| --- | --- | --- |
| Cache on disk | about 2.47 GB per size (one folder each): 7.41 GB after 16K | one folder, 2.47–2.56 GB throughout |
| Load at a size not seen before | 3.6–14 s | 1.2–1.4 s |
| Digit-copy suite at every size | exact | exact |

Test: `content_identifier_test` (the key ignores a KV cache's size, and follows constant input
sizes and the op).

Commit on `agello-next` of https://github.com/PowerBeef/LiteRT: `97e06d36d`.
