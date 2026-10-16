# Draft pull request (LiteRT): direct host copies for linear Metal buffers

> **Before posting:** Google's LiteRT `bd326784` (2026-10-14) now writes converted data in place on
> unlock for shared Metal buffers. Rework the write rows to show what ours adds: no conversion
> (reference/patches.md, OUR-011).

**Title:** Metal buffer handler: copy linear-buffer tensors straight to and from shared storage

**Body:**

Apple GPUs share memory with the CPU, but `buffer_handler_metal.mm` treats GPU tensors as
remote: tensors that aren't "packed" go through ML Drift's generic conversion on every lock. For
an LLM that runs on every step: the logits, the KV cache parameters, and session snapshots.

When the storage is a linear buffer and the host element type is the tensor's own (FP16, FP32,
8/16/32-bit integers), this PR reads and writes the shared `MTLBuffer` directly. Both linear
layouts ML Drift uses keep a slice's four channels together and are affine in the slice, so each
row is one `memcpy` (CC4) or one copy per slice, in `DataToBHWDC` / `DataFromBHWDC` order with the
same zero padding. The KV caches' slices lie 32 KB apart (a transpose of 4-byte groups), so their
rows are copied in 32 × 32 tiles. Textures, type conversions and private storage keep the old
path.

| Copy (Gemma 4 E4B, M6 Mac) | Today | Direct |
| --- | --- | --- |
| Logits, MTP verify (2 MB, FP16, 4 rows) | 1.71 ms | 0.08 ms |
| Logits, decode or drafter (512 KB) | 0.46 ms | 0.016 ms |
| KV cache parameters (28 B, every decode step) | 0.22–0.36 ms (a new command queue per upload) | < 0.001 ms |
| A session snapshot (48 INT8 caches, 192 MB at 8K) | 439 ms | 62 ms |

Decode with MTP +13.6% (132.3 → 150.2 tok/s), without +3.8%, prefill unchanged; greedy replies
identical (ABAB, 1 warm-up + 3 repetitions per arm).

Verification: an environment switch runs ML Drift's own conversion after every direct copy and
reports any differing byte, padding included. Over the bench and an app's inference suite
(text, tools, thinking, image, audio, long context, session switches): 10,480 copies, all
direct, none different.

Commits on `agello-next` of https://github.com/PowerBeef/LiteRT: `e17213ead`, `584f34610`.
