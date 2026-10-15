# T5: unified memory (host copies, waits, async)

Apple GPUs share memory with the CPU, but LiteRT's Metal buffer handler treats GPU tensors as
remote: every lock copies into a host shadow, and tensors that aren't "packed" go through ML
Drift's generic conversion. Telemetry (TELEMETRY.md, host copies by buffer size) showed where
that costs time on E4B.

## Step 1: direct copies for linear buffers (LiteRT `agello-next`)

`buffer_handler_metal.mm` reads and writes non-packed tensors straight from and to the shared
`MTLBuffer` when the storage is a linear buffer and the host element type is the tensor's own
(FP16, FP32, 8/16/32-bit integers). Both linear layouts ML Drift uses keep a slice's four channels
together and are affine in the slice, so each row is one `memcpy` (CC4) or one copy per slice, in
the BHWDC order ML Drift's `DataToBHWDC` and `DataFromBHWDC` use, with the same zero padding.
Textures, type conversions and private storage keep the old path. Telemetry labels each copy
`(direct)` or `(converted)`.

The KV caches' slices lie 32 KB apart (a transpose of 4-byte groups), so their rows are copied in
32 x 32 tiles: 3.4x faster than row by row in a microbenchmark (387 against 1,314 µs per 4 MB).

What the old path cost:

| Copy | Old path | Direct |
| --- | --- | --- |
| Logits, MTP verify (2 MB, FP16, 4 rows) | 1.71 ms: a zero-filled copy of the tensor, then an index computation per element | 0.08 ms |
| Logits, decode or drafter (512 KB) | 0.46 ms | 0.016 ms |
| KV cache parameters (28 B, every decode step) | 0.22–0.36 ms: ML Drift's upload makes a new command queue each time | < 0.001 ms |
| KV caches, a session snapshot (`kv_clone`: 48 INT8 caches, 192 MB at 8K) | 439 ms at 8K, about 900 ms at 32K | 62 ms at 8K, 118 ms at 32K |

Results (M6, E4B FP16, 2K prompt, 128 generated, ABAB, 1 warm-up + 3 repetitions per arm,
against the pinned build without the change):

| | Direct reads | Reads and writes | Noise band |
| --- | --- | --- | --- |
| Decode, MTP | +12.7% [+12.0, +13.6] (134.3 → 151.4 tok/s) | +13.6% [+12.2, +14.8] (132.3 → 150.2) | 1.0% / 0.0% |
| Decode, no MTP | +2.0% [+1.9, +2.7] (46.8 → 47.8) | +3.8% [+1.5, +6.0] (45.1 → 46.8) | 0.1% / 2.2% |
| Prefill | −0.2% | −0.9% [−2.3, +0.2] | within noise |

The second run was noisier without MTP; the parameter write alone saved 0.22 ms per decode step
(about 1%). Greedy replies are identical. With MTP, the GPU is busy 93% of the decode turn instead
of 83%, and process CPU per generated token went from 1.63 to 0.85 ms. Agello's Mac inference
suite went from 139 to 119 s, mostly from the faster session snapshots.

Verification: `AGELLO_VERIFY_DIRECT_COPIES=1` runs ML Drift's own conversion after every direct
copy and reports an error anomaly if a byte differs (a write compares the whole GPU buffer,
padding included). Clean on agello-bench (MTP on and off) and on Agello's Mac inference suite:
21/21 scenarios (text, tools, thinking, image, audio, long context, session switches), 10,480
copies, all direct, none different.

## Next

- Snapshots that stay on the GPU and copy only the filled prefix (T3): today's 62 ms at 8K still
  copies every position of every cache to the host.
- Zero-copy locks for shared buffers (return `contents`, no shadow) where the caller allows it.
- `MTLSharedEvent` association and asynchronous runs on Metal (`gpu_async` is unavailable today).
- Asynchronous prefill on Metal (the b/494284915 workaround).
- iPhone run before shipping: `inference`, `probes`, `calibrate`.
