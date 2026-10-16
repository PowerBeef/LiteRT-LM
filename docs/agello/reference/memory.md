# Memory: where every byte lives

> **Scope:** the memory Gemma 4 E4B and E2B take on Apple devices in this stack, at allocated
> contexts of 4K, 8K, 16K and 32K: the model file (mapped, not read), weights by section and where
> each section runs, the KV cache (formula, bytes per position, what a 32K request really
> allocates), activations and intermediate buffers where the graphs state them, host snapshots of
> the KV cache for kept conversations, the engine caches on disk, iOS limits, and what Agello
> measured, with conditions. Contradictions C1 (the iPhone 17 Pro's app memory limit) and C9 (E4B's
> KV cache at 32K) are resolved here.
> **Out of scope:** the model files' structure ([models/gemma4-e4b.md](models/gemma4-e4b.md),
> [models/gemma4-e2b.md](models/gemma4-e2b.md)); how iOS reports and enforces the limit, the
> entitlements and Metal storage modes in depth ([apple-platform.md](apple-platform.md)); the
> context switch mechanics ([litert-lm/engine-and-sessions.md](litert-lm/engine-and-sessions.md));
> the delegate's caches in depth ([litert/ml-drift-delegate.md](litert/ml-drift-delegate.md));
> ML Drift's memory manager ([ml-drift/metal-backend.md](ml-drift/metal-backend.md)).
> **Pinned to:** LiteRT-LM `7b872544`, LiteRT `362759838`, ML Drift `15fa9e2`; what Agello ships:
> LiteRT-LM `85b1a776` with Google's prebuilt v0.18.0 `CLiteRTLM.xcframework`; Agello `93122b647a`.
> **Applies to:** Apple (iOS, macOS). **Last verified:** 2026-10-15

## Summary

- **The file is mapped, not read.** Each section of the `.litertlm` file is memory-mapped on first
  use (`MAP_PRIVATE`, `MADV_DONTNEED` on Apple). E4B's file is 3.66 GB: 2.31 GB of GPU graphs (the
  decoder and the MTP drafter), 1.01 GB of CPU embedding tables, 0.34 GB of vision and audio
  models. E2B's is 2.59 GB, with more in embedding tables (1.39 GB) than in GPU graphs (0.86 GB).
- **The KV cache is the part that grows with the context:** 28,672 bytes per allocated position
  for E4B, 9,216 for E2B, one bank, INT8. At 32K the runtime allocates 32,000 positions:
  917,504,000 bytes for E4B (C9's "918 MB"), never the 939,524,096 that 32,768 would take.
- **What Agello measured** on the iPhone 17 Pro (our build, nothing else loaded): 1.02, 1.24,
  1.86 and 2.85 GB of footprint right after loading E4B at 4K, 8K, 16K and 32K, and a high-water
  mark of 4.81 GB during the 32K run (an 8,607-token prompt); Google's v0.18.0 binary loads within
  0.26 GB of ours. The footprint grows about 1.0 GB more than the KV cache from 4K to 32K. The first
  image adds about 1.3 GB.
- **The app limit (C1)** on the iPhone 17 Pro with the increased-memory-limit entitlement is 8 GiB
  (shown as "8.6 GB"); readings taken early in a run sometimes return 6 GiB ("6.4 GB").
- **On disk**, our runtime keeps one engine cache folder of about 2.5 GB for every context size;
  Google's writes a new ~2.47 GB copy of the GPU weights per context size, which Agello bounds.

## Concepts

| Term | Meaning here |
| --- | --- |
| Mapped | Part of the process's address space backed by the model file or a cache file; pages are read from storage when touched. |
| Resident | Pages actually in physical memory. |
| Footprint | `phys_footprint`, the quantity iOS's per-app limit applies to; Agello reads it with `task_info(TASK_VM_INFO)` ([apple-platform.md](apple-platform.md#unified-memory-and-the-footprint)). Metal allocations count in it (unified memory). |
| Graphics footprint | The part of the footprint the kernel attributes to the GPU (the "graphics" ledger). |
| Trained context | Positions the model was trained for: 131,072 for E2B and E4B. |
| Export window | The file's context dimension, the magic number 32003, rewritten at load. |
| Allocated context | What `maxNumTokens` becomes after the rewrite: the KV cache's length. |
| Filled length | Tokens actually held in the KV cache (`getTokenCount()`). |
| Engine cache | Files the GPU delegate writes next to the engine: serialized GPU programs and GPU-format weights. |
| KV snapshot | A host copy of a session's KV cache, made when another session takes the executor. |

MB here is 10⁶ bytes and GB 10⁹ bytes, as in the generated tables and Agello's reports; GiB and MiB
are named where they matter.

## How it works

### The model file: mapped, not read

LiteRT-LM's loader maps a section the first time something asks for it, under a mutex, and keeps the
mapping for the engine's life ([`runtime/util/litert_lm_loader.cc:379-400`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.cc#L379-L400)).
<!-- c:LM-FMT-019 --> Each mapping is `mmap(PROT_READ | PROT_WRITE, MAP_PRIVATE)`, followed on Apple by
`madvise(MADV_DONTNEED)` (on other platforms `MADV_WILLNEED`), so pages come from storage only when
touched ([`support/util/memory_mapped_file_posix.cc:132-145`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/support/util/memory_mapped_file_posix.cc#L132-L145)).
<!-- c:APL-028 --> A mapping that fails reports "Failed to map, error: " and `strerror(errno)`,
"Cannot allocate memory" for `ENOMEM`; that is what the extended-virtual-addressing entitlement guards
against ([apple-platform.md](apple-platform.md#the-entitlements)).

Mapped file pages are clean: the system can drop and re-read them. What enters the footprint is
everything the runtime allocates or copies: Metal buffers, host buffers, the KV cache, snapshots.
How each section's bytes end up is the subject of the next section.

### Weights by section

#### E4B

Our E4B file, `gemma-4-E4B-it-fp16-rope.litertlm` at `a1693e22`, is 3,660,726,272 bytes with 12
sections. <!-- c:MOD-E4B-024 --> <!-- c:MOD-E4B-025 --> Grouped by where Agello runs them:

| Section | Size (MB) | Runs on | What's inside |
| --- | --- | --- | --- |
| `tf_lite_prefill_decode` | 2,261.0 | GPU (Metal) | the decoder: 2,197.9 MB of weights after a 63.1 MB graph |
| `tf_lite_mtp_drafter` | 45.3 | GPU (Metal), when MTP is on | the drafter |
| `tf_lite_embedder` | 170.9 | CPU, one row per token | INT2 [262144, 2560] table |
| `tf_lite_per_layer_embedder` | 836.8 | CPU, 42 rows per token | 42 INT2 [262144, 256] tables + per-row parameters |
| `tf_lite_vision_encoder` | 224.1 | CPU in Agello | ViT, INT8 |
| `tf_lite_vision_adapter` | 7.9 | CPU (always) | FP32 projection |
| `tf_lite_audio_encoder_hw` | 94.1 | CPU (required) | conformer, INT2 and INT4 |
| `tf_lite_audio_adapter` | 15.7 | CPU (always) | FP32 projection |
| `SP_Tokenizer`, `LlmMetadataProto`, end-of-media rows | 4.7 | CPU | tokenizer, template, two constant rows |

Totals by placement, summed from the exact section sizes (MB = 10^6 bytes): GPU 2,261.0 + 45.3 =
2,306.4 MB; CPU embedding tables 170.9 + 836.8 = 1,007.7 MB; vision and audio 224.1 + 7.9 + 94.1 +
15.7 = 341.8 MB; with the 4.7 MB tokenizer, 3,660.6 MB of the file's 3,660.7 MB (the rest is the
metadata section, two 10,868-byte end-of-media sections and alignment padding). <!-- c:X-MEM-001 -->

Inside the decoder section, the weights are 1,945.1 MB of INT4 (258 tensors), 167.8 MB of INT2 (the
LM head) and 82.6 MB of INT8 (85 tensors), stored after the graph
([litert/ml-drift-delegate.md](litert/ml-drift-delegate.md#weight-cache)). <!-- c:RT-DLG-059 -->
<!-- c:MOD-E4B-030 --> The tied embedding is stored twice, once for the CPU lookup and once as the GPU
LM head: 335,544,320 bytes of INT2 in all. <!-- c:MOD-E4B-032 --> The per-layer embedder's
836,778,512 bytes are 704,643,072 bytes of INT2 tables and, almost all of the rest, a 4-byte scale
and an 8-byte zero point per table row (132,120,576 bytes). <!-- c:MOD-E4B-010 --> <!-- c:MOD-E4B-011 -->
The vision encoder holds 151.0 MB of INT8 and 65.5 MB of FLOAT32, mostly position-embedding
tables. <!-- c:MOD-E4B-047 --> The audio encoder holds 66.1 MB of INT2, 12.6 MB of INT4 and 12.0 MB
of FLOAT32. <!-- c:MOD-E4B-053 --> Where these come from:
[models/gemma4-e4b.md](models/gemma4-e4b.md#weights-and-quantization).

Agello creates the engine with the main backend on the GPU and `.cpu()` for vision and audio, so
both encoders run on the CPU whenever they are used. <!-- c:LM-EXE-029 --> The adapters run on the
CPU regardless (hard-coded in `EngineSettings::CreateDefault`, and marked `backend_constraint=cpu` in
the file). <!-- c:LM-EXE-028 --> The token and per-layer embedders are compiled for the CPU on every
platform but Android, and every text-token lookup is a CPU model run
([`runtime/components/embedding_lookup/embedding_lookup_text.cc:292-298`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/embedding_lookup/embedding_lookup_text.cc#L292-L298)).
<!-- c:LM-EXE-015 -->

#### E2B

Google's `gemma-4-E2B-it.litertlm` at `b3ca0d2f` is 2,588,147,712 bytes. <!-- c:MOD-E2B-012 -->
Its sections are embedder 103.8 MB, per-layer embedder 1,284.5 MB, audio encoder 94.1 MB, audio
adapter 9.4 MB, vision encoder 224.1 MB, vision adapter 4.7 MB, main graph 818.3 MB and drafter
44.3 MB. <!-- c:MOD-E2B-013 --> By placement, as LiteRT-LM would run it in Agello's configuration:

| Placement | Sections | MB |
| --- | --- | --- |
| GPU | main graph 818.3 + drafter 44.3 | 862.6 |
| CPU embedding tables | embedder 103.8 + per-layer embedder 1,284.5 | 1,388.3 |
| CPU vision and audio | 224.1 + 4.7 + 94.1 + 9.4 | 332.4 |
| Tokenizer | | 4.7 |
| Total (summed from exact section sizes) | | 2,588.0 of 2,588.1 |

<!-- c:X-MEM-002 -->

E2B keeps every section's weights inline in the flatbuffer (777,983,264 bytes in the main graph),
while E4B stores its main weights after the graph. <!-- c:MOD-E2B-014 --> <!-- c:MOD-E2B-070 --> Its
per-layer embedder is larger than E4B's although it has fewer layers, because its 35 tables are INT4
(1,174,405,120 bytes) where E4B's 42 are INT2. <!-- c:MOD-E2B-021 --> Google's E2B card says that for
text-only use the weight footprint "can be as low as 0.8 GB" with the 1.12 GB of embedding parameters
memory-mapped, and that the vision and audio models are loaded on demand (Google's statement, as of
2026-10-15). <!-- c:MOD-E2B-066 --> Agello doesn't offer E2B, and no measurement of it exists in
`docs/agello/results/` ([models/gemma4-e2b.md](models/gemma4-e2b.md#would-e2b-run-in-agello)).

#### Mapped or resident

What the code says about each kind of section on Apple:

- **CPU embedding tables.** The lookup reads only the looked-up row of an INT2 or INT4 table and
  that row's scale. <!-- c:RT-CPU-020 --> Per E4B token that is 2,560 / 4 = 640 bytes of the token
  table and 42 × 256 / 4 = 2,688 bytes of the per-layer tables, 3,328 bytes of INT2 data; a context
  of 4,096 distinct tokens touches at most 4,096 × 3,328 = 13,631,488 bytes of the 872.4 MB of INT2
  table data (pages are larger than rows, so more is mapped in). <!-- c:X-MEM-015 --> These tables
  stay mapped file pages; the rows touched become resident.
- **GPU weights.** On the first load, ML Drift prepares the weights on the GPU, reads each prepared
  tensor back, writes the weight cache file, then loads the file again and re-creates every tensor
  from it, because on Apple file-backed, memory-mapped pages are preferred
  ([`ml_drift_delegate/delegate/delegate_kernel.cc:539-566`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/ml_drift_delegate/delegate/delegate_kernel.cc#L539-L566),
  [`ml_drift_delegate/delegate/delegate_kernel.cc:591-632`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/ml_drift_delegate/delegate/delegate_kernel.cc#L591-L632)).
  <!-- c:RT-DLG-057 --> On a warm load the weights are looked up in the cache as mapped data and
  wrapped in shared Metal buffers with `newBufferWithBytesNoCopy`
  ([`ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.cc:506-530`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.cc#L506-L530),
  [`ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager_metal.h:243-267`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager_metal.h#L243-L267)).
  <!-- c:RT-DLG-058 --> So the decoder's GPU weights are pages of the weight cache file, not anonymous
  memory.
- **The measured footprint agrees.** Right after loading E4B at 4K, our build's footprint on the
  iPhone 17 Pro was 1.02 GB and Google's 1.09 GB, less than the 2,197.9 MB of decoder weights alone,
  so most weight bytes are not in the footprint at that moment. <!-- c:X-MEM-008 --> How many weight
  pages are resident while the GPU runs, and how iOS accounts file-backed pages wrapped as Metal
  buffers, isn't measured (open question).
- **Vision and audio.** Their sections are mapped like the others; what the CPU path allocates for
  them is in [What Agello measured](#images-and-audio).

### The KV cache

#### Bytes per position

E4B's export has 24 INT8 caches with 2 KV heads (layers 24–41 reuse caches 22 and 23): 20 local
caches of head size 256 and 4 global caches (5, 11, 17, 23) of head size 512, keys `[1, 2, T, D]`
and values transposed `[1, 2, D, T]`. One position takes 2 × (20 × (256 + 256) + 4 × (512 + 512)) =
28,672 bytes. <!-- c:MOD-E4B-037 --> <!-- c:LM-EXE-006 --> E2B's has 15 INT8 caches with one KV
head: 12 local of 256 and 3 global of 512, so 12 × 512 + 3 × 1,024 = 9,216 bytes per position.
<!-- c:MOD-E2B-031 --> That is 9/28 of E4B's. <!-- c:MOD-E2B-033 -->

The MTP drafter has no cache of its own: it binds the main model's caches 22 and 23 (13 and 14 for
E2B). <!-- c:LM-EXE-073 --> The delegate stores INT8 (and narrower) caches as byte buffers, one byte
per element, with no dequantize step
([`ml_drift_delegate/delegate/composite/add_values_to_cache_parser.cc:86-120`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/ml_drift_delegate/delegate/composite/add_values_to_cache_parser.cc#L86-L120)).
<!-- c:RT-DLG-023 -->

#### At 4K, 8K, 16K and 32K

| Allocated positions | E4B (28,672 B/position) | E2B (9,216 B/position) |
| --- | --- | --- |
| 4,096 | 117,440,512 (117.4 MB, 112 MiB) | 37,748,736 (37.7 MB) |
| 8,192 | 234,881,024 (234.9 MB, 224 MiB) | 75,497,472 (75.5 MB) |
| 16,384 | 469,762,048 (469.8 MB, 448 MiB) | 150,994,944 (151.0 MB) |
| 32,000 (a 32K request) | 917,504,000 (917.5 MB, 875 MiB) | 294,912,000 (294.9 MB) |
| 32,768 (never allocated with these files) | 939,524,096 (939.5 MB, 896 MiB) | 301,989,888 |

Each row is bytes per position × positions. <!-- c:MOD-E4B-038 --> <!-- c:MOD-E2B-032 -->

Every local cache is as long as the allocated context although its layers read at most the last
512 positions; at 32,000 positions E4B's 20 local caches take 655,360,000 bytes and the 4 global ones
262,144,000. <!-- c:MOD-E4B-081 --> The trained model would need far less: estimated from the
configs with local layers bounded by the 512-token window, E4B's INT8 cache is 44.0 MB at 4,096
tokens and 278.9 MB at 32,768, E2B's 15.7 and 103.8 MB. <!-- c:MOD-FAM-043 --> Only a re-export with
ring-buffer local caches would get there
([models/gemma4-e4b.md](models/gemma4-e4b.md#kv-caches)).

#### C9: 918 MB or 940 MB?

**Resolution.** Both are E4B's 28,672 bytes per position, in decimal megabytes, for different
position counts. LiteRT-LM replaces the file's context magic number 32003 by `maxNumTokens` when it is
between 1 and 32,002, and by the largest multiple of 256 below 32003, 32,000, when it is 32,003 or
more ([`runtime/executor/magic_number_configs_helper.cc:243-271`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/magic_number_configs_helper.cc#L243-L271)),
with only a log warning. <!-- c:LM-EXE-003 --> So a 32K request (`maxNumTokens` 32,768) allocates
32,000 positions, 917,504,000 bytes: AUDIT's 918 MB. The 940 MB of E4B-GRAPH and track T4 is 32,768 ×
28,672 = 939,524,096 bytes, a size this file never gets. <!-- c:MOD-E4B-039 --> The caches exactly as
exported, at 32,003 positions, would be 917,590,016 bytes, which also rounds to 918 MB.
<!-- c:MOD-FAM-047 --> In MiB the two figures are 875 and 896.

#### How it's allocated

- **One bank.** For files with a `param_tensor` input (both Gemma 4 E files), the static executor
  uses the `kGpuOptimizedInplace` policy on the GPU: the caches are created once as GPU output
  buffers, not as two ping-pong banks
  ([`runtime/executor/llm_litert_compiled_model_executor.cc:2165-2180`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_litert_compiled_model_executor.cc#L2165-L2180)).
  <!-- c:LM-EXE-017 -->
- **At load, at full size.** The cache is sized by the allocated context when the engine is created,
  not by the filled length; `clear_kv_cache_before_prefill` (default true) zeroes the first bank once,
  at allocation. <!-- c:LM-EXE-065 -->
- **Unset `maxNumTokens`.** When the setting is 0, the engine takes the file's `max_num_tokens` (the
  E files have none) or the smallest multiple of 4,096 above the prompt hint plus 1,023, which is
  4,096 through the C API. <!-- c:LM-EXE-005 --> Off Apple, a GPU file's own value above 4,096 is
  replaced by that formula; on Apple it is kept. <!-- c:LM-ENG-008 -->
- **Kept after the last session.** Releasing the last session resets the executor's step without
  freeing the KV buffers; they go with the engine. <!-- c:LM-ENG-098 -->

### Activations and intermediate buffers

The graphs state the sizes of the inputs and intermediates that grow with the allocated context
*T*; what ML Drift actually allocates for them after its fusions and memory planning isn't recorded
anywhere.

| Tensor (E4B) | Size | 4,096 | 8,192 | 16,384 | 32,000 |
| --- | --- | --- | --- | --- | --- |
| `prefill_1024` mask, BOOL `[1, 1, 1024, T]` | 1,024 × T bytes | 4,194,304 | 8,388,608 | 16,777,216 | 32,768,000 |
| `decode` mask, BOOL `[1, 1, 1, T]` | T bytes | 4,096 | 8,192 | 16,384 | 32,000 |
| `prefill_1024` softmax input, one of 23 SOFTMAX ops (`decode` has 42); declared FLOAT32 `[1, 2, 4096, 32003]` with T in place of 32003 once loaded, sized here at FP16 | 8 query heads × 1,024 rows × T × 2 bytes = 16,384 × T | 67,108,864 | 134,217,728 | 268,435,456 | 524,288,000 |

<!-- c:X-MEM-003 --> <!-- c:X-MEM-004 --> The executor fills the mask on the CPU, one byte per
allocated position per query row, for every prefill run and plain decode step. <!-- c:LM-EXE-014 -->
At 32,000 positions one `prefill_1024` softmax input is 524,288,000 bytes in FP16 (twice that at the
graph's declared FLOAT32), the
same for E2B's 8 query heads over one KV head as for E4B's 2 × 4. <!-- c:MOD-E2B-035 -->

Fixed-size buffers per step: the decode logits, 262,144 FP16 values (524,288 bytes) read back to the
CPU every step when no GPU sampler is linked, as in our build <!-- c:LM-CMP-031 -->, and 53,248 bytes
of embeddings and per-layer embeddings copied into the decode inputs. <!-- c:LM-EXE-014 -->

How ML Drift holds intermediates on Metal:

- Every `MTLBuffer` it creates is shared storage; standalone textures are private
  ([ml-drift/metal-backend.md](ml-drift/metal-backend.md#memory)). <!-- c:MD-MTL-015 -->
- Buffer-backed intermediates are packed at offsets in one buffer only when that plan totals at most
  128 MiB (and no more than the alternatives); otherwise each shared object gets its own buffer
  ([`ml_drift/metal/memory_manager.mm:320-340`](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/metal/memory_manager.mm#L320-L340)).
  <!-- c:MD-MTL-012 --> At long contexts the softmax tensors above alone exceed 128 MiB.
- The memory manager reuses buffers it already owns across models (the decode, prefill, verify and
  drafter graphs), allocates only when none fits, and keeps every buffer until the manager is
  destroyed. <!-- c:MD-MTL-013 --> So the buffers a long prefill needed stay allocated after it, which
  is consistent with the footprint after a long reply staying above the load figure (measured
  [below](#e4b-by-context-size-iphone-17-pro-idle-test-host)).
- In prefill, ML Drift converts INT4 and INT8 weights per run (`weights_convert_*` nodes: INT8 on
  Apple10, FP16 on older families), and on Apple10 converts the INT8 caches to FP16 for all 46
  attention products of `prefill_1024` (only the 23 key products in `prefill_128`). <!-- c:MD-KRN-002 --> <!-- c:MD-KRN-003 --> <!-- c:MD-KRN-024 --> How much memory those converted copies
  take isn't recorded (open question).
- A host read or write of a Metal tensor through a LiteRT lock allocates a host shadow of the
  tensor's packed size on first lock and copies through it. <!-- c:RT-RUN-026 -->

**The CPU backend (not Agello's default).** Our source build's XNNPACK, before our patch, sized a
per-channel scale array for the INT8 K cache's dequantizing matrix product at T × 4 × (2 × T) bytes,
536,870,912 bytes at T = 8,192 for one cache, where the fix needs 65,536 bytes. <!-- c:RT-CPU-013 -->
Our fork's WORKSPACE comment records 5.5 GB for E4B on the CPU without the patch (no raw record).
<!-- c:RT-CPU-014 --> After the fix, a CPU reply on the iPhone 17 Pro ran with a 3.67 GB footprint
and 4.92 GB still available. <!-- c:RT-CPU-017 --> Agello's notes say Google's binaries include the
fix; the XNNPACK revision inside Google's v0.18.0 binary isn't pinned here, and upstream's open
WORKSPACE pinned the unfixed XNNPACK `d89ef666` until `c05e67a2` (2026-10-14) <!-- c:OUR-012 -->
([litert/cpu-xnnpack.md](litert/cpu-xnnpack.md#open-questions)).

### Host snapshots of KV for kept conversations

One engine has one executor and one live KV cache. When a session that isn't loaded takes the
executor, `ResourceManager::AcquireExecutorWithContextHandler` saves the outgoing session's context
(its token history, runtime state and a deep copy of the KV state) and loads the incoming one
([`runtime/framework/resource_management/resource_manager.cc:745-800`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/framework/resource_management/resource_manager.cc#L745-L800)).
<!-- c:LM-ENG-048 --> <!-- c:LM-ENG-049 --> The details are on
[litert-lm/engine-and-sessions.md](litert-lm/engine-and-sessions.md#context-switches-what-is-copied-and-where);
what matters for memory:

- **Where the copy lives.** In our build, when every state buffer is a shared `MTLBuffer` of about
  the tensor's packed size, the snapshot copies the buffer's bytes as they are into host memory
  ([`runtime/executor/litert/state.cc:305-320`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/litert/state.cc#L305-L320))
  (ours: `aec81cb7`). <!-- c:LM-ENG-051 --> That host memory is an anonymous private mapping of
  whole pages plus a header page, unmapped when the snapshot is freed
  ([`runtime/util/metal_raw_buffer.mm:37-52`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/util/metal_raw_buffer.mm#L37-L52))
  (ours: `585203d3`). <!-- c:LM-ENG-052 --> Upstream copies Metal buffers into new host
  `TensorBuffer`s through a lock instead. <!-- c:LM-ENG-050 -->
- **How big.** A snapshot holds the 48 cache tensors: for E4B at 16,384 positions, 469,762,048 bytes
  of data, plus at most 48 × 64 KiB of buffer slack (the raw path accepts buffers up to 64 KiB
  longer than the tensor) and one header page per buffer, so about 470 to 473 MB. Measured on the
  Mac, the peak with a kept 16K conversation sat 472 MB above the no-helper control after our
  orphan-snapshot change (944 MB before it, two copies). <!-- c:X-MEM-013 --> <!-- c:LM-ENG-060 -->
- **How long.** The kept session's snapshot stays until that session next acquires the executor;
  that acquisition copies it back into the live buffers and frees it. <!-- c:LM-ENG-064 --> During a
  switch toward a session holding a snapshot, the live cache and two snapshots coexist briefly.
  <!-- c:LM-ENG-059 --> Agello `6d389e6` calls `getTokenCount()` on the kept conversation right after
  a helper, which acquires the executor and restores the snapshot at once. <!-- c:LM-ENG-065 -->
- **What it cost.** On the Mac (E4B, 16K, MTP on), the footprint after a helper sat 615 MB above the
  no-helper control with the first raw-snapshot build, 468 MB with the orphan skip and 0 MB with
  mapped snapshots; over 100 helper rounds, the raw-snapshot build crept from 2,796 to 2,910 MB while
  the mapped build stayed at 2,348–2,349 MB. <!-- c:OUR-021 --> <!-- c:OUR-022 --> On the iPhone
  17 Pro, peaks at 16K and 32K fell from 4.02–4.13 GB and 5.88–5.93 GB to 3.59 and 4.93 GB once Agello
  restored the kept conversation right after helpers (`footprintPeakGB`, which is read right after
  the reply, not a high-water mark; these runs include the test host's engine,
  [below](#earlier-reports-a-second-engine-in-the-test-host)); the after run also used a newer
  runtime build (`11beef01e805` instead of `2e54fbdb9ca9`), so the Agello change isn't the only
  difference ([F-078](facts.md#f-078)). <!-- c:LM-ENG-066 -->

Agello keeps the chat across a helper only on a runtime with a build id (ours) and only when
available memory is at least 28,672 × `maxNumTokens` + 400,000,000 bytes: 869,762,048 bytes at
16,384. <!-- c:APL-015 --> At a 32K setting the rule counts 32,768 positions, 22,020,096 bytes more
than the 917,504,000 the runtime allocates. <!-- c:X-MEM-016 --> With Google's v0.18.0 runtime,
Agello releases the chat before a helper and reseeds it afterwards; that runtime's `RestoreState`
moves a restored snapshot in as the live state instead of copying it into the bound buffers.
<!-- c:LM-ENG-055 -->

### Engine caches on disk

#### What the runtime writes

With a cache directory set, the GPU delegate writes two files named after the model file, its
modification time and its size ([`runtime/executor/executor_settings_base.cc:384-430`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/executor_settings_base.cc#L384-L430)):
<!-- c:RT-DLG-044 -->

| File | Contents | One entry per | Grows |
| --- | --- | --- | --- |
| `<model>_<mtime>_<size>_mldrift_program_cache.bin` | serialized GPU graphs, kernels and work groups (MSL source, not compiled code) | subgraph × context size × options | a new entry per context size; re-inserted keys append without reclaiming ([`serialization_program_cache.cc:127-130`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/ml_drift_delegate/delegate/serialization_program_cache/serialization_program_cache.cc#L127-L130)) |
| `<model>_<mtime>_<size>_mldrift_weight_cache.bin` | GPU-format weights | subgraph key (ours: shared by every context size) | upstream: a new copy of the weights per context size |

<!-- c:RT-DLG-062 --> <!-- c:RT-DLG-050 --> <!-- c:MD-MTL-024 --> The program cache key includes the
I/O tensor sizes, KV caches included, so each context size gets its own entry. <!-- c:RT-DLG-047 -->
Upstream's weight cache key includes the same sizes, so every context size wrote a new copy of the
same weights; ours (`97e06d36`) keys weights by the delegated nodes and their constant inputs
([`serialization_weight_cache.cc:531-572`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.cc#L531-L572)).
<!-- c:RT-DLG-055 --> A weight cache written by another ML Drift build of the fully connected weight
code is rejected ("Cache needs to be built again"), but the program cache key has no build or GPU
identity. <!-- c:RT-DLG-052 --> <!-- c:RT-DLG-048 --> A `:nocache` directory disables both files. On the CPU backend, XNNPACK writes its own `.xnnpack_cache` file by the same
naming scheme. <!-- c:RT-CPU-009 -->

#### What Agello keeps

Agello's `EngineCache` (`Agello/Inference/EngineCache.swift:21-130`, Agello `93122b647a`) puts the
caches under `Library/Caches/com.patricedery.agello.litertlm`. With a runtime that reports a build
id (ours) it uses one folder for every context size, `kv-all-<id>`, plus `embedding-<id>`, and
deletes other builds' folders; with Google's binary it uses `kv-<maxNumTokens>` and `embedding` and
keeps the current context size's folder and the most recently used other one. One-off engines use
`:nocache`. <!-- c:X-MEM-014 -->

#### Measured sizes (probe 14)

Agello's probe 14 loads E4B at 8K, 4K, 16K and 8K into the same cache root and records the folders:

| Device, build | After 8K | After 4K | After 16K | After 8K again | Loads (s) |
| --- | --- | --- | --- | --- | --- |
| iPhone 17 Pro, ours `2e54fbdb9ca9`, `kv-all` | 2,470,090,920 | 2,512,115,880 | 2,557,876,392 | 2,557,876,392 | 7.2, 1.5, 1.7, 1.0 |
| Mac mini M6, ours `c5460a024e71`, `kv-all` | 2,469,949,752 | 2,512,140,176 | 2,557,735,224 | 2,557,735,224 | 4.3, 1.3, 1.2, 0.6 |

<!-- c:OUR-023 --> Each new context size grew the shared folder by 42–46 MB (42,024,960 and 45,760,512
bytes on the iPhone), consistent with one more program-cache entry per size; the split of the folder
between the two files isn't recorded. <!-- c:X-MEM-012 --> With the per-size key, three Mac mini M6 runs
recorded probe 14: `probes-t4` (`results/mac/2026-10-14/agello/`, ours at LiteRT `aa05bd2d`, no
build id reported, folders `kv-<size>`), `probes-t4b` (build `b2d0db54134f`: LiteRT-LM `57376f53`,
LiteRT `aa05bd2d+2345ef8020f4`, ML Drift `55196d78+0485bb3ffab7`, as the build records them) and
`probes-p4` (build `c2a5c0a3df2a`: LiteRT-LM `5e946049`, LiteRT `31386317`, ML Drift `2184a7f7`),
both in `results/mac/2026-10-15/agello/`. In each, every context size had its own folder
(`kv-4096`, `kv-8192`, `kv-16384`) of 2.469–2.472 GB; no single `weight_cache.bin` grew. <!-- c:PRC-008 --> The folders
written add up to about 2.47, 4.94 and 7.41 GB after the 8K, 4K and 16K loads <!-- c:PRC-008 -->, but `EngineCache`
kept two at a time <!-- c:X-MEM-014 --> (the 16K row lists only `kv-4096` and `kv-16384`), so 7.41 GB was never on disk
at once. **Measurement gap:** no record loads several
context sizes into one directory under the upstream key, so how one weight cache file grows there
(a section appended per new subgraph key) is not measured.
EmbeddingGemma's folder was 179,974,496 bytes on the iPhone. <!-- c:MOD-EMB-070 -->

### iOS limits

The essentials, from [apple-platform.md](apple-platform.md#memory-on-ios):

- An app whose memory crosses the per-process limit becomes eligible for termination; the jetsam
  report says `per-process-limit`. <!-- c:APL-021 --> `os_proc_available_memory()` is the current
  limit minus the footprint, and Apple says limits can change during the app's life and shouldn't be
  cached. <!-- c:APL-007 --> <!-- c:APL-008 --> Agello's limit is footprint + available, nil when
  available is 0. <!-- c:APL-012 -->
- `com.apple.developer.kernel.increased-memory-limit` (iOS 15+) raises the limit on some device
  models; `com.apple.developer.kernel.extended-virtual-addressing` (iOS 14+) gives a larger address
  space for apps that map large files. Agello declares both. <!-- c:APL-009 --> <!-- c:APL-010 -->
  <!-- c:APL-011 -->
- The iPhone 17 Pro has 12,291,735,552 bytes of physical memory; its 8 GiB limit is 69.9% of it.
  <!-- c:APL-016 --> Metal's `recommendedMaxWorkingSetSize` there is 8,589,950,976 bytes, a
  performance estimate, not the limit. <!-- c:APL-017 -->
- No record measures the limit without the entitlement, on this iPhone or an 8 GB one.
  <!-- c:APL-005 -->

#### C1: 8.6 GB or 6.4 GB?

The iPhone 17 Pro's limit with Agello's entitlement is 8 GiB; early readings can show 6 GiB.
<!-- c:APL-001 --> <!-- c:APL-004 --> The evidence and the resolution are on
[apple-platform.md](apple-platform.md#c1-86-gb-or-64-gb).

What Agello does with it: an 8,192-token default on devices with at least 7·10⁹ bytes of physical
memory, a 32,768 ceiling in Settings when the limit is at least 6.5·10⁹ bytes (16,384 otherwise,
which an early 6 GiB reading triggers), and a reply stopped when less than 400,000,000 bytes remain
available. <!-- c:APL-023 --> <!-- c:APL-013 --> <!-- c:APL-014 --> <!-- c:APL-015 -->

### What Agello measured

All iPhone figures are from `docs/agello/results/iphone/2026-10-15/agello/` (iPhone 17 Pro, A19 Pro,
iOS 27.0.1, E4B `a1693e22` on the GPU with vision, audio and MTP on, FP16 activations, telemetry at
its default level).

#### E4B by context size (iPhone 17 Pro, idle test host)

Agello's calibration loads E4B at each size and sends a long prompt (1,285, 2,302, 4,398 and 8,607
tokens), then a follow-up. `calibrate-host-idle.json` is our build `fc71d704a349`,
`calibrate-google-host-idle.json` Google's v0.18.0 binary, both after the test host stopped loading
its own engine:

| Allocated context | KV cache (MB) | Ours: after load / after the reply (GB) | Ours: high-water mark (MB) | Ours: graphics at the end (MB) | Google's: after load / after the reply (GB) |
| --- | --- | --- | --- | --- | --- |
| 4K | 117.4 | 1.019 / 1.305 | 1,462.2 | 519.7 | 1.095 / 1.334 |
| 8K | 234.9 | 1.245 / 1.258 | 1,531.9 | 870.5 | 0.988 / 1.254 |
| 16K | 469.8 | 1.862 / 1.995 | 2,503.0 | 1,616.5 | 1.648 / 1.981 |
| 32K (32,000) | 917.5 | 2.853 / 3.851 | 4,809.4 | 2,730.0 | 2.862 / 3.856 |

<!-- c:X-MEM-005 --> <!-- c:X-MEM-006 --> "After load" is the footprint the test reads once the
engine is loaded, after Agello's one-token validation; "after the reply" is the report's
`footprintPeakGB`, which despite its name is one footprint reading taken right after the first reply
(`AgelloDeviceTests/CalibrationScenarioTests.swift:33-66`). The high-water mark is our telemetry's
`peak_footprint_mb`, `task_vm_info`'s `ledger_phys_footprint_peak` read when the scenario is recorded
([`agello/telemetry/agello/agello_telemetry.cc:676-690`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/agello/telemetry/agello/agello_telemetry.cc#L676-L690)):
a process-wide maximum, so because it rose in every scenario, each value was reached during that
scenario's load, prompt or follow-up. "Graphics" is the graphics footprint at the same moment.
At 32K, after the first reply, both builds left about 4.7 GB of the 8 GiB available.

From 4K to 32K the footprint after load grows by more than the KV cache:

| Step | Footprint after load | KV cache | Rest |
| --- | --- | --- | --- |
| 4K → 8K | +0.226 GB | +0.117 GB | +0.109 GB |
| 8K → 16K | +0.618 GB | +0.235 GB | +0.383 GB |
| 16K → 32K | +0.991 GB | +0.448 GB | +0.543 GB |
| 4K → 32K | +1.834 GB | +0.800 GB | +1.034 GB |

<!-- c:X-MEM-007 --> The rest grows with the context too; context-sized intermediates (the masks and
attention tensors [above](#activations-and-intermediate-buffers)) are candidates, but nothing records
which buffers they are.

When the calibration reloads at the next size, the log records the previous engine's release: our
build went from 1.15, 1.17 and 1.92 GB to 0.25–0.30 GB, Google's from 1.30, 1.15 and 1.90 GB to
0.26–0.29 GB, with the engine object freed each time. <!-- c:X-MEM-010 -->

#### Images and audio

The memory suite loads E4B at 8K and runs a text, an image, an audio, an 8K-prompt and a 16K turn
(reloading at 16K), on top of ballast that simulates a smaller limit. In the first scenario of the
idle-host runs (5.0 GB target), footprint minus ballast was as follows. Our run had no memory
warnings; Google's had one from the text turn on.

| Step | Ours `fc71d704a349` (ballast 1.442 GB) | Google's v0.18.0 (ballast 3.590 GB) |
| --- | --- | --- |
| Loaded at 8K | 1.292 GB | 1.356 GB |
| After the text turn | 1.347 GB | 1.402 GB |
| After the image turn | 2.639 GB (+1.293) | 2.765 GB (+1.364) |
| After the audio turn | 2.650 GB | 2.749 GB |
| After the 8K prompt | 2.541 GB | 2.655 GB |
| Reloaded at 16K, 16K prompt | 2.171 GB | 2.551 GB |

<!-- c:X-MEM-009 --> The first image adds about 1.3 GB with either runtime, and it stays until the
engine is reloaded. Agello runs the vision encoder and adapter on the CPU (4 threads)
<!-- c:RT-CPU-024 -->; which of the encoder's weights, activations or CPU-side buffers make up the
1.3 GB isn't recorded. Our run's ballast was sized from an early 6 GiB reading, so its effective
limit was 7.147 GB, not 5.0 GB. <!-- c:APL-025 --> How the envelope scenarios ended at 3.5, 4.0 and
5.0 GB: [apple-platform.md](apple-platform.md#the-8-gb-envelope-and-why-three-runs-didnt-test-it).

#### Earlier reports: a second engine in the test host

Until Agello `8bae20b` (2026-10-15), the app hosting the device tests ran its normal UI, which loaded
its own E4B engine next to the tests' engine; at `93122b647a` the app shows nothing while it hosts
tests (`Agello/AgelloApp.swift:23-28`, `:55-63`; its comment says this keeps the app from warming
up a second engine). Google's binary, measured both ways, shows the
difference: after-load footprints of 2.252, 2.638, 2.876 and 4.349 GB at 4K to 32K before
(`calibrate-cooled-google.json`) against 1.095, 0.988, 1.648 and 2.862 GB after. With our build
`fc71d704a349`, after unloading the tests' engine at the start of the 3.5 and 4.0 GB envelope
scenarios, `memory-host-engine-breakdown.json` records 1.41 and 1.85 GB of footprint, 0.76 GB of
it graphics, against 0.12 and 0.49 GB in `memory-host-idle.json`. <!-- c:X-MEM-011 -->
<!-- c:APL-044 --> So every device memory figure recorded before that change, including
`calibrate-ours`, `calibrate-cooled-*`, `memory-ours*`, `memory-google`, `memory-sim-a17*` and
`calibrate-restore-after-helper`, includes the host's engine; comparisons between builds measured
the same way still hold.

#### Mac

The generated engine run (agello-bench, our build, native macOS on the Mac mini M6, E4B at 8K, MTP
on) recorded a 1,556.1 MB footprint, 886.8 MB of it graphics (57.0%). <!-- c:APL-022 --> When the
iPhone app runs on the Mac, its limit is the Mac's physical memory (16 GiB there), so Mac runs don't
test the iPhone's limit. <!-- c:APL-019 -->

#### E2B

Nothing measured by us. Google reports, for an iPhone 17 Pro with its own LiteRT-LM on the GPU at a
2,048-token context, 1,450 MB of "CPU/GPU memory" for E2B and 3,380 MB for E4B (the cards' figures,
as of 2026-10-15; the method isn't stated, and the benchmarked files may be earlier builds).
<!-- c:MOD-E2B-065 --> <!-- c:MOD-FAM-046 --> Those numbers measure something other than the footprint
above.

### A budget for E4B, by context size

Putting the pieces together for E4B on the iPhone 17 Pro with our build (idle host, vision and audio
enabled, MTP on):

| | 4K | 8K | 16K | 32K |
| --- | --- | --- | --- | --- |
| Positions allocated | 4,096 | 8,192 | 16,384 | 32,000 |
| KV cache | 117.4 MB | 234.9 MB | 469.8 MB | 917.5 MB |
| `prefill_1024` mask | 4.2 MB | 8.4 MB | 16.8 MB | 32.8 MB |
| One `prefill_1024` softmax input (FP16) | 67.1 MB | 134.2 MB | 268.4 MB | 524.3 MB |
| Footprint after load (measured) | 1.02 GB | 1.24 GB | 1.86 GB | 2.85 GB |
| High-water mark with a long prompt (measured) | 1.46 GB | 1.53 GB | 2.50 GB | 4.81 GB |
| Kept-conversation snapshot (ours) | 117.4 MB | 234.9 MB | 469.8 MB | 917.5 MB |
| First image (measured at 8K) | | +1.29 GB | | |
| Engine cache on disk (ours, shared) | | about 2.5 GB for all sizes | | |

The derived rows repeat the tables above; the measured ones are single runs at thermal state 0. A
32K engine holding a kept chat and an image during a long prompt hasn't been measured.

## On Apple

| Item | Documented build (agello-next) | Shipped (Google's v0.18.0 binary) |
| --- | --- | --- |
| Model file | mapped, `MADV_DONTNEED` | same code |
| GPU weights | file-backed pages of the weight cache | same delegate design; the LiteRT inside isn't pinned |
| Weight cache key | content key: one copy for every context size | per-size key: one ~2.47 GB folder per context size in Agello |
| Engine cache folders (Agello) | `kv-all-<id>`, `embedding-<id>` | `kv-<n>` (two kept), `embedding` |
| KV snapshots | raw `MTLBuffer` copies into anonymous mappings; restored by copy | snapshot moved in as the live state; Agello doesn't keep the chat |
| Footprint after load, 4K to 32K | 1.02 to 2.85 GB | 0.99 to 2.86 GB |

Everything here is measured on Apple GPUs with unified memory: Metal buffers count in the
footprint, and no separate GPU memory exists to spill to.

## Settings

Rows live in [configuration.md](configuration.md); the ones that move memory:

| Setting | Effect on memory |
| --- | --- |
| `maxNumTokens` | allocated context: KV = 28,672 (E4B) or 9,216 (E2B) bytes per position; 32,003 or more gives 32,000; 0 gives 4,096 through the C API |
| Speculative decoding | maps and compiles the 45.3 MB drafter; no extra KV |
| Vision / audio backend | Agello: CPU; the first image costs about 1.3 GB |
| Activation type | FLOAT32 activations (not used by Agello) were reported to add about 0.45 GB on the iPhone 17 Pro with Google's file <!-- c:MOD-E4B-082 --> |
| Cache directory | `:nocache` disables both delegate caches; a folder per build (Agello) bounds the disk use |
| `ExperimentalFlags.gpuEnableMetalResidencySet` | residency set requested; off in Agello, effect unmeasured |
| Main backend CPU | XNNPACK; needs our XNNPACK patch in our source build |

## Limits and failure modes

| Symptom | Cause | What to do |
| --- | --- | --- |
| A 32K setting allocates less than expected | 32,003 or more becomes 32,000 positions | Expected; plan with 917.5 MB for E4B |
| Footprint after load is ~1.2 GB higher in older device reports | The test host's own engine (before Agello `8bae20b`) | Compare only reports recorded the same way |
| ~1.3 GB jump at the first image | The CPU vision path, held until the engine is reloaded | Budget for it; reload to release |
| The cache folder holds several 2.47 GB copies (Google's runtime) | Per-size weight cache key | Agello keeps two sizes; our runtime shares one copy |
| Footprint stays above the load figure after a long prompt | ML Drift keeps the buffers it allocated until the engine is freed | Free the engine to release them |
| "Cannot allocate memory" when re-creating an engine | Mappings of the previous engine still alive, or address space | Extended virtual addressing; release the old engine first ([troubleshooting.md](troubleshooting.md)) |
| A CPU engine at 8K takes several GB | Unpatched XNNPACK in a source build (whether Google's v0.18.0 binary has the fix isn't pinned: [cpu-xnnpack.md](litert/cpu-xnnpack.md#open-questions)) | Our XNNPACK patch, or a TensorFlow pin past the fix |
| The app limit reads 6.4 GB | An early reading | Re-read later; the limit is 8 GiB |

## Where in the code

| Concept | Symbol | Citation |
| --- | --- | --- |
| Section mapping | `LitertLmLoader::GetSectionBuffer` | [`runtime/util/litert_lm_loader.cc:379-400`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/util/litert_lm_loader.cc#L379-L400) |
| `mmap` and `madvise` | `MemoryMappedFile::Create` | [`support/util/memory_mapped_file_posix.cc:132-145`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/support/util/memory_mapped_file_posix.cc#L132-L145) |
| Embedders on the CPU | `EmbeddingLookupText` | [`runtime/components/embedding_lookup/embedding_lookup_text.cc:292-298`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/embedding_lookup/embedding_lookup_text.cc#L292-L298) |
| Context magic number → 32,000 | `GetTargetNumber` | [`runtime/executor/magic_number_configs_helper.cc:243-271`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/magic_number_configs_helper.cc#L243-L271) |
| KV allocation policy | `LlmLiteRtCompiledModelExecutorStatic::Create` | [`runtime/executor/llm_litert_compiled_model_executor.cc:2165-2180`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/llm_litert_compiled_model_executor.cc#L2165-L2180) |
| INT8 caches as bytes | `AddValuesToCacheOperationParser` | [`ml_drift_delegate/delegate/composite/add_values_to_cache_parser.cc:86-120`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/ml_drift_delegate/delegate/composite/add_values_to_cache_parser.cc#L86-L120) |
| Context switch | `ResourceManager::AcquireExecutorWithContextHandler` | [`runtime/framework/resource_management/resource_manager.cc:745-800`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/framework/resource_management/resource_manager.cc#L745-L800) |
| Raw snapshot (ours) | `LitertState::DeepCopy` | [`runtime/executor/litert/state.cc:305-320`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/executor/litert/state.cc#L305-L320) |
| Snapshot memory (ours) | `AllocateSnapshotMemory` | [`runtime/util/metal_raw_buffer.mm:37-52`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/runtime/util/metal_raw_buffer.mm#L37-L52) |
| Intermediates on Metal | `MemoryManager::AllocateMemoryForBuffers` | [`ml_drift/metal/memory_manager.mm:320-340`](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/metal/memory_manager.mm#L320-L340) |
| Cache file names | `GetProgramCacheFile`, `GetWeightCacheFile` | [`runtime/executor/executor_settings_base.cc:384-430`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/executor_settings_base.cc#L384-L430) |
| Weight cache build and reload | `DelegateKernel` | [`ml_drift_delegate/delegate/delegate_kernel.cc:539-566`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/ml_drift_delegate/delegate/delegate_kernel.cc#L539-L566), [`ml_drift_delegate/delegate/delegate_kernel.cc:591-632`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/ml_drift_delegate/delegate/delegate_kernel.cc#L591-L632) |
| Mapped weights as Metal buffers | `TryRestoringSerializedTensor` | [`ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager_metal.h:243-267`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager_metal.h#L243-L267) |
| Content weight key (ours) | `GenerateContentModelIdentifier` | [`ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.cc:531-572`](https://github.com/PowerBeef/LiteRT/blob/362759838e34e1525f1cc27daddaf663a193c059/ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.cc#L531-L572) |
| Agello's cache folders | `EngineCache` | `Agello/Inference/EngineCache.swift:21-130` (Agello `93122b647a`) |
| Agello's limits and policy | `MemoryProbe`, `DeviceProfile` | `Agello/Support/Diagnostics.swift:14-56`, `Agello/Inference/DeviceProfile.swift:16-77` |
| Idle test host | `AgelloApp.hostsTests` | `Agello/AgelloApp.swift:23-28`, `:55-63` |

## Open questions

- **Resident weights.** How many pages of the weight cache file (2.2 GB of GPU weights) and of the
  1.0 GB of CPU embedding tables are resident during a reply, and how iOS accounts file-backed pages
  wrapped with `newBufferWithBytesNoCopy`. The footprint after load (1.02 GB at 4K) says most aren't
  counted; a `vmmap` or `footprint` capture on the iPhone would say which.
- **The rest beyond the KV cache.** About 1.03 GB grows with the context between 4K and 32K that the
  KV cache doesn't explain. ML Drift's memory plan per graph (intermediate sizes and their sharing)
  would attribute it; no telemetry records it today.
- **Converted weights in prefill.** The `weights_convert_*` nodes' outputs and the FP16 copies of the
  K cache in `prefill_1024` have no recorded size.
- **The first image.** Which part of the ~1.3 GB is the vision encoder's weights in XNNPACK's layout,
  its activations at 2,520 patches, or the CPU side of the engine; and whether a GPU vision encoder
  would cost less.
- **Engine cache split.** How the ~2.5 GB `kv-all` folder divides between the weight cache and the
  program cache, and whether the drafter's weights are in it.
- **E2B on the iPhone.** No measurement; its inline weights and INT2 feed-forward layers are new to
  our GPU path ([models/gemma4-e2b.md](models/gemma4-e2b.md#would-e2b-run-in-agello)).
- **Combined peak.** A 32K engine with a kept chat, an image and a long prompt has not been
  measured.
- **The early 6 GiB reading** and the limit without the entitlement: see
  [apple-platform.md](apple-platform.md#open-questions).
