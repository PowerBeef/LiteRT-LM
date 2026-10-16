# Gemma 4 E4B: the attention graph (spike S4)

How the `.litertlm` graph attends: where the sliding window comes from, what
each softmax spans, which KV cache each layer reads, the quantization scales,
and how the MTP drafter attends. Read from
`litert-community/gemma-4-E4B-it-litert-lm` rev `2eee7ac3` with
`tools/model-surgery/inspect_e4b.py` (`--layers` prints the per-layer table).
Our split-RoPE copy has the same attention graph.

```sh
~/.venvs/agello-tools/bin/python tools/model-surgery/inspect_e4b.py <model> --signature decode --layers
~/.venvs/agello-tools/bin/python tools/model-surgery/inspect_e4b.py <model> --section tf_lite_mtp_drafter --signature mtp_drafter --layers
```

## The context length is a magic number

Every context-sized dimension is exported as **32003**: the KV caches
(`[1, 2, 32003, 256]` for K, `[1, 2, 256, 32003]` for the transposed V), the
`mask` input, the `cache_size` attribute of `odml.cache_update`, and two
baked INT32 constants in the local-window mask (`0…32002` and `512…32514`).
LiteRT-LM passes the real size to LiteRT as a magic-number configuration
(`runtime/executor/magic_number_configs_helper.h`), and the compiler rewrites
shapes and those constants to the KV capacity T (maxNumTokens). Anything that
changes T rebuilds the graph, which is why each context size gets its own
compiled program and weight cache.

## Signatures

| Section / signature | Queries | Softmax input | Layers | Mask inputs |
| --- | --- | --- | --- | --- |
| `decode` | 1 | `[1, 2, 4, T]` | 0–41 | `input_pos [1]`, `mask [1,1,1,T]` |
| `verify` | 4 (1 + 3 drafts) | `[1, 2, 16, T]` | 0–41 | `input_pos [4]`, `mask [1,1,4,T]` |
| `prefill_1024`, `prefill_128` | 1,024, 128 | `[1, 2, 4096, T]`, `[1, 2, 512, T]` | 0–23 (the KV producers) | `input_pos [N]`, `mask [1,1,N,T]` |
| drafter `mtp_drafter` | 1 | `[1, 2, 2, T]` | 4 blocks | `input_pos [1]`, `mask [1,1,1,T]` |

The softmax's second axis is the 2 KV heads, the third the query heads per KV
head (4 for E4B's 8 query heads, 2 for the drafter's 4) times the queries.

## Masks: local layers make their own window

Each attention ends `SELECT_V2(mask, scores, -1e30)` → `SOFTMAX` → PV.

- **Local layers (35 of 42)** ignore the `mask` input. One mask, computed once
  per graph from `input_pos` and shared by every local layer, keeps key `k`
  when `k ≤ pos` and `k > pos − 512`: `GREATER_EQUAL(pos, iota)`,
  `LESS(pos, iota + 512)`, `LOGICAL_AND`, then broadcast to the query heads.
  So the 512-token window is exact today; what's wasted is the work: QKᵀ,
  the select and the softmax still run over every filled key (the matmuls
  stop at the filled length through `param_tensor`; the select and softmax
  run over all of T).
- **Global layers (5, 11, 17, 23, 29, 35, 41)** use the runtime's `mask`
  input (causal, filled by LiteRT-LM on the host every step).

## Attention per layer (decode; prefill and verify are identical for their layers)

`odml.runtime_bmm` runs both products. QKᵀ (`is_src = false`) reads Q and the
updated K cache; PV (`is_src = true`) reads the probabilities and the
transposed V cache; `scale` is the cache's static INT8 dequantization scale.
`is_global` marks the 7 global layers. `odml.cache_update` (QUANTIZE +
DYNAMIC_UPDATE_SLICE) writes the new K and V with the same scales and has no
ring-buffer attribute in this export.

| Layers | Kind | Reads K / V cache | K scale | V scale | Mask |
| --- | --- | --- | --- | --- | --- |
| 0–4, 6–10, 12–16, 18–22 | local, head dim 256 | its own (0–22) | 0.00568–0.00622, per layer | 0.047244 | window from `input_pos` |
| 5, 11, 17, 23 | global, head dim 512 | its own | 0.001091–0.001147, per layer | 0.017857 | `mask` |
| 24–28, 30–34, 36–40 | local | **cache 22** (layer 22's) | 0.005955 | 0.047244 | window |
| 29, 35, 41 | global | **cache 23** (layer 23's) | 0.001091 | 0.017857 | `mask` |

So 24 caches are written, 20 local (head dim 256, 1 KB per token for K and
V) and 4 global (head dim 512, 2 KB per token): 28,672 bytes per token of
capacity. Layers 24–41 only read. Prefill runs layers 0–22 and the key/value
projection of layer 23, because only they write KV.

## The MTP drafter

- Inputs: `activations [1,1,5120]` (the token embedding and the main model's
  last activations, concatenated on the host), `input_pos`, `mask`,
  `param_tensor`, and the main model's **caches 22 and 23, read only** (no
  `cache_update` in the drafter).
- Four blocks: three local (window from `input_pos`, cache 22) and one global
  (`mask`, cache 23), 4 query heads over the 2 KV heads.
- Outputs: `logits [1,1,262144]` and `projected_activations [1,1,2560]` for
  the next draft.
- Its local blocks dequantize cache 22's keys with 0.005961, layer 4's
  scale, not layer 22's 0.005955 (0.1% off), and its global block uses layer
  5's 0.0010908608 instead of cache 23's 0.0010912281 (0.03% low): an export
  quirk that only affects drafts, which `verify` checks. E2B's drafter has no
  such mismatch; the effect on acceptance is unmeasured.

## What this means for the tracks

- **T4 (attention).** The window is a pure function of the position, so a
  kernel can skip keys outside `(pos − 512, pos]` with no graph change: the
  local layers (35 of 42, plus 3 of the drafter's 4) need at most 512 keys at
  any context length. The select and softmax should stop at the filled
  length like the matmuls do. Prefill is the costliest case: per layer and
  1,024-token chunk the softmax input is 8 × 1,024 × T values, 537 MB in FP16
  at T = 32,768 against 134 MB at 8,192, consistent with prefill slowing from
  790 to 606 tok/s for the same prompt.
- **T10 (ring-buffer local KV).** All 20 local caches, and the drafter's
  reads of cache 22, see only the last 512 positions, so local caches can be
  ring buffers of 512 + chunk positions; global caches stay full size.
  At T = 32,768 the local caches shrink from 671 MB to 31 MB (1,536
  positions for a 1,024-token prefill chunk), leaving the 4 global caches'
  8 KB per token: about 280 MB instead of 918 MB (32,000 allocated
  positions; a 32K setting never allocates 32,768).
  `odml.cache_update` already supports `is_ring_buffer`; the export doesn't
  use it, so this needs a model variant, and the local mask then has to
  follow the ring's slot order.
- **Fused dequantization.** K and V scales are static per layer, so INT8
  attention kernels can take them as constants.
